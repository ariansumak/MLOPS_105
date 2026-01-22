from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

try:
    from google.cloud import storage
except ImportError:
    storage = None

try:
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.legacy.report import Report
except ImportError as exc:
    raise ImportError("Evidently is required for drift reporting. Install with `uv add evidently`.") from exc

from pneumoniaclassifier.data_drift import _build_dataset


def _get_bucket_name() -> str | None:
    """Return the configured GCS bucket name."""

    return os.getenv("PNEUMONIA_GCS_BUCKET")


def _get_prefix() -> str:
    """Return the GCS prefix for prediction logs."""

    return os.getenv("PNEUMONIA_GCS_PREFIX", "predictions/")


def _get_current_limit() -> int:
    """Return the max number of log files to load."""

    return int(os.getenv("PNEUMONIA_CURRENT_MAX_LOGS", "200"))


def _get_reference_dir() -> Path:
    """Return the reference image directory."""

    return Path(os.getenv("PNEUMONIA_REFERENCE_DIR", "data/chest_xray/train"))


def _get_reference_max_images() -> int | None:
    """Return optional cap on reference images."""

    value = os.getenv("PNEUMONIA_REFERENCE_MAX_IMAGES")
    return int(value) if value else None


def _get_report_dir() -> Path:
    """Return the directory for saved drift reports."""

    return Path(os.getenv("PNEUMONIA_REPORT_DIR", "reports/drift_monitor"))


def _get_image_size() -> int:
    """Return the image size used for feature extraction."""

    return int(os.getenv("PNEUMONIA_IMAGE_SIZE", "224"))


def _load_gcs_logs(bucket_name: str, prefix: str, limit: int) -> list[dict[str, Any]]:
    """Load the most recent prediction log files from GCS."""

    if storage is None:
        raise ImportError(
            "google-cloud-storage is required for GCS logging. "
            "Install with `uv add google-cloud-storage`."
        )
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    blobs = sorted(blobs, key=lambda blob: blob.updated or datetime.min, reverse=True)
    records: list[dict[str, Any]] = []
    for blob in blobs[:limit]:
        payload = blob.download_as_text()
        records.append(json.loads(payload))
    return records


def _load_local_logs(log_dir: Path, limit: int) -> list[dict[str, Any]]:
    """Load the most recent prediction logs from a local folder."""

    if not log_dir.exists():
        raise FileNotFoundError(f"Local log dir not found: {log_dir}")
    files = sorted(log_dir.glob("prediction_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    records: list[dict[str, Any]] = []
    for path in files[:limit]:
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _logs_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert prediction logs to a DataFrame."""

    rows: list[dict[str, Any]] = []
    for record in records:
        features = record.get("features", {}) or {}
        probabilities = record.get("probabilities", {}) or {}
        row = {
            "label": record.get("label"),
            "confidence": record.get("confidence"),
            **{f"prob_{key}": value for key, value in probabilities.items()},
            **features,
        }
        rows.append(row)
    if not rows:
        raise ValueError("No prediction logs found to build current dataset.")
    return pd.DataFrame(rows)


def _build_reference_dataset() -> pd.DataFrame:
    """Build the reference dataset from images on disk."""

    reference_dir = _get_reference_dir()
    max_images = _get_reference_max_images()
    reference_df = _build_dataset(reference_dir, image_size=_get_image_size(), max_images=max_images)
    reference_df = reference_df.drop(columns=["image_path"])
    return reference_df


def _build_current_dataset() -> pd.DataFrame:
    """Build the current dataset from prediction logs."""

    limit = _get_current_limit()
    bucket_name = _get_bucket_name()
    if bucket_name:
        records = _load_gcs_logs(bucket_name, _get_prefix(), limit)
    else:
        log_dir = Path(os.getenv("PNEUMONIA_LOG_DIR", "reports/prediction_logs"))
        records = _load_local_logs(log_dir, limit)
    return _logs_to_dataframe(records)


def _run_report(reference: pd.DataFrame, current: pd.DataFrame, report_dir: Path) -> dict[str, Any]:
    """Run drift report and return a compact summary."""

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / "monitoring_drift_report.html"
    json_path = report_dir / "monitoring_drift_report.json"
    report.save_html(str(html_path))
    report_data = report.as_dict()
    json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    summary = {
        "report_html": str(html_path),
        "report_json": str(json_path),
    }
    for metric in report_data.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            summary.update(metric.get("result", {}))
    return summary


def create_app() -> FastAPI:
    """Create the drift monitoring API."""

    app = FastAPI(title="Pneumonia Drift Monitor", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return service health status."""

        return {"status": "ok"}

    @app.get("/drift")
    def drift() -> dict[str, Any]:
        """Compute drift between reference images and recent predictions."""

        try:
            reference_df = _build_reference_dataset()
            current_df = _build_current_dataset()
            report_dir = _get_report_dir()
            return _run_report(reference_df, current_df, report_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
