from __future__ import annotations

import io
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from pneumoniaclassifier.image_features import extract_features

try:
    from google.cloud import storage
except ImportError:
    storage = None


@dataclass(frozen=True)
class PredictionLogEntry:
    """Schema for a prediction log entry."""

    timestamp: str
    filename: str | None
    content_hash: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    features: dict[str, float]
    model_name: str | None
    checkpoint_path: str | None


_GCS_CLIENT: storage.Client | None = None
_GCS_BUCKET_NAME: str | None = None
_GCS_BUCKET: storage.Bucket | None = None


def _get_timestamp() -> str:
    """Return a UTC timestamp in ISO format."""

    return datetime.now(tz=timezone.utc).isoformat()


def _hash_content(content: bytes) -> str:
    """Return a SHA-256 hash for content bytes."""

    return sha256(content).hexdigest()


def _get_local_log_dir() -> Path | None:
    """Resolve the local logging directory from env."""

    log_dir = os.getenv("PNEUMONIA_LOG_DIR")
    if not log_dir:
        return None
    return Path(log_dir)


def _get_gcs_bucket() -> storage.Bucket | None:
    """Return the configured GCS bucket, if available."""

    global _GCS_CLIENT, _GCS_BUCKET_NAME, _GCS_BUCKET
    bucket_name = os.getenv("PNEUMONIA_GCS_BUCKET")
    if not bucket_name:
        return None
    if storage is None:
        raise ImportError(
            "google-cloud-storage is required for GCS logging. "
            "Install with `uv add google-cloud-storage`."
        )
    if _GCS_BUCKET is not None and _GCS_BUCKET_NAME == bucket_name:
        return _GCS_BUCKET

    _GCS_CLIENT = storage.Client()
    _GCS_BUCKET_NAME = bucket_name
    _GCS_BUCKET = _GCS_CLIENT.bucket(bucket_name)
    return _GCS_BUCKET


def _get_gcs_prefix() -> str:
    """Return the GCS prefix for prediction logs."""

    return os.getenv("PNEUMONIA_GCS_PREFIX", "predictions/")


def _serialize_entry(entry: PredictionLogEntry) -> str:
    """Serialize the log entry to JSON."""

    return json.dumps(asdict(entry), ensure_ascii=True, indent=2)


def _write_local(entry: PredictionLogEntry) -> Path | None:
    """Write the log entry to the local filesystem."""

    log_dir = _get_local_log_dir()
    if log_dir is None:
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"prediction_{uuid4().hex}.json"
    path = log_dir / file_name
    path.write_text(_serialize_entry(entry), encoding="utf-8")
    return path


def _write_gcs(entry: PredictionLogEntry) -> str | None:
    """Upload the log entry to GCS."""

    bucket = _get_gcs_bucket()
    if bucket is None:
        return None
    prefix = _get_gcs_prefix()
    file_name = f"{prefix.rstrip('/')}/prediction_{uuid4().hex}.json"
    blob = bucket.blob(file_name)
    blob.upload_from_string(_serialize_entry(entry), content_type="application/json")
    return file_name


def log_prediction(
    *,
    content: bytes,
    label: str,
    confidence: float,
    probabilities: dict[str, float],
    filename: str | None,
    model_name: str | None,
    checkpoint_path: str | None,
    image_size: int | None,
) -> dict[str, Any]:
    """Log a prediction to local storage and/or GCS."""

    image = Image.open(io.BytesIO(content)).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size, image_size))
    features = extract_features(image)

    entry = PredictionLogEntry(
        timestamp=_get_timestamp(),
        filename=filename,
        content_hash=_hash_content(content),
        label=label,
        confidence=confidence,
        probabilities=probabilities,
        features=features,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
    )

    result: dict[str, Any] = {
        "local_path": None,
        "gcs_path": None,
    }

    try:
        result["local_path"] = str(_write_local(entry)) if _get_local_log_dir() else None
    except Exception:
        result["local_path"] = None

    try:
        result["gcs_path"] = _write_gcs(entry)
    except Exception:
        result["gcs_path"] = None

    return result
