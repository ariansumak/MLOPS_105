from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import typer
from PIL import Image

from pneumoniaclassifier.image_features import extract_features
try:
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
    from evidently.legacy.report import Report
    from evidently.legacy.test_suite import TestSuite
    from evidently.legacy.tests import TestShareOfMissingValues
except ImportError as exc:
    raise ImportError("Evidently is required for drift reporting. Install with `uv add evidently`.") from exc

app = typer.Typer(add_completion=False)


def _iter_image_paths(root_dir: Path) -> Iterable[Path]:
    """Yield image paths under the given directory."""

    extensions = {".png", ".jpg", ".jpeg"}
    for path in root_dir.rglob("*"):
        if path.suffix.lower() in extensions and path.is_file():
            yield path


def _build_dataset(root_dir: Path, image_size: int, max_images: int | None) -> pd.DataFrame:
    """Build a feature DataFrame from images arranged in label subfolders."""

    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    rows: list[dict[str, object]] = []
    for index, image_path in enumerate(_iter_image_paths(root_dir)):
        if max_images is not None and index >= max_images:
            break
        label = image_path.parent.name
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((image_size, image_size))
            features = extract_features(image)
        rows.append(
            {
                "image_path": str(image_path),
                "label": label,
                **features,
            }
        )

    if not rows:
        raise ValueError(f"No images found under {root_dir}")

    return pd.DataFrame(rows)


def _write_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path,
    output_name: str,
) -> Path:
    """Generate an Evidently report and save it to HTML."""

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    report.save_html(str(output_path))
    report_json_path = output_dir / "drift_report.json"
    report_json_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    return output_path


def _write_test_summary(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path,
    output_name: str,
) -> Path:
    """Run a small Evidently test suite and save the JSON summary."""

    suite = TestSuite(tests=[TestShareOfMissingValues()])
    suite.run(reference_data=reference, current_data=current)
    summary = suite.as_dict()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


@app.command()
def run(
    reference_dir: Path = typer.Option(
        Path("data/chest_xray/train"),
        help="Reference dataset directory with label subfolders.",
    ),
    current_dir: Path = typer.Option(
        Path("data/chest_xray/test"),
        help="Current dataset directory with label subfolders.",
    ),
    output_dir: Path = typer.Option(Path("reports/drift"), help="Directory for report outputs."),
    image_size: int = typer.Option(224, help="Image size for feature extraction."),
    max_images: int | None = typer.Option(None, help="Max images per dataset (for faster runs)."),
) -> None:
    """Run local drift detection between two image directories."""

    reference_df = _build_dataset(reference_dir, image_size=image_size, max_images=max_images)
    current_df = _build_dataset(current_dir, image_size=image_size, max_images=max_images)
    reference_df = reference_df.drop(columns=["image_path"])
    current_df = current_df.drop(columns=["image_path"])

    report_path = _write_report(reference_df, current_df, output_dir, "drift_report.html")
    summary_path = _write_test_summary(reference_df, current_df, output_dir, "drift_test_summary.json")

    typer.echo(f"Drift report saved to: {report_path}")
    typer.echo(f"Drift test summary saved to: {summary_path}")


if __name__ == "__main__":
    app()
