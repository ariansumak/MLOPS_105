from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


def _load_report(report_path: Path) -> dict:
    """Load the drift report JSON."""

    if not report_path.exists():
        raise FileNotFoundError(f"Report JSON not found at {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _find_metric(report: dict, metric_name: str) -> dict | None:
    """Find a metric result entry by name."""

    for metric in report.get("metrics", []):
        if metric.get("metric") == metric_name:
            return metric.get("result", {})
    return None


def _summarize_drift(report: dict, top_k: int) -> dict[str, object]:
    """Extract key drift statistics from an Evidently report."""

    dataset_drift = _find_metric(report, "DatasetDriftMetric")
    drift_table = _find_metric(report, "DataDriftTable")

    summary: dict[str, object] = {}
    if dataset_drift:
        summary["dataset_drift_detected"] = dataset_drift.get("dataset_drift")
        summary["drift_share_threshold"] = dataset_drift.get("drift_share")
        summary["number_of_columns"] = dataset_drift.get("number_of_columns")
        summary["number_of_drifted_columns"] = dataset_drift.get("number_of_drifted_columns")
        summary["share_of_drifted_columns"] = dataset_drift.get("share_of_drifted_columns")

    drifted_columns: list[dict[str, object]] = []
    if drift_table:
        for column in drift_table.get("drift_by_columns", {}).values():
            if column.get("drift_detected"):
                drifted_columns.append(
                    {
                        "column_name": column.get("column_name"),
                        "column_type": column.get("column_type"),
                        "drift_score": column.get("drift_score"),
                        "stattest_name": column.get("stattest_name"),
                        "stattest_threshold": column.get("stattest_threshold"),
                    }
                )
    drifted_columns = sorted(
        drifted_columns, key=lambda item: (item.get("drift_score") is None, item.get("drift_score"))
    )
    summary["drifted_columns"] = drifted_columns[:top_k]
    summary["drifted_columns_total"] = len(drifted_columns)
    return summary


def _format_summary(summary: dict[str, object]) -> str:
    """Format a readable summary of drift results."""

    lines: list[str] = []
    lines.append("Drift summary")
    lines.append("=================")
    lines.append(f"Dataset drift detected: {summary.get('dataset_drift_detected')}")
    lines.append(f"Drift share threshold: {summary.get('drift_share_threshold')}")
    lines.append(f"Number of columns: {summary.get('number_of_columns')}")
    lines.append(f"Number of drifted columns: {summary.get('number_of_drifted_columns')}")
    lines.append(f"Share of drifted columns: {summary.get('share_of_drifted_columns')}")
    lines.append("")
    lines.append("Top drifted columns")
    lines.append("-------------------")
    for item in summary.get("drifted_columns", []):
        lines.append(
            f"- {item.get('column_name')} ({item.get('column_type')}): "
            f"score={item.get('drift_score')}, "
            f"test={item.get('stattest_name')}, "
            f"threshold={item.get('stattest_threshold')}"
        )
    if not summary.get("drifted_columns"):
        lines.append("- None detected.")
    return "\n".join(lines)


@app.command()
def run(
    report_path: Path = typer.Option(
        Path("reports/drift/drift_report.json"),
        help="Path to the Evidently drift report JSON.",
    ),
    output_path: Path | None = typer.Option(
        None,
        help="Optional output path for a text summary.",
    ),
    top_k: int = typer.Option(10, help="Number of drifted columns to display."),
) -> None:
    """Summarize the key drift values from an Evidently report JSON file."""

    report = _load_report(report_path)
    summary = _summarize_drift(report, top_k=top_k)
    text = _format_summary(summary)

    typer.echo(text)
    if output_path is not None:
        output_path.write_text(text, encoding="utf-8")
        typer.echo(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    app()
