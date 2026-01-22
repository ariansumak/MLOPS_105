# M27 Drift Report: How to Run and Interpret

This guide explains how to generate the drift report and how to interpret the main results.

## 1) Generate the report

Run drift detection with train as reference and test as current (baseline):

```bash
uv run python -m pneumoniaclassifier.data_drift \
  --reference-dir data/chest_xray/train \
  --current-dir data/chest_xray/test \
  --output-dir reports/drift \
  --max-images 200
```

Outputs:
- `reports/drift/drift_report.html` (full report)
- `reports/drift/drift_report.json` (machine‑readable summary)
- `reports/drift/drift_test_summary.json` (missing‑values test)

## 2) Quick summary (recommended)

```bash
uv run python -m pneumoniaclassifier.drift_summary --top-k 10
```

This prints the key drift metrics and the top drifted columns.

## 3) What the “columns” are

The script turns each image into numeric features + label:

- `mean_r`, `mean_g`, `mean_b`: mean pixel intensity per RGB channel
- `std_r`, `std_g`, `std_b`: standard deviation per RGB channel
- `brightness`: average grayscale intensity
- `contrast`: grayscale standard deviation
- `entropy`: grayscale histogram entropy
- `label`: class name from the folder (e.g., NORMAL / PNEUMONIA)

These columns are what Evidently compares between reference and current.

## 4) Key values to interpret

### Dataset drift detected
- **`dataset_drift_detected`** is a boolean.
- It becomes **True** when the share of drifted columns exceeds the **drift share threshold**.

### Share of drifted columns
- **`share_of_drifted_columns`** = drifted columns / total columns.
- Example: 4 drifted out of 10 → 0.4.

### Drift share threshold
- **`drift_share_threshold`** is the cutoff Evidently uses.
- If share_of_drifted_columns ≥ drift_share_threshold, dataset drift is detected.

### Drifted columns list
- Lists which columns drifted and which statistical test detected the drift.
- Use this to see *which* features changed and how strongly.

## 5) How to interpret the numbers

Rule of thumb (adjust to your project):

- **0.0–0.2**: low drift → monitor
- **0.2–0.4**: moderate drift → investigate
- **> 0.5**: high drift → consider retraining or at least re‑evaluation

Important: drift alone does not guarantee worse model performance. Use drift as a signal to validate the model on recent data or retrain if performance drops.

## 6) Where to look in the HTML report

In `drift_report.html`, focus on:

- **Dataset Drift** section: shows the overall drift decision.
- **Drifted Columns Table**: shows which columns drifted + tests.
- **Target Drift**: check label distribution shifts.

## 7) Common pitfalls

- If your “current” data is just test data, drift is only a baseline check.
- Real drift detection requires live prediction logs (from deployment).
- If you include file paths as a feature, drift will always trigger (paths change). The script excludes `image_path` automatically.

## 8) Evidence for REPORT_CHECKLIST.md

For M27 you can include:

- A screenshot of `drift_report.html` showing the drift summary.
- The output of `drift_summary` as a short text snippet.
- A sentence on how you interpret the drift share and what action you would take.
