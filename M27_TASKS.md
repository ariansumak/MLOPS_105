# M27 Data Drift Tasks (Focused Checklist + Repo Gaps)

This file summarizes what to do for the M27 tasks in `REPORT_CHECKLIST.md` and the course exercises page.
It also documents what is currently missing in this repo and a concrete approach to close those gaps.

Source exercises: https://skaftenicki.github.io/dtu_mlops/s8_monitoring/data_drifting/

## Current state in this repo (M27-relevant)

- Inference API exists in `src/pneumoniaclassifier/api.py`.
- Bento service exists in `src/pneumoniaclassifier/bento_service.py`.
- No code for drift detection, Evidently reports, or monitoring service.
- No input/output request logging from the deployed API.
- No GCS bucket integration in the inference API for storing request/response data.
- No Cloud Run/Functions drift monitoring service or deployment config.

## Missing items to complete M27

- Add request/response logging in the inference API (local + cloud storage).
- Add drift detection utilities (Evidently report + programmatic tests).
- Add monitoring API that reads latest prediction data and computes drift.
- Add deployment artifacts for the monitoring API (Dockerfile + Cloud Run config).
- Add a small client script to generate recent prediction data for drift checks.

## What the checklist items require

- Check robustness to data drift (M27)
- Collect input-output data from the deployed app (M27)
- Deploy a drift detection API in the cloud (M27)

## Part A: Local drift detection with Evidently (robustness check) ✅ Done

Goal: demonstrate that you can detect drift between training data and recent predictions.

1) Install Evidently and dependencies.
   - `uv add evidently scikit-learn pandas`

2) Ensure your inference API can save inputs + predictions.
   - Convert the existing inference endpoint to FastAPI (if not already).
   - Add a background task that appends to a CSV (timestamp + inputs + prediction).
   - Example header for iris in the course: `time, sepal_length, sepal_width, petal_length, petal_width, prediction`.
   - For our project, use the pneumonia inputs (image path or extracted features) + predicted class/score.

3) Generate some current data.
   - Call the API multiple times to populate the CSV.

4) Implement drift reporting in a new `data_drift.py`.
   - Load a reference dataset (training data) and the current prediction CSV.
   - Run an Evidently report with `DataDriftPreset` and save HTML.
   - Inspect whether drift is detected.

5) Extend the report for data quality and target drift.
   - Add `DataQualityPreset` and verify it detects injected NaNs.
   - Add `TargetDriftPreset` to check label distribution changes.

6) Add a programmatic drift test.
   - Use Evidently `TestSuite` / `Test` to check for missing values or drift thresholds.
   - Print or log the result (this can be used in CI or scheduled jobs later).

Deliverable: an HTML report and/or programmatic test output showing drift results on our data.

Status in this repo:
- Implemented `src/pneumoniaclassifier/data_drift.py` (feature extraction + Evidently report).
- Implemented `src/pneumoniaclassifier/drift_summary.py` (compact summary from JSON report).
- Generated `reports/drift/drift_report.html`, `reports/drift/drift_report.json`, and `reports/drift/drift_test_summary.json`.

## Part B: Collect input-output data in the cloud

Goal: store live requests and predictions from the deployed API.

### What code needs to be added in this repo

1) Add request/response logging to the inference API.
   - File: `src/pneumoniaclassifier/api.py`
   - Add a small logger function that writes a JSON payload for every `/predict` call.
   - Fields to include: timestamp, model version, input filename (or hash), predicted label, confidence, probabilities.
   - Keep local logging optional via `PNEUMONIA_LOG_DIR`.

2) Add GCS upload support.
   - New helper module: `src/pneumoniaclassifier/gcs_logging.py` (or similar).
   - Use `google-cloud-storage` to write JSON files to a bucket.
   - Env vars: `PNEUMONIA_GCS_BUCKET`, `PNEUMONIA_GCS_PREFIX`.

3) Add a small client to generate requests (optional but useful).
   - File: `src/pneumoniaclassifier/generate_predictions.py`.
   - Sends sample images to the `/predict` endpoint to populate the bucket.

Status in this repo:
- Added `src/pneumoniaclassifier/prediction_logging.py` to log prediction inputs/outputs.
- Wired logging into `src/pneumoniaclassifier/api.py` using FastAPI background tasks.
- Added Dockerfile for the prediction API: `dockerfiles/api.dockerfile`.

1) Create a GCP bucket for monitoring data.
   - Upload the model and any reference data you need.

2) Modify the deployed API to:
   - Load the model from the bucket at startup.
   - Save every request + prediction back to the bucket under a unique filename.
   - Use `google-cloud-storage` (`uv add google-cloud-storage`).

3) Validate locally, then in the cloud.
   - Run with `uvicorn` locally and use a small client script to send requests.
   - Confirm the request/response JSONs appear in the bucket.

4) Containerize and deploy.
   - Write a Dockerfile for the API.
   - Deploy to Cloud Run (or Functions) and re-test data collection.

Deliverable: bucket contains a growing set of request/response files from the deployed app.

## Part C: Deploy a drift detection API in the cloud

Goal: a monitoring service that reads reference data + latest predictions and computes drift.

1) Create a monitoring FastAPI app.
   - Reads reference data (training set) and the latest N predictions from the bucket.
   - Runs Evidently drift checks and returns a report (HTML or JSON).

2) Implement missing pieces.
   - A function like `fetch_latest_data(n: int)` to pull the latest prediction files.

3) Test locally with a small sample of data from the bucket.

4) Containerize and deploy the monitoring service.
   - Build Docker image.
   - Deploy to Cloud Run (or Functions) as a separate service.

5) Generate drift over time.
   - Run a client that calls the prediction API repeatedly to simulate drift.
   - Re-run the monitoring API and show drift indicators change.

Deliverable: a deployed monitoring endpoint that detects drift from real request logs.

Status in this repo:
- Added `src/pneumoniaclassifier/monitoring_api.py` for drift checks.
- Added Dockerfile for the monitoring API: `dockerfiles/drift_monitor.dockerfile`.

## Concrete repo approach (what to add and where)

1) Extend the inference API to log requests and predictions.
   - File: `src/pneumoniaclassifier/api.py`
   - Add a `PNEUMONIA_LOG_DIR` env var for local CSV/JSON logging.
   - Add optional GCS logging with `google-cloud-storage` and `PNEUMONIA_GCS_BUCKET`.
   - Log: timestamp, model version, input filename (or hash), predicted label, confidence.

2) Add a drift reporting script.
   - New file: `src/pneumoniaclassifier/data_drift.py`
   - Inputs:
     - Reference dataset (training data) on disk.
     - Current data from logged predictions (CSV/JSON).
   - Outputs:
     - `reports/drift_report.html`
     - Optional JSON summary for CI checks.

3) Add a monitoring API for drift checks.
   - New folder: `services/drift_monitor/` or `src/pneumoniaclassifier/monitoring_api.py`
   - FastAPI endpoint `/drift` that:
     - Reads latest N predictions (local or GCS).
     - Runs Evidently report or test suite.
     - Returns summary metrics and/or HTML report location.

4) Add Docker + Cloud Run deploy artifacts for monitoring.
   - New file: `dockerfiles/drift_monitor.Dockerfile`
   - New file: `cloudbuild.yaml` update or a separate build config.
   - Deploy to Cloud Run as a separate service.

5) Add a small client to generate recent prediction data.
   - New file: `src/pneumoniaclassifier/generate_predictions.py`
   - Sends sample images to `/predict` and logs results (or directly calls inference).

6) Update docs / report evidence.
   - Add screenshots of drift report + GCS logging to the report.
   - Note monitoring endpoint URL and deployment steps.

## Evidence to include in the report

- Drift HTML report screenshot(s) or summary stats.
- Confirmation that requests/predictions are stored in GCS.
- URL or description of the monitoring service deployment (Cloud Run/Functions).
