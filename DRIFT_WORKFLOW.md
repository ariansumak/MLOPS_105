# Drift Workflow (Local + Cloud)

This workflow shows how to generate prediction logs and run drift checks locally or in the cloud.

## Option A: Cloud-first (recommended)

### 1) Deploy the inference backend with GCS logging

Set these env vars on the inference service:

- `PNEUMONIA_GCS_BUCKET`: GCS bucket for prediction logs.
- `PNEUMONIA_GCS_PREFIX`: (optional) prefix for logs, default `predictions/`.

Example (local command; for cloud runtime, set the same env vars there):

```bash
PNEUMONIA_GCS_BUCKET="your-bucket" \
PNEUMONIA_GCS_PREFIX="predictions/" \
uv run uvicorn pneumoniaclassifier.api:app --app-dir src --host 0.0.0.0 --port 8000
```

### 2) Send predictions (creates logs automatically)

```bash
curl -F "file=@/path/to/image.jpg" http://127.0.0.1:8000/predict
```

Logs appear in:

```
gs://your-bucket/predictions/prediction_<uuid>.json
```

### 3) Deploy the drift API (reads GCS logs)

Set these env vars on the drift service:

- `PNEUMONIA_GCS_BUCKET`: same bucket as inference logs.
- `PNEUMONIA_GCS_PREFIX`: same prefix as inference logs.
- `PNEUMONIA_REFERENCE_GCS_BUCKET`: bucket containing reference images (optional).
- `PNEUMONIA_REFERENCE_GCS_PREFIX`: prefix for reference images (optional).
- `PNEUMONIA_REFERENCE_REFRESH`: `true` to re-download reference data on startup.

Example:

```bash
PNEUMONIA_GCS_BUCKET="your-bucket" \
PNEUMONIA_GCS_PREFIX="predictions/" \
PNEUMONIA_REFERENCE_GCS_BUCKET="your-bucket" \
PNEUMONIA_REFERENCE_GCS_PREFIX="reference/" \
uv run uvicorn pneumoniaclassifier.monitoring_api:app --app-dir src --host 0.0.0.0 --port 8002
```

### 4) Check drift

```bash
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8002/drift
```

---

## Option B: Local-first (manual upload)

### 1) Run inference locally with local logging

```bash
PNEUMONIA_LOG_DIR="reports/prediction_logs" \
uv run uvicorn pneumoniaclassifier.api:app --app-dir src --host 0.0.0.0 --port 8000
```

### 2) Send predictions

```bash
curl -F "file=@/path/to/image.jpg" http://127.0.0.1:8000/predict
```

### 3) Upload logs to GCS

```bash
gsutil cp reports/prediction_logs/*.json gs://your-bucket/predictions/
```

### 4) Deploy drift API (same as Option A, step 3)

---

## Notes

- Drift API requires logs to exist. If there are no log files, `/drift` will fail.
- Large reference sets can be slow. Use:
  - `PNEUMONIA_REFERENCE_MAX_IMAGES=200`
  - `PNEUMONIA_CURRENT_MAX_LOGS=50`
  - `PNEUMONIA_IMAGE_SIZE=128`
