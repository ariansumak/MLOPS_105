# pneumoniaclassifier

End-to-end project for training, serving, and monitoring a pneumonia classifier. This README focuses on how to run
everything locally and in the cloud.

## Architecture

```txt
Frontend (Django) --> FastAPI Inference --> Prediction logs (local or GCS)
                                  |
                                  v
                        Drift API (Evidently)
                                  |
                     Reference images (train set)
```


## Framework map

| Layer | Tools |
| --- | --- |
| Training / Experimentation | PyTorch, Torchvision, Hydra, OmegaConf, W&B |
| Serving | FastAPI, Uvicorn, Django (frontend), BentoML (optional) |
| Monitoring / Drift | Evidently |
| MLOps / Infra | Docker, DVC (GCS), Google Cloud (Cloud Run, GCS) |

Local mode:
- Reference images live on disk (default: `data/chest_xray/train`).
- Predictions logs are written to `reports/prediction_logs`.
- Drift reports are written to `reports/drift_monitor`.

Cloud mode:
- Prediction logs are written to `gs://<bucket>/predictions/`.
- Reference images are stored in `gs://<bucket>/reference/`.
- Drift service downloads reference images to `/tmp/reference` and writes reports to `/tmp/drift_monitor`.

## Quickstart (local)

### 1) Train and save a checkpoint

```bash
uv run python src/pneumoniaclassifier/train.py
```

Default checkpoint path: `models/m22_model.pt`. Update `configs/inference.yaml` or set `PNEUMONIA_CONFIG` if needed.

### 2) Run inference API (FastAPI)

```bash
PNEUMONIA_LOG_DIR="reports/prediction_logs" \
uv run uvicorn pneumoniaclassifier.api:app --app-dir src --host 0.0.0.0 --port 8000
```

### 3) Run frontend (Django)

```bash
uv run python frontend_django/manage.py runserver 0.0.0.0:8001
```

Open `http://localhost:8001` and point the UI to `http://localhost:8000`.

### 4) Generate predictions

```bash
curl -F "file=@/path/to/image.jpg" http://127.0.0.1:8000/predict
```

### 5) Run drift locally

```bash
PNEUMONIA_LOG_DIR="reports/prediction_logs" \
uv run uvicorn pneumoniaclassifier.monitoring_api:app --app-dir src --host 0.0.0.0 --port 8002
```

```bash
curl http://127.0.0.1:8002/drift
```

Reports appear in `reports/drift_monitor`.

## Cloud workflow (GCS + Cloud Run)

### 1) Upload reference images to GCS

```bash
gsutil -m cp -r data/chest_xray/train gs://<bucket>/reference/
```

### 2) Deploy inference (Cloud Run)

Environment variables:
- `PNEUMONIA_GCS_BUCKET=<bucket>`
- `PNEUMONIA_GCS_PREFIX=predictions/`

### 3) Deploy drift service (Cloud Run)

Environment variables:
- `PNEUMONIA_GCS_BUCKET=<bucket>`
- `PNEUMONIA_GCS_PREFIX=predictions/`
- `PNEUMONIA_REFERENCE_GCS_BUCKET=<bucket>`
- `PNEUMONIA_REFERENCE_GCS_PREFIX=reference/`
- `PNEUMONIA_REFERENCE_REFRESH=true`
- `PNEUMONIA_REFERENCE_DIR=/tmp/reference`
- `PNEUMONIA_REPORT_DIR=/tmp/drift_monitor`

Optional performance caps:
- `PNEUMONIA_REFERENCE_MAX_IMAGES=200`
- `PNEUMONIA_CURRENT_MAX_LOGS=50`
- `PNEUMONIA_IMAGE_SIZE=128`

IAM notes:
- Inference service account needs `roles/storage.objectCreator`.
- Drift service account needs `roles/storage.objectViewer`.

### 4) Run drift in the cloud

Call the drift endpoint:
```bash
curl https://<drift-service-url>/drift
```

## Docker

### Build and run inference locally

```bash
docker build -f dockerfiles/api.dockerfile -t pneumonia-api:latest .
docker run --rm -p 8000:8000 \
  -e PORT=8000 \
  -e PNEUMONIA_LOG_DIR=/app/reports/prediction_logs \
  pneumonia-api:latest
```

### Build and run drift locally

```bash
docker build -f dockerfiles/drift_monitor.dockerfile -t pneumonia-drift:latest .
docker run --rm -p 8002:8002 \
  -e PORT=8002 \
  -e PNEUMONIA_LOG_DIR=/app/reports/prediction_logs \
  pneumonia-drift:latest
```

## Training (Docker)

```bash
docker build -f dockerfiles/train.dockerfile -t pneumonia-train:latest .
docker run --rm \
  -v $(pwd)/data/chest_xray:/data \
  pneumonia-train \
  data.data_dir=/data \
  train.epochs=1 \
  train.device=cpu \
  wandb.enabled=false
```

## Project structure

```txt
configs/                Configuration files
data/                   Dataset root (expected: chest_xray/train|val|test)
dockerfiles/            Dockerfiles for api, drift monitor, training
frontend/               Static frontend
frontend_django/        Django UI
models/                 Model checkpoints
reports/                Logs and drift reports
src/pneumoniaclassifier Core package (training, inference, API, drift)
tests/                  Unit tests
vertex_ai/              Vertex AI configs
```
