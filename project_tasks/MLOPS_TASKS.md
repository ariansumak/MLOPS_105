# MLOps Deployment Tasks (M22-M26)

This folder documents the planned artifacts for deployment tasks. Keep additions isolated in their own folders so
parallel work can be merged cleanly.

## M22: FastAPI inference service
- Add `services/fastapi_app/` with `app/main.py` and a minimal router for inference.
- Include request/response models and a model loader that loads the M22 model artifact.
- Add `services/fastapi_app/Dockerfile` and `services/fastapi_app/pyproject.toml` (or reuse root config).
- Add `services/fastapi_app/README.md` with local run instructions.
- Add `configs/` entries or environment variables for model path and device.

## M23: GCP deployment
- Add `infra/gcp/` with one of:
  - Cloud Run: service configuration and deploy script.
  - Cloud Functions: function entrypoint and deployment instructions.
- Add `infra/gcp/README.md` with required GCP setup and IAM roles.
- Add CI job step to deploy on tagged releases or manual workflow.

## M24: API tests and CI
- Add `tests/api/` with pytest tests for health and inference endpoints.
- Add `ci/` or `.github/workflows/` pipeline to run `uv run pytest tests/`.
- Add minimal test data fixtures under `tests/fixtures/`.

## M24: Load testing
- Add `load_tests/` with a load test script (Locust or k6).
- Add `load_tests/README.md` describing test parameters and expected throughput.
- Add CI optional job or local script for smoke load tests.

## M25: Specialized deployment API
- Add `services/onnx_service/` and/or `services/bentoml_service/` for optimized inference.
- Include conversion script to export the M22 model to ONNX.
- Add `README.md` with conversion steps and benchmarking notes.

## M26: Frontend
- Add `frontend/` with a lightweight UI to call the inference endpoint.
- Include `frontend/README.md` with local dev and build instructions.
- Keep static assets within `frontend/public/` and source in `frontend/src/`.

## Version control and collaboration
- Keep large model artifacts out of git; store paths and versions in config files.
- Keep each task in its own folder to avoid merge conflicts.
- Add or update `.gitignore` for generated files and model artifacts.
- Use README files per folder to document local setup and entrypoints.
