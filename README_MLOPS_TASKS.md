# M22 to M26 status and remaining work

This README focuses only on milestones M22 to M26 and what still needs to be done.

## M22 - FastAPI inference application

Status: done

Evidence
- API implementation: `src/pneumoniaclassifier/api.py`
- Tests target API module: `tests/api/test_api.py`

Notes
- The API exposes health and prediction endpoints and loads model + config.

## M23 - Deploy in GCP (Functions or Cloud Run)

Status: pending

What is missing
- A deployed service URL in GCP (Cloud Run or Functions).
- Deployment artifact and build configuration that produces a runnable container.
- A short usage snippet (curl or similar) to show the deployed API invocation.

Suggested completion path
1) Build a container for the API.
2) Push it to Artifact Registry.
3) Deploy to Cloud Run and record the service URL.
4) Add a short invocation example to the report.

## M24 - API tests and load tests

Status: in progress

What exists
- API unit/integration-style tests: `tests/api/test_api.py`
- Load testing script: `load_tests/locustfile.py`
- Load test instructions: `load_tests/README.md`

What is missing to finish M24
- Explicit confirmation that API tests run in CI for this repo.
- A documented load test run with results (response times, failure rate).

Process to complete M24
1) API tests
   - Run locally: `uv run pytest tests/`
   - Confirm CI runs tests (see `.github/workflows/tests.yaml`).
2) Load tests
   - Start API locally on `http://localhost:8000`.
   - Run a smoke test:
     `uv run locust -f load_tests/locustfile.py --headless -u 5 -r 1 -t 30s --host http://localhost:8000`
   - Record results and summarize in the report.

## M25 - Specialized deployment API (ONNX or BentoML)

Status: in progress

What exists
- BentoML service: `src/pneumoniaclassifier/bento_service.py`
- Bento build config: `bentofile.yaml`
- Shared inference utilities: `src/pneumoniaclassifier/inference.py`

How to run and invoke (local)
1) Start the service:
   `uv run bentoml serve src.pneumoniaclassifier.bento_service:PneumoniaClassifierService`
2) Invoke prediction:
   `curl -F "image=@tests/fixtures/sample.png" http://127.0.0.1:3000/predict`

What is missing
- A short validation note or test run (record response output).
- Optional: containerize the Bento service and deploy to Cloud Run.

Suggested completion path
1) Run the Bento service and capture a sample response.
2) Add the response to the report or README.
3) (Optional) Build and deploy the Bento container to Cloud Run.

## M26 - Frontend for the API

Status: done

Evidence
- Frontend app: `frontend_django/`
- Local run command documented in `README.md`

Notes
- The frontend can point to the API base URL and upload an image for prediction.
