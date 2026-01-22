# Load Tests

This folder contains a Locust script for exercising the FastAPI health and inference endpoints.

## Prerequisites
- Start the API locally (default host: `http://localhost:8000`).
- Ensure `tests/fixtures/sample.png` exists (it is loaded by the Locust user).

## Run a local load test
```bash
uv run locust -f load_tests/locustfile.py --host http://localhost:8000
```

## Run a smoke load test
```bash
uv run locust -f load_tests/locustfile.py --headless -u 5 -r 1 -t 30s --host http://localhost:8000
```

## Expected behavior
- `/health` returns HTTP 200.
- `/predict` returns HTTP 200 with a JSON response containing label, confidence, and probabilities.
