# Django Frontend (M26)

A DTU-themed Django frontend for testing the FastAPI inference API.

## Run locally

1. Start the FastAPI backend (default `http://localhost:8000`).
2. Run the Django app:

```bash
uv run python frontend_django/manage.py runserver 0.0.0.0:8001
```

3. Open `http://localhost:8001` in your browser.

## Notes

- The frontend calls `GET /health` and `POST /predict` on the configured backend.
- The prediction request uses form-data with a `file` field.
