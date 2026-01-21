# Frontend (M26)

Static frontend for testing the FastAPI inference service.

## Run locally

Open `frontend/index.html` in a browser. Update the API base URL if your server runs elsewhere.

## Notes

- The API must expose `GET /health` and `POST /predict` endpoints.
- The prediction request expects form-data with a `file` field.
