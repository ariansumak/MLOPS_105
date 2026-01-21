from __future__ import annotations

from pathlib import Path

from locust import HttpUser, between, task


class InferenceUser(HttpUser):
    """Locust user that exercises health and predict endpoints."""

    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        """Load fixture bytes once per user."""

        fixture_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample.png"
        self.image_bytes = fixture_path.read_bytes()

    @task(1)
    def health(self) -> None:
        """Check the health endpoint."""

        self.client.get("/health")

    @task(3)
    def predict(self) -> None:
        """Send an inference request with the fixture image."""

        files = {"file": ("sample.png", self.image_bytes, "image/png")}
        self.client.post("/predict", files=files)
