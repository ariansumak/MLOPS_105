from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from omegaconf import DictConfig, OmegaConf

from pneumoniaclassifier import api as api_module


class DummyModel(torch.nn.Module):
    """Simple model stub for API tests."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return fixed logits for deterministic responses."""

        batch_size = inputs.shape[0]
        logits = torch.tensor([[0.1, 0.9]], dtype=torch.float32)
        return logits.repeat(batch_size, 1)


def _dummy_config() -> DictConfig:
    """Build a minimal inference configuration for tests."""

    return OmegaConf.create(
        {
            "model": {
                "name": "dummy",
                "num_classes": 2,
                "pretrained": False,
                "checkpoint_path": "unused",
            },
            "inference": {
                "device": "cpu",
                "class_names": ["NORMAL", "PNEUMONIA"],
                "image_size": 224,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }
    )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a FastAPI test client with stubbed model and config."""

    def dummy_transform(_image: object) -> torch.Tensor:
        """Return a zeroed tensor for deterministic preprocessing."""

        return torch.zeros(3, 224, 224, dtype=torch.float32)

    monkeypatch.setattr(api_module, "_load_config", lambda _path: _dummy_config())
    monkeypatch.setattr(api_module, "_load_model", lambda _cfg, _device: DummyModel())
    monkeypatch.setattr(api_module, "_build_transform", lambda _cfg: dummy_transform)
    app = api_module.create_app()
    return TestClient(app)


def _fixture_image_bytes() -> bytes:
    """Load fixture image bytes for API tests."""

    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "sample.png"
    return fixture_path.read_bytes()


def test_health_endpoint(client: TestClient) -> None:
    """Ensure the health endpoint returns an ok status."""

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_prediction(client: TestClient) -> None:
    """Ensure the predict endpoint returns a label and probabilities."""

    files = {"file": ("sample.png", _fixture_image_bytes(), "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    payload = response.json()
    expected_confidence = torch.softmax(torch.tensor([0.1, 0.9]), dim=0)[1].item()
    assert payload["label"] == "PNEUMONIA"
    assert payload["confidence"] == pytest.approx(expected_confidence, rel=1e-3)
    assert set(payload["probabilities"].keys()) == {"NORMAL", "PNEUMONIA"}


def test_predict_rejects_non_image_upload(client: TestClient) -> None:
    """Ensure the predict endpoint rejects non-image uploads."""

    files = {"file": ("sample.txt", b"not an image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert response.json()["detail"] == "Only image uploads are supported."
