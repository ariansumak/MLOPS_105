from __future__ import annotations

import io
import os
from pathlib import Path
import time

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import DictConfig
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms

from prometheus_client import Counter, Histogram, make_asgi_app, generate_latest, CONTENT_TYPE_LATEST


from pneumoniaclassifier.modeling import build_model, load_model_from_checkpoint
from pneumoniaclassifier.inference import (
    build_transform,
    get_device,
    load_config,
    load_model,
    predict_image,
    resolve_config_path,
)


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str


class PredictionResponse(BaseModel):
    """Inference response schema."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class RootResponse(BaseModel):
    """Root response schema."""

    message: str
    endpoints: dict[str, str]


def _get_cors_origins() -> list[str]:
    """Return allowed CORS origins."""

    raw_origins = os.getenv("PNEUMONIA_CORS_ORIGINS")
    if raw_origins:
        return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return [
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ]


def _resolve_config_path() -> Path:
    """Resolve the inference configuration path."""

    return resolve_config_path()


def _load_config(config_path: Path) -> DictConfig:
    """Load the inference configuration."""

    return load_config(config_path)


def _get_device(device_name: str) -> torch.device:
    """Resolve the requested device into a torch device."""

    return get_device(device_name)


def _build_transform(cfg: DictConfig) -> transforms.Compose:
    """Create the image preprocessing pipeline."""

    return build_transform(cfg)


def _load_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """Load the model for inference."""

    return load_model(cfg, device)

PREDICTION_COUNTER = Counter(
    "pneumonia_predictions_total", 
    "Total number of predictions", 
    ["label"]
)

# Track how long the inference takes
INFERENCE_LATENCY = Histogram(
    "pneumonia_inference_duration_seconds", 
    "Time spent performing model inference"
)

# Track errors (e.g., invalid images)
ERROR_COUNTER = Counter(
    "pneumonia_prediction_errors_total", 
    "Total number of prediction errors",
    ["error_type"]
)

def create_app() -> FastAPI:
    """Create the FastAPI application."""

    config_path = _resolve_config_path()
    cfg = _load_config(config_path)
    device = _get_device(cfg.inference.device)
    model = _load_model(cfg, device)
    transform = _build_transform(cfg)
    class_names = list(cfg.inference.class_names)

    app = FastAPI(title="Pneumonia Classifier API", version="0.1.0")

    @app.get("/metrics")
    def metrics():
        """Expose Prometheus metrics."""
        return Response(
            generate_latest(), 
            media_type=CONTENT_TYPE_LATEST
        )

    cors_origins = _get_cors_origins()
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
    app.state.cfg = cfg
    app.state.device = device
    app.state.model = model
    app.state.transform = transform
    app.state.class_names = class_names

    @app.get("/", response_model=RootResponse)
    def root() -> RootResponse:
        """Return basic API info."""

        return RootResponse(
            message="Pneumonia Classifier API is running.",
            endpoints={
                "health": "/health",
                "predict": "/predict",
                "docs": "/docs",
            },
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Return service health status."""

        return HealthResponse(status="ok")

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)) -> PredictionResponse:
        """Run inference on an uploaded image."""

        if file.content_type is None or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image uploads are supported.")

        image_bytes = await file.read()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except OSError as exc:
            ERROR_COUNTER.labels(error_type="image_decode_error").inc()
            raise HTTPException(status_code=400, detail="Invalid image file.") from exc
        
        tensor = app.state.transform(image).unsqueeze(0).to(app.state.device)

        start_time = time.time()

        with torch.inference_mode():
            logits = app.state.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        INFERENCE_LATENCY.observe(time.time() - start_time)

        confidence, index = torch.max(probs, dim=0)
        label = app.state.class_names[int(index)]
        probabilities = {name: float(probs[i]) for i, name in enumerate(app.state.class_names)}

        label, confidence, probabilities = predict_image(
            image=image,
            model=app.state.model,
            transform=app.state.transform,
            device=app.state.device,
            class_names=app.state.class_names,
        )

        PREDICTION_COUNTER.labels(label=label).inc()

        return PredictionResponse(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
        )

    return app


app = create_app()
