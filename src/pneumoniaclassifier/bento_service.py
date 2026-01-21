from __future__ import annotations

import io
import os

import bentoml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from pneumoniaclassifier.inference import (
    build_transform,
    get_device,
    load_config,
    load_model,
    predict_image,
    resolve_config_path,
)

_SERVICE: PneumoniaClassifierService | None = None


def _get_service() -> PneumoniaClassifierService:
    """Return the initialized Bento service instance."""

    if _SERVICE is None:
        raise RuntimeError("Bento service is not initialized.")
    return _SERVICE


def _get_cors_origins() -> list[str]:
    """Return allowed CORS origins."""

    raw_origins = os.getenv("PNEUMONIA_CORS_ORIGINS")
    if raw_origins:
        return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return [
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ]


compat_app = FastAPI(title="Pneumonia Classifier Bento", version="0.1.0")
cors_origins = _get_cors_origins()
if cors_origins:
    compat_app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )


@compat_app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""

    return {"status": "ok"}


@compat_app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    """Run inference on an uploaded image."""

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    service = _get_service()
    return service.predict_image(image)


@bentoml.asgi_app(compat_app, path="/")
@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 30},
)
class PneumoniaClassifierService:
    """BentoML service for pneumonia image classification."""

    def __init__(self) -> None:
        config_path = resolve_config_path()
        cfg = load_config(config_path)
        self._device = get_device(cfg.inference.device)
        self._model = load_model(cfg, self._device)
        self._transform = build_transform(cfg)
        self._class_names = list(cfg.inference.class_names)
        global _SERVICE
        _SERVICE = self

    def predict_image(self, image: Image.Image) -> dict[str, object]:
        """Run inference on a provided image."""

        label, confidence, probabilities = predict_image(
            image=image,
            model=self._model,
            transform=self._transform,
            device=self._device,
            class_names=self._class_names,
        )
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    @bentoml.api(route="/bento_predict")
    def predict(self, image: Image.Image) -> dict[str, object]:
        """Run inference on a raw image payload."""

        return self.predict_image(image)
