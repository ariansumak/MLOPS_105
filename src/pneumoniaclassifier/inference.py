from __future__ import annotations

import os
from pathlib import Path

import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms


from pneumoniaclassifier.modeling import build_model, load_model_from_checkpoint


def resolve_config_path() -> Path:
    """Resolve the inference configuration path."""

    env_path = os.getenv("PNEUMONIA_CONFIG")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[2] / "configs" / "inference.yaml"


def load_config(config_path: Path) -> DictConfig:
    """Load the inference configuration."""

    if not config_path.exists():
        raise FileNotFoundError(f"Inference config not found at {config_path}")
    return OmegaConf.load(config_path)


def get_device(device_name: str) -> torch.device:
    """Resolve the requested device into a torch device."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_transform(cfg: DictConfig) -> transforms.Compose:
    """Create the image preprocessing pipeline."""

    image_size = int(cfg.inference.image_size)
    mean = list(cfg.inference.mean)
    std = list(cfg.inference.std)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """Load the model for inference."""    
    if "artifact_path" in cfg.model:
        return load_model_from_wandb(cfg, device)

    checkpoint_path = Path(cfg.model.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model = build_model(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    )
    return load_model_from_checkpoint(model, checkpoint_path, device)



def load_model_from_wandb(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """Load model checkpoint from a W&B artifact."""

    artifact_ref = cfg.model.artifact_path
    if artifact_ref is None:
        raise ValueError("cfg.model.artifact must be set for W&B loading")

    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    artifact_dir = Path(artifact.download())

    # assume exactly one checkpoint file
    ckpt_files = list(artifact_dir.glob("*.pt"))
    if not ckpt_files:
        raise FileNotFoundError("No .pt file found in W&B artifact")

    checkpoint_path = ckpt_files[0]

    model = build_model(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        pretrained=False,  # important!
    )

    model = load_model_from_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model


def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    class_names: list[str],
) -> tuple[str, float, dict[str, float]]:
    """Run inference on a PIL image."""

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    confidence, index = torch.max(probs, dim=0)
    label = class_names[int(index)]
    probabilities = {name: float(probs[i]) for i, name in enumerate(class_names)}
    return label, float(confidence), probabilities
