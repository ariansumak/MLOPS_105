from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchvision import models

from pneumoniaclassifier.data import get_dataloaders
from pneumoniaclassifier.evaluate import evaluate


@dataclass
class DataConfig:
    """Configuration for dataset locations and loading."""

    _target_: str = "pneumoniaclassifier.data.get_dataloaders"
    data_dir: str = "data/chest_xray"
    batch_size: int = 16
    num_workers: int = 2
    augment: bool = True


@dataclass
class ModelConfig:
    """Configuration for the model architecture and training."""

    name: str = "efficientnet_b0"
    num_classes: int = 2
    pretrained: bool = True
    unfreeze_blocks: int = 0


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""

    _target_: str = "torch.optim.Adam"
    lr: float = 1e-4
    weight_decay: float = 1e-4


@dataclass
class TrainLoopConfig:
    """Configuration for the training loop."""

    epochs: int = 5
    seed: int = 42
    device: str = "auto"
    output_dir: Path = Path("reports")
    log_interval_steps: int = 50


@dataclass
class EvalConfig:
    """Configuration for validation during training."""

    interval_steps: int = 200
    run_at_epoch_end: bool = True


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""

    enabled: bool = True
    project: str = "pneumoniaclassifier"
    entity: str | None = None
    run_name: str | None = None
    log_model: bool = False


@dataclass
class TrainConfig:
    """Training configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


ConfigStore.instance().store(name="train", node=TrainConfig)


def _get_device(device: str) -> torch.device:
    """Resolve the requested device into a torch device."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Build an EfficientNet model with an updated classifier head."""

    model_registry = {
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
        "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
        "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
    }
    model_entry = model_registry.get(model_name)
    model_fn = model_entry[0] if model_entry is not None else None
    if model_fn is None:
        raise ValueError(f"Unsupported model name: {model_name}")

    weights = model_entry[1] if pretrained else None
    model = model_fn(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _set_trainable_layers(model: nn.Module, unfreeze_blocks: int) -> None:
    """Freeze all parameters except the classifier and optionally the last N EfficientNet blocks."""

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    if unfreeze_blocks <= 0:
        return

    if not hasattr(model, "features"):
        raise ValueError("Model does not expose a features attribute for block unfreezing.")

    blocks = list(model.features.children())
    if unfreeze_blocks > len(blocks):
        raise ValueError(f"unfreeze_blocks={unfreeze_blocks} exceeds available blocks ({len(blocks)}).")
    for block in blocks[-unfreeze_blocks:]:
        for param in block.parameters():
            param.requires_grad = True


# def _create_loader(
#     dataset: MyDataset,
#     batch_size: int,
#     num_workers: int,
#     shuffle: bool,
# ) -> DataLoader:
#     """Create a dataloader with consistent settings."""

#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=torch.cuda.is_available(),
#     )


def _filter_trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """Return parameters that require gradients."""

    return (param for param in model.parameters() if param.requires_grad)


def _init_wandb(config: TrainConfig) -> None:
    """Initialize a Weights & Biases run when enabled."""

    if not config.wandb.enabled:
        return

    import wandb

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.run_name,
        config=OmegaConf.to_container(OmegaConf.structured(config), resolve=True),
        save_code=True,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main")
def train(cfg: DictConfig) -> None:
    """Train an EfficientNet model using the provided configuration."""

    # config = OmegaConf.merge(OmegaConf.structured(TrainConfig), cfg)
    # train_config = OmegaConf.to_object(config)

    # Set seeds & Device
    torch.manual_seed(cfg.train.seed)
    device = _get_device(cfg.train.device)

    train_loader, val_loader, _ = hydra.utils.instantiate(cfg.data)

    model = _build_model(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    )
    _set_trainable_layers(model, cfg.model.unfreeze_blocks)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=_filter_trainable_parameters(model))

    _init_wandb(cfg)
    if cfg.wandb.enabled and cfg.wandb.log_model:
        import wandb

        wandb.watch(model, log="all", log_freq=100)

    global_step = 0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for inputs, targets in train_loader:
            global_step += 1
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            epoch_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            epoch_correct += (preds == targets).sum().item()
            epoch_total += batch_size

            if cfg.train.log_interval_steps > 0 and global_step % cfg.train.log_interval_steps == 0:
                batch_acc = (preds == targets).float().mean().item()
                if cfg.wandb.enabled:
                    # import wandb

                    wandb.log(
                        {
                            "train/step_loss": loss.item(),
                            "train/step_accuracy": batch_acc,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

            if cfg.eval.interval_steps > 0 and global_step % cfg.eval.interval_steps == 0:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                if cfg.wandb.enabled:
                    # import wandb

                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/accuracy": val_acc,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

        epoch_loss = epoch_loss / max(epoch_total, 1)
        epoch_acc = epoch_correct / max(epoch_total, 1)

        if cfg.eval.run_at_epoch_end:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = 0.0, 0.0

        if cfg.wandb.enabled:
            # import wandb

            wandb.log(
                {
                    "train/epoch_loss": epoch_loss,
                    "train/epoch_accuracy": epoch_acc,
                    "val/epoch_loss": val_loss,
                    "val/epoch_accuracy": val_acc,
                    "epoch": epoch,
                }
            )

        print(
            f"Epoch {epoch}/{cfg.train.epochs} "
            f"train_loss={epoch_loss:.4f} train_acc={epoch_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    train()
