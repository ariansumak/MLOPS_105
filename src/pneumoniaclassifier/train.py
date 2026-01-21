from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

import wandb
from pneumoniaclassifier.evaluate import evaluate
from pneumoniaclassifier.modeling import build_model, set_trainable_layers


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
    save_checkpoint: bool = True
    checkpoint_path: Path = Path("models/m22_model.pt")


@dataclass
class TrainEpochConfig:
    """Configuration for training epoch."""

    epoch: int
    log_interval_steps: int
    wandb_enabled: bool


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

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.run_name,
        config=OmegaConf.to_container(OmegaConf.structured(config), resolve=True),
        save_code=True,
    )


def _save_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    """Save the model state dict to a checkpoint path.

    Args:
        model: Trained model to persist.
        checkpoint_path: Destination path for the checkpoint.
    """

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main")
def train(cfg: DictConfig) -> None:
    """Train an EfficientNet model using the provided configuration."""
    # Set seeds & Device
    torch.manual_seed(cfg.train.seed)
    device = _get_device(cfg.train.device)

    train_loader, val_loader, _ = hydra.utils.instantiate(cfg.data)

    model = build_model(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    )
    set_trainable_layers(model, cfg.model.unfreeze_blocks)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=_filter_trainable_parameters(model))

    _init_wandb(cfg)
    if cfg.wandb.enabled and cfg.wandb.log_model:
        wandb.watch(model, log="all", log_freq=100)

    global_step = 0
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, train_acc, global_step = train_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            global_step=global_step,
            log_interval_steps=cfg.train.log_interval_steps,
            eval_interval_steps=cfg.eval.interval_steps,
            wandb_enabled=cfg.wandb.enabled,
            show_progress=True,
        )

        if cfg.eval.run_at_epoch_end:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = 0.0, 0.0

        # Log epoch metrics
        if cfg.wandb.enabled:
            wandb.log(
        {
            "train/epoch_loss": train_loss,
            "train/epoch_accuracy": train_acc,
            "val/epoch_loss": val_loss,
            "val/epoch_accuracy": val_acc,
            "epoch": epoch,
        })

        print(
            f"Epoch {epoch}/{cfg.train.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    if cfg.train.save_checkpoint:
        _save_checkpoint(model, Path(cfg.train.checkpoint_path))


if __name__ == "__main__":
    train()
