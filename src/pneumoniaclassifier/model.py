from torch import nn
from torchvision import models


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
