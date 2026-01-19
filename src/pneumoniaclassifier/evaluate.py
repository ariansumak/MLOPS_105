from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = False,
    description: str = "Validation",
) -> tuple[float, float]:
    """Evaluate a model on a validation dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        data_iter = loader
        if show_progress:
            data_iter = tqdm(loader, desc=description, dynamic_ncols=True, leave=False)
        for inputs, targets in data_iter:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy
