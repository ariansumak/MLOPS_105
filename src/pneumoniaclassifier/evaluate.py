from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pneumoniaclassifier.data import get_dataloaders
from pneumoniaclassifier.modeling import build_model
import typer

app = typer.Typer()

@app.command()
def evaluate_cli(
    model_checkpoint: str = "models/model.ckpt",
    data_dir: str = "data/chest_xray",
    batch_size: int = 32,
    device: str = "cpu"
):
    """Standalone CLI to evaluate a saved model checkpoint."""
    device = torch.device(device)
    _, _, test_loader = get_dataloaders(data_dir, batch_size=batch_size)
    
    # Load model (Example: assuming a generic build)
    model = build_model("efficientnet_b0", num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Call your original evaluate logic
    loss, acc = evaluate(model, test_loader, criterion, device, show_progress=True)
    print(f"Evaluation Results -> Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
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

if __name__ == "__main__":
    app()