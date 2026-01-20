import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pneumoniaclassifier.model import _build_model
from pneumoniaclassifier.train import train_epoch


@pytest.fixture
def dummy_data():
    """Create dummy dataset (keep this simple)."""
    X_train = torch.randn(32, 3, 224, 224)
    y_train = torch.randint(0, 2, (32,))
    dataset = TensorDataset(X_train, y_train)
    return DataLoader(dataset, batch_size=8)


@pytest.fixture
def real_model():
    """Use the actual model from model.py."""
    return _build_model(
        model_name="efficientnet_b0",
        num_classes=2,
        pretrained=False,  # Don't load weights for speed
    )


def test_train_epoch_with_real_model(real_model, dummy_data):
    """Test train_epoch with actual EfficientNet model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(real_model.parameters())
    device = torch.device("cpu")

    epoch_loss, epoch_acc, _ = train_epoch(
        real_model,
        dummy_data,
        dummy_data,
        criterion,
        optimizer,
        device,
        epoch=1,
        global_step=0,
        show_progress=False,
        wandb_enabled=False,
    )

    assert isinstance(epoch_loss, float)
    assert isinstance(epoch_acc, float)
    assert 0 <= epoch_acc <= 1
