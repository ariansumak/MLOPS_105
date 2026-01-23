import pytest
import torch

from pneumoniaclassifier.modeling import build_model, _set_trainable_layers


def test_build_model():
    """Test model output shape"""
    model = build_model("efficientnet_b0", num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert output.shape == (2, 2)


def test_set_trainable_layers():
    model = build_model("efficientnet_b0", num_classes=2, pretrained=False)
    _set_trainable_layers(model, unfreeze_blocks=0)

    # Classifier should be trainable
    assert any(p.requires_grad for p in model.classifier.parameters())

    # Features should be frozen
    assert not any(p.requires_grad for p in model.features.parameters())


def test_invalid_model_name_raises_error():
    """Test that invalid model name raises ValueError."""
    with pytest.raises(ValueError):
        build_model("invalid_model", num_classes=2, pretrained=False)
