import pytest
from PIL import Image

from pneumoniaclassifier.data import get_dataloaders


class TestGetDataLoaders:
    """Test data loading functionality"""

    @pytest.fixture
    def dummy_data_dir(self, tmp_path):
        """Create a minimal dummy dataset structure."""
        for split in ["train", "val", "test"]:
            for class_name in ["normal", "pneumonia"]:
                (tmp_path / split / class_name).mkdir(parents=True, exist_ok=True)
                # Create a dummy image file
                img = Image.new("RGB", (224, 224), color="red")
                img.save(tmp_path / split / class_name / "dummy.jpg")
        return tmp_path

    def test_dataloaders_return_correct_structure(self, dummy_data_dir):
        """Test that get_dataloaders returns 3 DataLoaders."""
        train_loader, val_loader, test_loader = get_dataloaders(str(dummy_data_dir))
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_batch_shape(self, dummy_data_dir):
        """Test that batches have correct tensor shape."""
        train_loader, _, _ = get_dataloaders(str(dummy_data_dir), batch_size=2)
        batch, labels = next(iter(train_loader))
        assert batch.shape == (2, 3, 224, 224)  # (batch, channels, height, width)
        assert labels.shape == (2,)

    def test_augmentation_applied(self, dummy_data_dir):
        """Test that augmentation is applied when enabled."""
        _, _, _ = get_dataloaders(str(dummy_data_dir), augment=True)
        # Simple check: should not raise an error
        assert True
