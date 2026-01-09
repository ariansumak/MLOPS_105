from pathlib import Path

import typer
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

DATA_DIR = 'data/chest_xray'

def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 4, augment: bool = True) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory with train/val/test folders
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        augment: Whether to apply data augmentation to training data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(f'{data_dir}/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(f'{data_dir}/val', transform=test_transform)
    test_dataset = datasets.ImageFolder(f'{data_dir}/test', transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    typer.run(get_dataloaders(DATA_DIR))
