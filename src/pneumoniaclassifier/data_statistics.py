"""Dataset statistics and visualization utilities."""

from __future__ import annotations
import typer
from pathlib import Path

app = typer.Typer()

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def dataset_statistics(data_dir: str | Path = "data/chest_xray") -> dict:
    """
    Analyze and visualize dataset statistics.

    Generates:
    - Summary statistics (train/val/test sample counts)
    - Class distribution plots
    - Sample images visualization

    Args:
        data_dir: Path to dataset root directory with train/val/test subdirectories.

    Returns:
        Dictionary containing:
            - train_samples: Number of training samples
            - val_samples: Number of validation samples
            - test_samples: Number of test samples
            - class_counts: Dict of class names and their sample counts
            - class_distribution: Dict with distribution percentages

    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    # Count samples
    stats = {"train_samples": 0, "val_samples": 0, "test_samples": 0}
    class_counts = {}

    splits = {"train": "train", "val": "val", "test": "test"}

    for split_name, split_dir in splits.items():
        split_path = data_dir / split_dir

        if not split_path.exists():
            continue

        split_count = 0
        for class_dir in split_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                num_samples = len(list(class_dir.glob("*.*")))
                split_count += num_samples

                if class_name not in class_counts:
                    class_counts[class_name] = {"train": 0, "val": 0, "test": 0}
                class_counts[class_name][split_name] = num_samples

        stats[f"{split_name}_samples"] = split_count

    # Print statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Training samples: {stats['train_samples']}")
    print(f"Validation samples: {stats['val_samples']}")
    print(f"Test samples: {stats['test_samples']}")
    print(f"Total samples: {sum(stats.values())}")
    print("\nClass distribution:")
    for class_name, counts in sorted(class_counts.items()):
        total = sum(counts.values())
        print(f"  {class_name}: {total} (train: {counts['train']}, val: {counts['val']}, test: {counts['test']})")
    print("=" * 60 + "\n")

    # Visualize class distribution
    _plot_class_distribution(data_dir, class_counts)

    # Visualize sample images
    _plot_sample_images(data_dir)

    # Calculate distribution percentages
    total_samples = sum(stats.values())
    class_distribution = {
        class_name: {
            "count": sum(counts.values()),
            "percentage": (sum(counts.values()) / total_samples * 100) if total_samples > 0 else 0,
        }
        for class_name, counts in class_counts.items()
    }

    stats["class_counts"] = class_counts
    stats["class_distribution"] = class_distribution

    return stats


def _plot_class_distribution(data_dir: Path, class_counts: dict) -> None:
    """Plot class distribution across splits."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stacked bar chart
    classes = list(class_counts.keys())
    splits = ["train", "val", "test"]
    split_data = {split: [class_counts[cls][split] for cls in classes] for split in splits}

    x = np.arange(len(classes))
    width = 0.25

    for i, split in enumerate(splits):
        axes[0].bar(x + i * width, split_data[split], width, label=split)

    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Number of Samples")
    axes[0].set_title("Class Distribution Across Splits")
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(classes)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Pie chart for overall distribution
    total_counts = [sum(class_counts[cls].values()) for cls in classes]
    axes[1].pie(
        total_counts,
        labels=classes,
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1].set_title("Overall Class Distribution")

    plt.tight_layout()
    output_path = Path("reports/figures") / "class_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"✓ Class distribution plot saved to {output_path}")
    plt.close()


def _plot_sample_images(data_dir: Path, num_samples_per_class: int = 3) -> None:
    """Plot sample images from dataset."""
    train_dir = data_dir / "train"

    if not train_dir.exists():
        return

    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    num_classes = len(classes)

    if num_classes == 0:
        return

    fig, axes = plt.subplots(num_classes, num_samples_per_class, figsize=(12, 4 * num_classes))

    # Handle case of single class
    if num_classes == 1:
        axes = axes.reshape(1, -1)

    for class_idx, class_name in enumerate(classes):
        class_dir = train_dir / class_name
        # Match all common image formats
        images = (
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.png"))
            + list(class_dir.glob("*.JPG"))
            + list(class_dir.glob("*.JPEG"))
            + list(class_dir.glob("*.PNG"))
        )[:num_samples_per_class]

        for img_idx, img_path in enumerate(images):
            try:
                img = Image.open(img_path).convert("RGB")
                # Resize for faster plotting
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                axes[class_idx, img_idx].imshow(img)
                axes[class_idx, img_idx].set_title(f"{class_name}")
                axes[class_idx, img_idx].axis("off")
            except (OSError, RuntimeError):
                axes[class_idx, img_idx].text(0.5, 0.5, "Failed to load", ha="center", va="center")
                axes[class_idx, img_idx].axis("off")

    plt.tight_layout()
    output_path = Path("reports/figures") / "sample_images.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"✓ Sample images plot saved to {output_path}")
    plt.close()

@app.command()
def stats(data_dir: str = "data/chest_xray"):
    """Analyze and visualize dataset statistics."""
    from pneumoniaclassifier.data_statistics import dataset_statistics
    dataset_statistics(data_dir=data_dir)

if __name__ == "__main__":
    app()
