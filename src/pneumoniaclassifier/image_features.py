from __future__ import annotations

import numpy as np
from PIL import Image


def extract_features(image: Image.Image) -> dict[str, float]:
    """Extract numeric features from an image for drift detection."""

    array = np.asarray(image).astype(np.float32) / 255.0
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], repeats=3, axis=2)
    if array.shape[2] > 3:
        array = array[:, :, :3]

    channel_means = array.mean(axis=(0, 1))
    channel_stds = array.std(axis=(0, 1))

    grayscale = array.mean(axis=2)
    brightness = float(grayscale.mean())
    contrast = float(grayscale.std())

    hist, _ = np.histogram(grayscale, bins=16, range=(0.0, 1.0), density=True)
    hist = hist + 1e-12
    entropy = float(-np.sum(hist * np.log(hist)))

    return {
        "mean_r": float(channel_means[0]),
        "mean_g": float(channel_means[1]),
        "mean_b": float(channel_means[2]),
        "std_r": float(channel_stds[0]),
        "std_g": float(channel_stds[1]),
        "std_b": float(channel_stds[2]),
        "brightness": brightness,
        "contrast": contrast,
        "entropy": entropy,
    }
