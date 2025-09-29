# src/preprocess.py

"""Simplified 8x8 image preprocessing for digit recognition."""

from skimage import io, color, transform
import numpy as np


def preprocess_image(path):
    """Convert image to 8x8 format for digit recognition."""
    # Load image
    image = io.imread(path)

    # Convert to grayscale
    gray = color.rgb2gray(image) if image.ndim == 3 else image.astype(float)

    # Normalize to [0, 1]
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

    # Invert if background is brighter than digit
    if gray.mean() > 0.5:
        gray = 1.0 - gray

    # Resize directly to 8x8 (skip cropping/padding for simplicity)
    resized = transform.resize(gray, (8, 8), anti_aliasing=True)

    # Normalize again just in case
    resized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)

    return resized.flatten().astype(np.float32)
