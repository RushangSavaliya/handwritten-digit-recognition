"""Simple 8x8 image preprocessing."""
from skimage import io, color, transform, filters
import numpy as np


def preprocess_image(path):
    """Convert image to 8x8 format for digit recognition."""
    # Load and convert to grayscale
    image = io.imread(path)
    gray = color.rgb2gray(image) if image.ndim == 3 else image.astype(float)
    
    # Normalize and invert if needed
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    if gray.mean() > 0.5:
        gray = 1.0 - gray
    
    # Find digit region
    threshold = filters.threshold_otsu(gray) if gray.max() > gray.min() else 0.2
    mask = gray > threshold
    if mask.mean() < 0.01:
        mask = gray > threshold * 0.5
    
    # Crop to bounding box
    coords = np.argwhere(mask)
    if coords.size > 0:
        (y0, x0), (y1, x1) = coords.min(0), coords.max(0) + 1
        cropped = gray[y0:y1, x0:x1]
    else:
        cropped = gray
    
    # Make square and resize to 8x8
    h, w = cropped.shape
    size = max(h, w)
    pad_h, pad_w = (size - h) // 2, (size - w) // 2
    squared = np.pad(cropped, ((pad_h, size - h - pad_h), (pad_w, size - w - pad_w)))
    
    resized = transform.resize(squared, (8, 8), anti_aliasing=True)
    resized = filters.gaussian(resized, sigma=0.3)
    
    # Final normalization
    resized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
    return resized.flatten().astype(np.float32)
