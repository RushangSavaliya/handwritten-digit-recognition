# src/preprocess.py

from skimage import io, color, transform
import numpy as np

def preprocess_image(image_path):
    # Load image
    image = io.imread(image_path)
    # Convert to grayscale
    gray = color.rgb2gray(image) if image.ndim == 3 else image.astype(float)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    if gray.mean() > 0.5:
        gray = 1 - gray
    resized = transform.resize(gray, (8, 8), anti_aliasing=True)
    resized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
    return resized.flatten().astype(np.float32)