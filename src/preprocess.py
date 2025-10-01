# src/preprocess.py

import cv2
import numpy as np

def preprocess_image(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Normalize to 0-1 range
    gray = image.astype(np.float32) / 255.0
    
    # Invert if background is white (mean > 0.5)
    if gray.mean() > 0.5:
        gray = 1 - gray
    
    # Resize to 8x8
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    
    # Final normalization
    resized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
    
    return resized.flatten().astype(np.float32)