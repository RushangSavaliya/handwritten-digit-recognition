# Handwritten Digit Recognition

Minimal digit classifier using KNN with Scikit-learn digits dataset.

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Predict a digit from your image

```bash
python src/predict.py my_images/3_300x300_20-brush.png
```

## How it works

1. Loads Scikit-learn digits dataset (8x8 images)
2. Trains KNN classifier
3. Preprocesses your image to 8x8