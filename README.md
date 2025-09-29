# Handwritten Digit Recognition

Minimal digit classifier using KNN with sklearn digits dataset.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Predict digit
python src/predict.py my_images/3_300x300_20-brush.png

# Use different k value 
python src/predict.py my_images/3_300x300_20-brush.png 5
```

## How it works

1. Loads sklearn digits dataset (8x8 images)
2. Trains KNN classifier
3. Preprocesses your image to 8x8 
4. Predicts digit with confidence score