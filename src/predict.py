# src/predict.py

"""Digit prediction from images."""

import sys
from model import train_knn
from preprocess import preprocess_image


def predict(image_path, k=3):
    """Predict digit from image."""
    model = train_knn(k)
    img_data = preprocess_image(image_path)
    prediction = int(model.predict([img_data])[0])
    return prediction


def main():
    """Command line interface."""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image> [k]")
        return

    image_path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    prediction = predict(image_path, k)
    print(f"Predicted digit: {prediction}")


if __name__ == "__main__":
    main()
