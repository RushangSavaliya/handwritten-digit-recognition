# src/predict.py

import sys
from model import train_knn
from preprocess import preprocess_image

def predict(image_path):
    model, _, _ = train_knn()
    img_data = preprocess_image(image_path)
    prediction = int(model.predict([img_data])[0])
    return prediction

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image>")
        return

    image_path = sys.argv[1]

    prediction = predict(image_path)
    print(f"Predicted digit: {prediction}")

if __name__ == "__main__":
    main()
