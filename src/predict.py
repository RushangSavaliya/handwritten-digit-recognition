"""Digit prediction from images."""
import sys
from model import train_knn
from preprocess import preprocess_image


def predict(image_path, k=3):
    """Predict digit from image."""
    model = train_knn(k)
    img_data = preprocess_image(image_path)
    prediction = int(model.predict([img_data])[0])
    
    # Calculate confidence
    distances, _ = model.kneighbors([img_data], n_neighbors=k, return_distance=True)
    confidence = 1.0 / (1e-8 + distances.mean())
    
    return prediction, confidence


def main():
    """Command line interface."""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image> [k]")
        return
    
    image_path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    prediction, confidence = predict(image_path, k)
    print(f"Predicted digit: {prediction} (confidence: {confidence:.2f})")


if __name__ == "__main__":
    main()
