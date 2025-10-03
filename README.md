# Handwritten Digit Recognition 🔢

A minimal yet effective handwritten digit classifier using K-Nearest Neighbors (KNN) algorithm with the Scikit-learn digits dataset. This project demonstrates image preprocessing and machine learning techniques for digit recognition.

## ✨ Features

- **Simple KNN classifier** trained on the Scikit-learn digits dataset
- **Automatic image preprocessing** including grayscale conversion, inversion detection, and resizing
- **Real-time prediction** from custom handwritten digit images
- **Minimal dependencies** for easy setup and deployment
- **Clean, modular code structure** for easy understanding and modification

## 🔧 Prerequisites

- Python 3.7 or higher
- pip package manager

## 🚀 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/RushangSavaliya/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Basic Prediction

Predict a digit from your own handwritten image:

```bash
python src/predict.py my_images/your_digit_image.png
```

### Example

```bash
python src/predict.py my_images/3_300x300_20-brush.png
```

**Output:**

```text
Predicted digit: [3]
```

## 🔍 How It Works

The digit recognition process follows these steps:

1. **Dataset Loading**: Loads the Scikit-learn digits dataset (8×8 grayscale images of digits 0-9)
2. **Model Training**: Trains a K-Nearest Neighbors classifier with k=1
3. **Image Preprocessing**:
   - Converts input image to grayscale
   - Detects and inverts white backgrounds
   - Resizes to 8×8 pixels to match training data
   - Normalizes pixel values
4. **Prediction**: Uses the trained model to classify the preprocessed image

## 🔬 Technical Details

### Dependencies

- **NumPy**: Numerical computing and array operations
- **Scikit-learn**: Machine learning library (KNN classifier and digits dataset)
- **OpenCV**: Image processing and computer vision

### Model Specifications

- **Algorithm**: K-Nearest Neighbors (KNN)
- **K value**: 1 (nearest neighbor)
- **Training data**: Scikit-learn digits dataset (1,797 samples)
- **Input size**: 8×8 grayscale images (64 features)
- **Output**: Single digit prediction (0-9)

---

**GitHub**: [@RushangSavaliya](https://github.com/RushangSavaliya) | **Repository**: [handwritten-digit-recognition](https://github.com/RushangSavaliya/handwritten-digit-recognition)
