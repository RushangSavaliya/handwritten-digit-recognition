# src/predict.py

import sys
from model import train_knn  # src/model.py
from preprocess import preprocess_image  # src/preprocess.py

if len(sys.argv) < 2:
    print("Usage: python predict.py <image>")
else:
    # 1st: train the model
    model = train_knn()
    # 2nd: preprocess the input image
    image_data = preprocess_image(sys.argv[1])
    # 3rd: predict the digit
    prediction = model.predict([image_data])[0]
    print("Predicted digit:", prediction)