# src/model.py

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

def train_knn():
    digits = datasets.load_digits()
    features = digits.data.astype(np.float32) / 16.0
    labels = digits.target
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(features, labels)
    return model