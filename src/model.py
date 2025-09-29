# src/model.py

"""Simple KNN digit recognition."""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def train_knn(k=3):
    """Load data, train and return KNN model."""
    digits = datasets.load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model
