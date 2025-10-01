# src/model.py

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def train_knn():
    digits = datasets.load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)
    return model, X_test, y_test
    