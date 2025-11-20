"""Fisher's Linear Discriminant Classifier Implementation"""

import scipy
import numpy as np
from sklearn.datasets import load_breast_cancer
from typing import List, Tuple


def get_Sw(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coumpute Within-Class Scatter Matrix"""
    labels = np.unique(y)
    n_features = X.shape[1]
    Sw = np.zeros((n_features, n_features))
    for label in labels:
        XOfLabel = X[y == label]
        meanOfClass = np.mean(XOfLabel, axis=0)
        diff = XOfLabel - meanOfClass
        Sw += diff.T @ diff
    return Sw


def get_Sb(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Between-Class Scatter Matrix"""
    overallMean = np.mean(X, axis=0)
    labels = np.unique(y)
    n_features = X.shape[1]
    Sb = np.zeros((n_features, n_features))
    for label in labels:
        XOfLabel = X[y == label]
        n_samplesOfLabel = XOfLabel.shape[0]
        meanOfClass = np.mean(XOfLabel, axis=0)
        meanDiff = (meanOfClass - overallMean).reshape(n_features, 1)
        Sb += n_samplesOfLabel * (meanDiff @ meanDiff.T)
    return Sb


def solve_fisher_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve for Fisher's Linear Discriminant Direction"""
    Sw = get_Sw(X, y)
    Sb = get_Sb(X, y)

    # Solve the generalized eigenvalue problem equation Sb*w = lambda*Sw*w
    eigvals, eigvecs = scipy.linalg.eigh(Sb, Sw)
    maxEigIndex = np.argmax(eigvals)
    w = eigvecs[:, maxEigIndex]
    return w


def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8):
    """Split data into training and testing sets"""
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    train_size = int(n_samples * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test


def train_fisher_classifier(X: np.ndarray, y: np.ndarray):
    """Train Fisher Classifier"""
    w = solve_fisher_direction(X, y)
    projections = X @ w  # compute projection
    labels = np.unique(y)
    meanProjections = []
    for label in labels:
        meanProjections.append((label, np.mean(projections[y == label])))
    sortedMeans = sorted(meanProjections, key=lambda x: x[1])
    thresholds = []
    for i in range(len(sortedMeans) - 1):
        meanPrev = sortedMeans[i][1]
        meanPost = sortedMeans[i + 1][1]
        thresholds.append((meanPrev + meanPost) / 2)

    return w, thresholds, sortedMeans


def predict_fisher_classifier(
    X: np.ndarray, w: np.ndarray, thresholds: list, sortedMeans: List[Tuple[int, float]]
) -> np.ndarray:
    """Predict using Fisher Classifier"""
    projections = X @ w
    y_pred = np.zeros(projections.shape[0])
    labels = [item[0] for item in sortedMeans]
    for i, projection in enumerate(projections):
        for j, threshold in enumerate(thresholds):
            if projection < threshold:
                y_pred[i] = labels[j]
                break
        else:
            y_pred[i] = labels[-1]
    return y_pred


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray):
    """Evaluate Classifier Accuracy and Confusion Matrix"""
    accuracy = np.mean(y_true == y_pred)

    # Compute confusion matrix
    labels = np.unique(y_true)
    n_classes = len(labels)
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    # Fill confusion matrix
    for t, p in zip(y_true, y_pred):
        true_index = np.where(labels == t)[0][0]
        pred_index = np.where(labels == p)[0][0]
        confusion[true_index, pred_index] += 1

    return accuracy, confusion


if __name__ == "__main__":
    np.random.seed(42)

    data = load_breast_cancer()
    X = data["data"]
    y = data["target"]

    X_train, y_train, X_test, y_test = split_data(X, y)

    w, thresholds, sortedMeans = train_fisher_classifier(X_train, y_train)

    y_pred = predict_fisher_classifier(X_test, w, thresholds, sortedMeans)

    accuracy, confusion = evaluate_classifier(y_test, y_pred)
    print(f"Fisher Classifier Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:\n", confusion)
