import numpy as np

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1-eps)
    # we clip values in case they are very close to zero or zero in which case log(0) will tend to infinity crashing the program
    return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))

def binary_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1-eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)
