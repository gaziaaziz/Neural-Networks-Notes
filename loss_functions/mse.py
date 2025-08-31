import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return -2 * (y_true - y_pred) / y_true.size
