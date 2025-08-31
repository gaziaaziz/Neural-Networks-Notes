import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean(y_true - y_pred) ** 2)

def rmse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    rmse_value = rmse(y_true, y_pred)
    if rmse_value == 0:
        return np.zeros_like(y_true)
    return -(y_true - y_pred)/ (rmse_value * y_true.size)