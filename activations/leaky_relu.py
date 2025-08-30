import numpy as np

def leaky_relu(x: np.ndarray, alpha) -> np.ndarray:
    return np.where(x > 0, x, alpha*x)

def leaky_relu_derivative(x: np.ndarray, alpha) -> np.ndarray:
    return np.where(x > 0, 1, alpha)