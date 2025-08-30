import numpy as np

def identity(x: np.ndarray) -> np.ndarray:
    return x

def identity_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)

