import numpy as np

def tanH(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanH_derivative(x):
    return 1 - np.tanh(x) ** 2