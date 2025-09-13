import numpy as np
class Relu:
    def __init__(self):
        self.activation = None

    def forward(self, x):
        self.activation = np.maximum(0, x)
        return self.activation

    def derivative(self, grad_output):
        return grad_output * (self.activation > 0).astype(float)