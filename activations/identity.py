import numpy as np

class Identity:
    def __init__(self):
        self.activation = None

    def forward(self, x):
        self.activation = x
        return self.activation

    def derivative(self, grad_output):
        return grad_output* (np.ones_like(self.activation))

