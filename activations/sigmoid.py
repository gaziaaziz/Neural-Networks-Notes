import numpy as np
class Sigmoid:
    def __init__(self):
        self.activation = None

    def forward(self, x):
        self.activation = 1/(1+np.exp(-x))
        return self.activation

    def derivative(self, grad_output):
        return grad_output*(self.activation * (1-self.activation))

