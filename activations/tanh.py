import numpy as np

class Tanh:
    def  __init__(self):
        self.activation = None
def forward(self, x):
    self.activation = np.tanh(x)
    return self.activation

def derivative(self, grad_output):
    return grad_output * (1 - self.activation ** 2)