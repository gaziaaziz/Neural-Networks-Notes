import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim)* 0.01
        self.b = np.zeros(1, output_dim)

    def forward(self, X):
        return np.dot(self.W, X) + self.b

