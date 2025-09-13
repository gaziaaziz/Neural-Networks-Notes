import numpy as np

class MSE:

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_true, y_pred):
        self.y_true, self.y_pred = y_true, y_pred
        return np.mean((self.y_true - self.y_pred) ** 2)

    def derivative(self):
        return -2 * (self.y_true - self.y_pred) / self.y_true.size
