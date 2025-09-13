import numpy as np

class BinaryCrossEntropy:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_true, y_pred):
        self.y_true, self.y_pred = y_true, y_pred
        eps = 1e-15
        self.y_pred = np.clip(self.y_pred, eps, 1-eps)
        # we clip values in case they are very close to zero or zero in which case log(0) will tend to infinity crashing the program
        return -np.mean(self.y_true * np.log(self.y_pred) + (1-self.y_true) * np.log(1-self.y_pred))

    def derivative(self, y_true, y_pred):
        return (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred) * self.y_true.size)
