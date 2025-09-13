import numpy as np

class RMSE:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_true, y_pred):
        self.y_true, self.y_pred = y_true, y_pred
        return np.sqrt(np.mean(y_true - y_pred) ** 2)

    def derivative(self):
        rmse_value = self.forward(self.y_true, self.y_pred)
        if rmse_value == 0:
            return np.zeros_like(self.y_true)
        return -(self.y_true - self.y_pred)/ (rmse_value * self.y_true.size)



