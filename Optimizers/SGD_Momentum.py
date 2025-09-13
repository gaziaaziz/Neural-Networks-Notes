import numpy as np

class SgdMomentum:
    def __init__(self, lr = 0.01, beta = 0.9):
        self.lr = lr
        self.beta = beta
        self.v = None

    def update_params(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        updated_params = []
        for i in range(len(params)):
            self.v[i] = self.beta * self.v[i] + (1-self.beta) * grads[i]
            new_param = params[i] - self.lr * self.v[i]
            updated_params.append(new_param)
        return updated_params




