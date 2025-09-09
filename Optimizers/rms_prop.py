import numpy as np

class rms_prop:
    def __init__(self,ma, lr = 0.01, beta = 0.9, eps = 1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.ma = None

    def update_params(self, params, grads):
        if self.ma is None:
            self.ma = [np.zeros_like(p) for p in params]
        updated_params = []
        for i in range(len(params)):
            self.ma[i] = self.beta * self.ma[i] + (1-self.beta) * (grads[i] **2)
            new_param = params[i] - (self.lr / (np.sqrt(self.ma[i] + self.eps))) * grads[i]
            updated_params.append(new_param)
        return updated_params
