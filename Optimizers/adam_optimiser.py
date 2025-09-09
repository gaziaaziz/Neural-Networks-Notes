import numpy as np

class AdamOptimiser:
    def __init__(self, lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.v = None
        self.m = None
        self.t = 0

    def update_params(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]

        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        updated_param = []
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * (grads[i] ** 2)
            m_hat = self.m[i] / (1-(self.beta1 ** self.t))
            v_hat = self.v[i] / (1-(self.beta2 ** self.t))
            new_param = params[i] - (m_hat * self.lr)  / np.sqrt(v_hat) + self.eps
            updated_param.append(new_param)
        return updated_param


