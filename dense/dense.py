import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim):
        self.X = None
        self.db = None
        self.dW = None
        self.W = np.random.randn(input_dim, output_dim)* 0.01
        self.b = np.zeros((1, output_dim))

    def forward(self, X):
        self.X = X
        return (np.dot(X, self.W)) + self.b

    def backward(self, grad_output):
        self.dW = np.dot(self.X.T, grad_output)
        self.db = np.sum(grad_output, axis= 0, keepdims= True)

        return np.dot(grad_output, self.W.T)

    def update_params(self, optimizer):
        params = [self.W, self.b]
        grads = [self.dW, self.db]
        updated_param = optimizer.update_params(params, grads)
        self.W, self.b = updated_param[0], updated_param[1]


