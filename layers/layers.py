from activations.sigmoid import Sigmoid
from dense.dense import Dense
from activations.relu import Relu
from losses.cross_entropy import BinaryCrossEntropy
from optimizers.adam_optimiser import Adam

class Layers:
    def __init__(self, input_dim, hidden_dim, output_dim, lr = 0.001):
        self.y = None
        self.y_pred = None
        self.dense1 = Dense(input_dim, hidden_dim)
        self.activation1 = Relu()
        self.dense2 = Dense(hidden_dim, output_dim)
        self.activation2 = Sigmoid()
        self.loss_func = BinaryCrossEntropy()
        self.optimizer = Adam(lr=lr)

    def forward(self, X, y):
        z1 = self.dense1.forward(X)
        a1 = self.activation1.forward(z1)
        z2 = self.dense2.forward(a1)
        self.y_pred = self.activation2.forward(z2)
        loss = self.loss_func.forward(y, self.y_pred)
        return loss, self.y_pred

    def backward(self):
        dL = self.loss_func.derivative(self.y, self.y_pred)
        dZ2 = self.activation2.derivative(dL)
        dA1 = self.dense2.backward(dZ2)
        dZ1 = self.activation1.derivative(dA1)
        dX = self.dense1.backward(dZ1)

    def updated_params(self):
        self.dense1.update_params(self.optimizer)
        self.dense2.update_params(self.optimizer)




