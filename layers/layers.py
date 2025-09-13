from activations.sigmoid import Sigmoid
from dense.dense import Dense
from activations.relu import Relu
from losses.cross_entropy import BinaryCrossEntropy
from optimizers.SGD_Momentum import SgdMomentum

class Layers:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.dense1 = Dense(input_dim, hidden_dim)
        self.activation1 = Relu()
        self.dense2 = Dense(hidden_dim, output_dim)
        self.activation2 = Sigmoid()
        self.loss_func = BinaryCrossEntropy()
        self.optimizer = SgdMomentum()

    def forward(self, X, y):
        z1 = self.dense1.forward(X)
        a1 = self.activation1.forward(z1)
        z2 = self.dense2.forward(a1)
        y_pred = self.activation2.forward(z2)
        loss = self.loss_func.forward(y, y_pred)
        return loss, y_pred

