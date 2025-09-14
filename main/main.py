import numpy as np
from layers.layers import Layers
from train.train import TrainModel

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

model = Layers(input_dim = 2, hidden_dim = 2, output_dim = 1, lr = 0.001)
trainer = TrainModel(model)
trainer.train(X, y, epochs = 200)