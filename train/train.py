class TrainModel:
    def __init__(self, model):
        self.model = model
    def train(self, X, y, epochs = 100):
        for epoch in range(epochs):
            y_pred, loss = self.model.forward(X, y)
            self.model.backward()
            self.model.updated_params()

            if(epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {float(loss):.4f}")