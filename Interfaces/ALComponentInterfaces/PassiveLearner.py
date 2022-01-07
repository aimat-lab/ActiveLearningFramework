class PassiveLearner:

    def initial_training(self, x_train, y_train, batch_size, epochs):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def train(self, x, y):
        raise NotImplementedError
