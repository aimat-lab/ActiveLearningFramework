class PassiveLearner:
    def initial_training(self, x_train, y_train, batch_size, epochs):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def train_instance(self, x, y):
        raise NotImplementedError

    def train_batch(self, x_train, y_train, batch_size, epochs):
        raise NotImplementedError
