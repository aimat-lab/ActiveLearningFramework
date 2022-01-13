class PassiveLearner:
    """
    Interface for the PL (to be extended SL model)
    """

    def initial_training(self, x_train, y_train, **kwargs):
        # TODO: maybe move scaling of data into initiation??
        """
        Initial batch training => for determination of initial weights and potentially for setting the scaler for the input data

        :param x_train: training input (array of input data)
        :param y_train: training labels (array of correct output data)
        :param kwargs: can contain additional properties for the training
        """
        raise NotImplementedError

    def predict(self, x):
        """
        Get the predicted output based on the current state of the ML model (current training status)

        :param x: input values
        :return: prediction y (usually one numerical value, but can alter), certainty c (how sure is the model about the maid prediction)
        """
        raise NotImplementedError

    def train(self, x, y):
        """
        Train the ML model with one instance (can internally store the training instances in order to achieve batch training)

        :param x: input values
        :param y: label (correct output)
        """
        raise NotImplementedError
