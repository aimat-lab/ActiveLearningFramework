class PassiveLearner:
    """
    Interface for the PL (to be extended SL model)
    """

    def save_model(self):
        """
        Save the model in separate file

        => necessary for multiprocessing compatibility: will be called after every other function of this class, except load_model

        e.g.:
            - model.save('my_model')
            - del model
        """
        raise NotImplementedError

    def load_model(self):
        """
        Load the model form the separate file (see save_model)

        => necessary for multiprocessing compatibility: will be called before every other function of this class, except save_model

        e.g.:
            - model = tensorflow.keras.models.load_model('my_model')
        """
        raise NotImplementedError

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
        Get the predicted output based on the current system_state of the ML model (current training status)

        :param x: input values
        :return: prediction y (usually one numerical value, but can alter), uncertainty c (how sure is the model about the maid prediction)
        """
        raise NotImplementedError

    def train(self, x, y):
        """
        Train the ML model with one instance (can internally store the training instances in order to achieve batch training)

        :param x: input values
        :param y: label (correct output)
        """
        raise NotImplementedError
