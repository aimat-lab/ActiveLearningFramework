from typing import List, Tuple, Optional

from helpers import X, Y, AddInfo_Y


class PassiveLearner:
    """
    Interface for the PL (to be extended SL model)
    """

    def save_model(self) -> None:
        """
        Save the model in separate file

        => necessary for multiprocessing compatibility: will be called after every other function of this class, except load_model

        e.g.:
            - model.save('my_model')
            - del model
        """
        raise NotImplementedError

    def load_model(self) -> None:
        """
        Load the model form the separate file (see save_model)

        => necessary for multiprocessing compatibility: will be called before every other function of this class, except save_model

        e.g.:
            - model = tensorflow.keras.models.load_model('my_model')
        """
        raise NotImplementedError

    def initial_training(self, x_train: List[X], y_train: List[Y], **kwargs) -> None:
        """
        Initial batch training => for determination of initial weights and potentially for setting the scaler for the input data

        :param x_train: training input (array of input data)
        :param y_train: training labels (array of correct output data)
        :param kwargs: can contain additional properties for the training
        """
        raise NotImplementedError

    def predict(self, x: X) -> Tuple[Y, Optional[AddInfo_Y]]:
        """
        Get the predicted output based on the current training state of the ML model

        :param x: input values
        :return: prediction y (usually one numerical value, but can alter), optionally additional information (e.g. uncertainty)
        """
        raise NotImplementedError

    def train(self, x: X, y: Y) -> None:
        """
        Train the ML model with one instance (can internally store the training instances in order to achieve batch training)

        :param x: input values
        :param y: label (correct output)
        """
        raise NotImplementedError
