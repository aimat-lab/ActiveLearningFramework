from typing import Tuple, Optional, Iterable, Sequence

from additional_component_interfaces import ReadOnlyPassiveLearner
from helpers import X, Y, AddInfo_Y


class PassiveLearner(ReadOnlyPassiveLearner):
    """
    Interface for the PL (to be extended SL model) => with the ability to write/change the SL model
    """

    def predict_set(self, xs: Iterable[X]) -> Sequence[Iterable[Y], Sequence[AddInfo_Y]]:
        raise NotImplementedError

    def load_model(self) -> None:
        raise NotImplementedError

    def close_model(self) -> None:
        raise NotImplementedError

    def pl_satisfies_evaluation(self) -> bool:
        raise NotImplementedError

    def save_model(self) -> None:
        """
        Save the model in separate file

        => necessary for multiprocessing compatibility: will be called after every other function of this class, except load_model

        e.g.: model.save('my_model'); del model
        """
        raise NotImplementedError

    def initial_training(self, x_train: Sequence[X], y_train: Sequence[Y], **kwargs) -> None:
        """
        Initial batch training => for determination of initial weights

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
