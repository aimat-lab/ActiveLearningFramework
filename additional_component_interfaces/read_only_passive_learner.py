from typing import Tuple, Optional

from helpers import X, Y, AddInfo_Y


class ReadOnlyPassiveLearner:

    def predict(self, x: X) -> Tuple[Y, Optional[AddInfo_Y]]:
        """
        Get the predicted output based on the current training state of the ML model

        :param x: input values
        :return: prediction y (usually one numerical value, but can alter), optionally additional information (e.g. uncertainty)
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

    def pl_satisfies_evaluation(self) -> bool:
        """
        Evaluates the performance of the current SL model (pl) => decides if pl is trained enough (satisfies performance acceptance criterion)
            - can be based on history (performance of pl doesn't get significantly better/gets worse)
            - can be based on threshold => predictions are accurate enough

        :return: whether pl is trained well enough
        """
        raise NotImplementedError
