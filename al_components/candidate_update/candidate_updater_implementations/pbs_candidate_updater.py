import logging
from typing import Tuple, Any

from numpy import ndarray

from additional_component_interfaces import PassiveLearner
from al_components.candidate_update import CandidateUpdater
from helpers.exceptions import IncorrectParameters, NoMoreCandidatesException, NoNewElementException
from workflow_management.database_interfaces import CandidateSet


# TODO: implement alternative pool/pbs candidate updater => only part of pool at a time updates (maybe extra scenario)

class Pool(CandidateSet):
    """
    Datasource for the candidate updater in a PbS scenario => extends the candidate set (whole pool represents the candidates)
        - initialized with unlabelled instances (fetched from 'natural distribution')
    """

    def is_empty(self) -> bool:
        raise NotImplementedError

    def add_instance(self, x: ndarray, y_prediction: ndarray, uncertainty: Any) -> None:
        raise NotImplementedError

    def get_first_instance(self) -> Tuple[ndarray, ndarray, Any]:
        raise NotImplementedError

    def get_instance(self, x: ndarray) -> Tuple[ndarray, ndarray, Any]:
        raise NotImplementedError

    def remove_instance(self, x: ndarray) -> None:
        raise NotImplementedError

    def initiate_pool(self, x_initial: ndarray) -> None:
        """
        Pool needs to be initialized with a list of input instances

        :param x_initial: array of inputs (array of array)
        """
        raise NotImplementedError

    def update_instances(self, xs: ndarray, new_y_predictions: ndarray, new_certainties: ndarray) -> None:
        """
        alter the prediction and uncertainty for the provided candidates (identified by provided input)

        :param xs: array of input values (array of array)
        :param new_y_predictions: array of new predictions (array of array)
        :param new_certainties: array of new certainties about prediction (array of Any)
        
        :raises NoSuchElement: if instance identified through x does not exist
        """
        raise NotImplementedError

    def retrieve_all_instances(self) -> Tuple[ndarray, ndarray, ndarray]:
        """
        retrieves all candidates from database (database is left unchanged)

        :return: tuple of numpy arrays [x] (array of arrays), [prediction] (array of arrays), [uncertainty] (array of Any)
        :raises NoNewElementException: if no instance is in database
        """
        raise NotImplementedError


# noinspection PyPep8Naming
class PbS_CandidateUpdater(CandidateUpdater):
    # TODO: logging, documentation

    def __init__(self, candidate_set: Pool, pl: PassiveLearner):
        if (candidate_set is None) or (not isinstance(candidate_set, Pool)) or (pl is None) or (not isinstance(pl, PassiveLearner)):
            raise IncorrectParameters("PbS_CandidateUpdater needs to be initialized with a candidate_set (of type Pool) and pl (of type PassiveLearner)")
        else:
            self.candidate_set = candidate_set
            self.pl = pl

    def update_candidate_set(self):
        # noinspection PyUnusedLocal
        xs = None
        try:
            (xs, _, _) = self.candidate_set.retrieve_all_instances()
        except NoNewElementException:
            raise NoMoreCandidatesException()
        if len(xs) == 0:
            raise NoMoreCandidatesException()

        predictions, uncertainties = [], []
        for x in xs:
            prediction, uncertainty = self.pl.predict(x)
            predictions.append(prediction)
            uncertainties.append(uncertainty)

        self.candidate_set.update_instances(xs, predictions, uncertainties)
        logging.info("updated whole candidate pool with new predictions and uncertainties")
