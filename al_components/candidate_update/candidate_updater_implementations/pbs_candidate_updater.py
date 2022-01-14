import logging

from additional_component_interfaces import PassiveLearner
from al_components.candidate_update import CandidateUpdater
from helpers.exceptions import IncorrectParameters, EndTrainingException
from workflow_management.database_interfaces import CandidateSet


class Pool(CandidateSet):

    def initiate_pool(self, x_initial):
        """
        Pool needs to be initialized with a list of input instances
        :param x_initial: all instances available for
        """
        raise NotImplementedError

    def update_instances(self, xs, new_y_predictions, new_certainties):
        """
        alter the prediction and uncertainty for the provided candidates (identified by provided input)

        :param xs: array of input values
        :param new_y_predictions: array of new predictions
        :param new_certainties: array of new certainties about prediction
        
        :raises NoSuchElement: if instance identified through x does not exist
        """
        raise NotImplementedError

    def retrieve_all_instances(self):
        """
        retrieves all candidates from database (database is left unchanged)

        :return tuple of numpy arrays [x], [prediction], [uncertainty]
        :raises NoNewElementException: if no instance is in database
        """
        raise NotImplementedError


class PbS_CandidateUpdater(CandidateUpdater):

    def __init__(self, candidate_set: Pool, pl: PassiveLearner):
        if (candidate_set is None) or (not isinstance(candidate_set, Pool)) or (pl is None) or (not isinstance(pl, PassiveLearner)):
            raise IncorrectParameters("PbS_CandidateUpdater needs to be initialized with a candidate_set (of type Pool) and pl (of type PassiveLearner)")
        else:
            self.candidate_set = candidate_set
            self.pl = pl

    def update_candidate_set(self):
        (xs, _, _) = self.candidate_set.retrieve_all_instances()
        if len(xs) == 0:
            raise EndTrainingException("Pool is empty => no more candidates")

        predictions, uncertainties = [], []
        for x in xs:
            prediction, uncertainty = self.pl.predict(x)
            predictions.append(prediction)
            uncertainties.append(uncertainty)  # TODO uncertainty

        self.candidate_set.update_instances(xs, predictions, uncertainties)
        logging.info("updated whole candidate pool with new predictions and uncertainties")
