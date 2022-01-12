from additional_component_interfaces import PassiveLearner
from al_components.candidate_update import CandidateUpdater
from helpers.exceptions import IncorrectParameters
from workflow_management.database_interfaces import CandidateSet


class Pool(CandidateSet):

    def initiate_pool(self, x_initial):
        """
        Pool needs to be initialized with a list of input instances
        :param x_initial: all instances available for
        """
        raise NotImplementedError

    def update_instance(self, x, new_y_prediction, new_certainty):
        """
        alter the prediction and certainty for a candidate (identified by provided input)

        :param new_y_prediction: the new prediction
        :param x: input values
        :param new_certainty: the new certainty about prediction
        :raises NoSuchElement: if instance identified through x does not exist
        """
        raise NotImplementedError

    def retrieve_all_instances(self):
        """
        retrieves all candidates from database (database is left unchanged)

        :return tuple of numpy arrays [x], [prediction], [certainty]
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
        # TODO: implement (retrieve all instances, add predictions to them, update them)
        raise NotImplementedError
