from additional_component_interfaces import PassiveLearner
from al_components.candidate_update import CandidateUpdater
from helpers.exceptions import IncorrectParameters
from workflow_management.database_interfaces import CandidateSet


class PbS_CandidateUpdater(CandidateUpdater):

    def __init__(self, candidate_set: CandidateSet, pl: PassiveLearner):
        if (candidate_set is None) or (not isinstance(candidate_set, CandidateSet)) or (pl is None) or (not isinstance(pl, PassiveLearner)):
            raise IncorrectParameters("PbS_CandidateUpdater needs to be initialized with a candidate_set (of type CandidateSet) and pl (of type PassiveLearner)")
        else:
            self.candidate_set = candidate_set
            self.pl = pl

    def update_candidate_set(self):
        # TODO: implement (retrieve all instances, add predictions to them, update them)
        raise NotImplementedError
