from additional_component_interfaces import PassiveLearner
from al_components.candidate_update import CandidateUpdater
from helpers.exceptions import IncorrectParameters
from workflow_management.database_interfaces import CandidateSet


class Stream:

    def get_next_element(self):
        raise NotImplementedError


# noinspection PyPep8Naming
class SbS_CandidateUpdater(CandidateUpdater):

    def __init__(self, candidate_set: CandidateSet, source_stream: Stream, pl: PassiveLearner):
        if (candidate_set is None) or (not isinstance(candidate_set, CandidateSet)) or (pl is None) or (not isinstance(source_stream, Stream)) or (source_stream is None) or (not isinstance(pl, PassiveLearner)):
            raise IncorrectParameters("SbS_CandidateUpdater needs to be initialized with a candidate_set (of type CandidateSet), a source_stream (of type Stream), and pl (of type PassiveLearner)")
        else:
            self.candidate_set = candidate_set
            self.source = source_stream
            self.pl = pl

    def update_candidate_set(self):
        # TODO: implement (fetch element from source stream, add prediction, enter into candidate) (including logging)
        raise NotImplementedError
