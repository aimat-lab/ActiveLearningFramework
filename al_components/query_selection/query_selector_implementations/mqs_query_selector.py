from dataclasses import dataclass

from numpy import ndarray

from al_components.query_selection import QuerySelector
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import X
from workflow_management.database_interfaces import CandidateSet


def decide_discard(info):
    return info < 0.7


# noinspection PyPep8Naming
@dataclass()
class MQS_QuerySelector(QuerySelector):  # TODO: currently, this selector is the same as the sbs one => maybe change?
    info_analyser: InformativenessAnalyser
    candidate_set: CandidateSet

    def select_query_instance(self) -> (X, bool):
        (x, _, _) = self.candidate_set.get_first_instance()
        info = self.info_analyser.get_informativeness(x)
        if decide_discard(info):
            return x, False
        else:
            return x, True
