from dataclasses import dataclass

from al_components.query_selection import QuerySelector
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from workflow_management.database_interfaces import CandidateSet, QuerySet


def decide_discard(info):
    return info < 0.7


@dataclass()
class MQS_QuerySelector(QuerySelector):  # TODO: currently, this selector is the same as the sbs one => maybe change?
    info_analyser: InformativenessAnalyser
    candidate_set: CandidateSet

    def select_query_instance(self):
        (x, _, _) = self.candidate_set.get_first_instance()
        info = self.info_analyser.get_informativeness(x)
        if decide_discard(info):
            return x, False
        else:
            return x, True
