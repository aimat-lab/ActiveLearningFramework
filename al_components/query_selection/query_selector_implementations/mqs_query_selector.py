from dataclasses import dataclass

from al_components.query_selection import QuerySelector, InformativenessAnalyser
from workflow_management import CandidateSet, QuerySet


def decide_discard(info):
    return info < 0.7


@dataclass()
class MQS_QuerySelector(QuerySelector):  # TODO: currently, this selector is the same as the sbs one => maybe change?
    info_analyser: InformativenessAnalyser
    candidate_set: CandidateSet
    query_set: QuerySet

    def select_query_instance(self):
        (x, _, _) = self.candidate_set.get_instance()
        info = self.info_analyser.get_informativeness(x)
        self.candidate_set.remove_instance(x)
        if decide_discard(info):
            self.select_query_instance()
        else:
            self.query_set.add_instance(x)
