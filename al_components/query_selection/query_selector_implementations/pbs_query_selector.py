from dataclasses import dataclass

from al_components.query_selection import QuerySelector
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from workflow_management.database_interfaces import CandidateSet, QuerySet


@dataclass()
class PbS_QuerySelector(QuerySelector):
    info_analyser: InformativenessAnalyser
    candidate_set: CandidateSet
    query_set: QuerySet

    def select_query_instance(self):
        (xs, _, _) = self.candidate_set.retrieve_all_instances()
        max_x, max_info = None, -1
        for x in xs:
            info = self.info_analyser.get_informativeness(x)
            if max_info < info:
                max_x = x
                max_info = info
        self.query_set.add_instance(max_x)
