from dataclasses import dataclass

from al_components.candidate_update.candidate_updater_implementations import Pool
from al_components.query_selection import QuerySelector
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import X


# noinspection PyPep8Naming
@dataclass()
class PbS_QuerySelector(QuerySelector):
    """"
    Implementation of the query selector for the PbS scenario => will evaluate the whole candidate pool
    """
    info_analyser: InformativenessAnalyser
    candidate_set: Pool

    def select_query_instance(self) -> (X, bool):
        (xs, _) = self.candidate_set.retrieve_all_instances()

        max_x, max_info = None, -1
        for x in xs:
            info = self.info_analyser.get_informativeness(x)
            if max_info < info:
                max_x = x
                max_info = info

        return max_x, True
