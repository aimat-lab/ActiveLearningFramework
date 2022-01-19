from dataclasses import dataclass

from al_components.query_selection import QuerySelector
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import X
from workflow_management.database_interfaces import CandidateSet


# noinspection PyPep8Naming
@dataclass()
class MQS_QuerySelector(QuerySelector):  # TODO: currently, this selector is the same as the sbs one => maybe change?
    info_analyser: InformativenessAnalyser
    candidate_set: CandidateSet

    # noinspection PyMethodMayBeStatic
    def decide_discard(self, info: float) -> bool:
        # TODO: maybe not static/hard implemented => instead default method and one that can be inserted in implementation?
        # TODO: if threshold kept => what should be value?? => should value adapt over time
        return info < 0.7

    def select_query_instance(self) -> (X, bool):
        """
        Get the first element (ordered by time of insertion) of the candidate set and decide based on informativeness whether to discard or query it

        - instance can be generated
        - discarded: remove instance permanently from candidate set

        :return: the evaluated instance, [True if instance should be queried, False if instance should be discarded]
        """
        (x, _) = self.candidate_set.get_first_instance()
        info = self.info_analyser.get_informativeness(x)
        if self.decide_discard(info):
            return x, False
        else:
            return x, True
