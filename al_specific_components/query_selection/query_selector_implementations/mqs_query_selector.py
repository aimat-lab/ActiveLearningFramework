import logging

from al_specific_components.query_selection import QuerySelector, InformativenessAnalyser
from helpers import X, framework_properties
from workflow_management.database_interfaces import CandidateSet

log = logging.getLogger("MQS query selector")


# noinspection PyPep8Naming
class MQS_QuerySelector(QuerySelector):  # TODO: currently, this selector is the same as the sbs one => maybe change?

    def __init__(self, info_analyser: InformativenessAnalyser, candidate_set: CandidateSet):
        log.debug("Initialize the MQS query selector with the parameters info_analyser and candidate_set")
        self.info_analyser = info_analyser
        self.candidate_set = candidate_set

    # noinspection PyMethodMayBeStatic
    def decide_discard(self, info: float) -> bool:
        return info < framework_properties.mqs_threshold_query_decision

    def select_query_instance(self) -> (X, float, bool):
        """
        Get the first element (ordered by time of insertion) of the candidate set and decide based on informativeness whether to discard or query it

        - instance can be generated
        - discarded: remove instance permanently from candidate set

        :return: the evaluated instance, informativeness value, [True if instance should be queried, False if instance should be discarded]
        """
        log.debug("Retrieve candidate")
        (x, _) = self.candidate_set.get_first_instance()
        log.debug(f"Evaluate informativeness of candidate x")
        info = self.info_analyser.get_informativeness(x)
        if self.decide_discard(info):
            log.debug(f"Decided to discard")
            return x, info, False
        else:
            log.debug(f"Decided to query")
            return x, info, True
