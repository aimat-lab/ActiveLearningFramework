import logging

from al_specific_components.query_selection import QuerySelector
from al_specific_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import X
from workflow_management.database_interfaces import CandidateSet

log = logging.getLogger("SbS query selector")


# noinspection PyPep8Naming
class SbS_QuerySelector(QuerySelector):

    def __init__(self, info_analyser: InformativenessAnalyser, candidate_set: CandidateSet):
        log.debug("Initialize the SbS query selector with the parameters info_analyser and candidate_set")
        self.info_analyser = info_analyser
        self.candidate_set = candidate_set

    # noinspection PyMethodMayBeStatic
    def decide_discard(self, info: float) -> bool:
        # TODO: maybe not static/hard implemented => instead default method and one that can be inserted in implementation?
        # TODO: if threshold kept => what should be value?? => should value adapt over time
        return info < 0.5

    def select_query_instance(self) -> (X, float, bool):
        """
        Get the first element (ordered by time of insertion) of the candidate set and decide based on informativeness whether to discard or query it

        - instance can be generated
        - discarded: remove instance permanently from candidate set

        :return: the evaluated instance, informativeness value, [True if instance should be queried, False if instance should be discarded]
        """
        log.debug("Retrieve candidate")
        (x, _) = self.candidate_set.get_first_instance()
        log.debug(f"Evaluate informativeness of candidate x: {x}")
        info = self.info_analyser.get_informativeness(x)
        log.debug(f"Informativeness for x: info={info}, x={x}")
        if self.decide_discard(info):
            log.debug(f"Decided to discard {x}")
            return x, info, False
        else:
            log.debug(f"Decided to query {x}")
            return x, info, True
