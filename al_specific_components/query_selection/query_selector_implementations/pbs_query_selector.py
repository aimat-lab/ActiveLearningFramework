import logging

from al_specific_components.candidate_update.candidate_updater_implementations import Pool
from al_specific_components.query_selection import QuerySelector, InformativenessAnalyser
from helpers import X, framework_properties
from helpers.exceptions import IncorrectParameters

log = logging.getLogger("PbS query selector")


# noinspection PyPep8Naming
class PbS_QuerySelector(QuerySelector):
    """"
    Implementation of the query selector for the PbS scenario => will evaluate the whole candidate pool
    """

    def __init__(self, info_analyser: InformativenessAnalyser, candidate_set: Pool):
        log.debug("Initialize the PbS query selector with the parameters info_analyser and candidate_set")
        if (candidate_set is None) or (not isinstance(candidate_set, Pool)) or (info_analyser is None) or (not isinstance(info_analyser, InformativenessAnalyser)):
            raise IncorrectParameters("PbS_CandidateUpdater needs to be initialized with a candidate_set (of type Pool) and info_analyser (of type InformativenessAnalyser)")
        else:
            self.candidate_set = candidate_set
            self.info_analyser = info_analyser
            self.store_sorted_list = []
            self.refresh_counter = 0

    def select_query_instance(self) -> (X, float, bool):
        """
        Evaluates the whole candidate set => selects instance maximizing the informativeness measure

        :return: the maximizing instance, informativeness value, True (instances in pbs scenario are never discarded => just stay in pool)
        """
        log.debug("Get all candidates from pool")
        (xs, _) = self.candidate_set.retrieve_all_instances()

        log.debug("Evaluate informativeness for all instances => find maximizing instance")

        max_number_refresh_counter = framework_properties.pbs_refresh_counter_sorted_list
        max_info: float
        max_x: X

        if self.refresh_counter == 0:
            self.store_sorted_list = sorted([(self.info_analyser.get_informativeness(x), x) for x in xs], key=lambda x: x[0], reverse=True)
            self.refresh_counter = max_number_refresh_counter
            max_info, max_x = self.store_sorted_list[max_number_refresh_counter - self.refresh_counter]
        else:
            self.refresh_counter -= 1
            max_info, max_x = self.store_sorted_list[max_number_refresh_counter - self.refresh_counter]

        log.info(f"Found maximizing instance")
        return max_x, max_info, True
