from al_components.candidate_update.candidate_updater_implementations import Pool
from al_components.query_selection import QuerySelector
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import X

from helpers.exceptions import IncorrectParameters


# noinspection PyPep8Naming
class PbS_QuerySelector(QuerySelector):
    """"
    Implementation of the query selector for the PbS scenario => will evaluate the whole candidate pool
    """

    def __init__(self, info_analyser: InformativenessAnalyser, candidate_set: Pool):
        if (candidate_set is None) or (not isinstance(candidate_set, Pool)) or (info_analyser is None) or (not isinstance(info_analyser, InformativenessAnalyser)):
            raise IncorrectParameters("PbS_CandidateUpdater needs to be initialized with a candidate_set (of type Pool) and info_analyser (of type InformativenessAnalyser)")
        else:
            self.candidate_set = candidate_set
            self.info_analyser = info_analyser

    def select_query_instance(self) -> (X, float, bool):
        """
        Evaluates the whole candidate set => selects instance maximizing the informativeness measure

        :return: the maximizing instance, informativeness value, True (instances in pbs scenario are never discarded => just stay in pool)
        """
        (xs, _) = self.candidate_set.retrieve_all_instances()

        max_x, max_info = None, -1
        for x in xs:
            info = self.info_analyser.get_informativeness(x)
            if max_info < info:
                max_x = x
                max_info = info

        return max_x, max_info, True
