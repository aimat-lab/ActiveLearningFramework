import logging
import time
from multiprocessing.managers import ValueProxy

from al_components.query_selection import QuerySelector, init_query_selector
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import Scenarios, SystemStates
from helpers.exceptions import NoNewElementException
from workflow_management.database_interfaces import QuerySet, CandidateSet

al_controller_logging_prefix = "al_controller: "


class ActiveLearnerController:
    """
    Controls workflow of active choosing of training data => query selection
    """

    def __init__(self, candidate_set: CandidateSet, query_set: QuerySet, scenario: Scenarios, info_analyser: InformativenessAnalyser):
        """
        Set arguments for al, initialize query selector

        :param candidate_set: dataset containing candidates for the query selection => concrete implementation depends on scenario
        :param query_set: dataset in which queries will be posted
        :param scenario: the al scenario
        :param info_analyser: evaluator of informativeness
        """

        logging.info(f"{al_controller_logging_prefix} Init active learner controller => set candidate set, query_set, init query_selector")

        self.candidate_set: CandidateSet = candidate_set
        self.query_set: QuerySet = query_set
        self.query_selector: QuerySelector = init_query_selector(scenario, info_analyser, candidate_set)

    def training_job(self, system_state: ValueProxy):
        """
        Actual training job of the active learner controller => selects query
        Also including the soft training end job

        Job sequence:
            1. retrieve candidates
                1. if candidates available:
                    1. use query selector to actively select next training instance => discard not queried candidates
                    2. query instance (add to query set)
                    3. restart job
                1. else:
                    1. sleep (or set state to FINISH_TRAINING__ORACLE and return if state is FINISH_TRAINING__AL)
                    2. restart job

        :param system_state: The current system state (shared over all controllers, values align with enum SystemStates)
        :return: if the process should end => indicated by system_state
        """

        if system_state.value > int(SystemStates.TRAINING):
            logging.warning(f"{al_controller_logging_prefix} Training process terminated => end training job of active learner, system_state={SystemStates(system_state.value).name}")
            return

        try:
            while True:
                query_instance, is_queried = self.query_selector.select_query_instance()
                if is_queried:
                    self.candidate_set.remove_instance(query_instance)
                    self.query_set.add_instance(query_instance)
                    logging.info(f"{al_controller_logging_prefix} Selected new unlabelled queried: x = `{query_instance}`")
                    break
                else:
                    self.candidate_set.remove_instance(query_instance)
                    logging.info(f"{al_controller_logging_prefix} Discarded the candidate: x = `{query_instance}`")

        except NoNewElementException:
            if system_state.value == int(SystemStates.FINISH_TRAINING__AL):
                logging.info(f"{al_controller_logging_prefix} Candidate database empty, all candidates evaluated => only Oracle and PL need to softly end training")

                system_state.set(int(SystemStates.FINISH_TRAINING__ORACLE))
                logging.warning(f"{al_controller_logging_prefix} Enter Oracle finish training state => soft end (set system_state: {SystemStates(system_state.value).name})")

                return

            else:
                logging.info(f"{al_controller_logging_prefix} Wait for new candidates")
                time.sleep(5)

        self.training_job(system_state)
        return
