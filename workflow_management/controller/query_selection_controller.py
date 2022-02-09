import logging
import time
from multiprocessing.managers import ValueProxy

from al_components.query_selection import QuerySelector, init_query_selector
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import Scenarios, SystemStates
from helpers.exceptions import NoNewElementException
from workflow_management.database_interfaces import QuerySet, CandidateSet, LogQueryDecisionDB

log = logging.getLogger("Query selection controller")


class QuerySelectionController:
    """
    Controls workflow of active choosing of training data => query selection
    """

    def __init__(self, candidate_set: CandidateSet, log_query_decision_db: LogQueryDecisionDB, query_set: QuerySet, scenario: Scenarios, info_analyser: InformativenessAnalyser):
        """
        Set arguments for al, initialize query selector

        :param candidate_set: dataset containing candidates for the query selection => concrete implementation depends on scenario
        :param query_set: dataset in which queries will be posted
        :param scenario: the al scenario
        :param info_analyser: evaluator of informativeness
        """

        log.info("Init query selection controller")

        log.debug("Set candidate set, log query decision database, and query set")
        self.candidate_set: CandidateSet = candidate_set
        self.log_query_decision_db: LogQueryDecisionDB = log_query_decision_db
        self.query_set: QuerySet = query_set

        self.query_selector: QuerySelector = init_query_selector(scenario, info_analyser, candidate_set)
        log.debug(f"Initialized the query selector, scenario: {scenario} => query selector type: {type(self.query_selector)}")

    def training_job(self, system_state: ValueProxy):
        """
        Actual training job of the active learner controller => selects query
        Also including the soft training end job

        Job sequence:
            1. retrieve candidates
                1. if candidates available:
                    1. use query selector to actively select next training instance => discard not queried candidates; log decision in log db
                    2. query instance (add to query set)
                    3. restart job
                1. else:
                    1. sleep (or set state to FINISH_TRAINING__ORACLE and return if state is FINISH_TRAINING__INFO)
                    2. restart job

        :param system_state: The current system state (shared over all controllers, values align with enum SystemStates)
        :return: if the process should end => indicated by system_state
        """
        try:

            if system_state.value >= int(SystemStates.TERMINATE_TRAINING):
                log.warning(f"Training process terminated => end training job of active learner, system_state={SystemStates(system_state.value).name}")
                return

            try:
                while True:
                    query_instance, info_value, gets_queried = self.query_selector.select_query_instance()
                    (_, additional_info) = self.candidate_set.get_instance(query_instance)
                    self.log_query_decision_db.add_instance(x=query_instance, info_value=info_value, queried=gets_queried, additional_info=additional_info)
                    if gets_queried:
                        self.candidate_set.remove_instance(query_instance)
                        self.query_set.add_instance(query_instance)
                        log.info(f"Selected new unlabelled query: x = `{query_instance}`")
                        break
                    else:
                        self.candidate_set.remove_instance(query_instance)
                        log.info(f"Discarded the candidate: x = `{query_instance}`")

            except NoNewElementException:
                if system_state.value == int(SystemStates.FINISH_TRAINING__INFO):
                    log.info("Candidate database empty, all candidates evaluated => only Oracle and PL need to softly end training")

                    system_state.set(int(SystemStates.FINISH_TRAINING__ORACLE))
                    log.warning(f"Enter Oracle finish training state => soft end (set system_state: {SystemStates(system_state.value).name})")

                    return

                else:
                    log.info("Wait for new candidates")
                    time.sleep(5)

            self.training_job(system_state)
            return

        except Exception as e:
            log.error("An error occurred during the execution of query selector training job => terminate system", e)
            system_state.set(int(SystemStates.ERROR))
            return
