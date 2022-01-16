import logging
import time
from multiprocessing.managers import ValueProxy

from al_components.query_selection import QuerySelector, init_query_selector
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import Scenarios, SystemStates
from helpers.exceptions import NoNewElementException
from workflow_management.database_interfaces import QuerySet, CandidateSet


class ActiveLearnerController:

    def __init__(self, candidate_set: CandidateSet, query_set: QuerySet, scenario: Scenarios, info_analyser: InformativenessAnalyser):
        self.candidate_set: CandidateSet = candidate_set
        self.query_set: QuerySet = query_set
        self.query_selector: QuerySelector = init_query_selector(scenario, info_analyser, candidate_set)

    def training_job(self, system_state: ValueProxy):
        if system_state.value > int(SystemStates.TRAINING):
            logging.info(f"Ended training job of active learner, system_state={SystemStates(system_state.value).name}")
            return
        try:
            while True:
                query_instance, is_queried = self.query_selector.select_query_instance()
                if is_queried:
                    self.candidate_set.remove_instance(query_instance)
                    self.query_set.add_instance(query_instance)
                    logging.info(f"New unlabelled queried: x = `{query_instance}`")
                    break
                else:
                    self.candidate_set.remove_instance(query_instance)
                    logging.info(f"Discarded the candidate: x = `{query_instance}`")

        except NoNewElementException:
            logging.info("Wait for new candidates")
            time.sleep(5)

        self.training_job(system_state)
        return
