import logging

from al_components.query_selection import QuerySelector, init_query_selector
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import Scenarios
from workflow_management.database_interfaces import QuerySet, CandidateSet


class ActiveLearnerController:

    def __init__(self, candidate_set: CandidateSet, query_set: QuerySet, scenario: Scenarios, info_analyser: InformativenessAnalyser):
        self.candidate_set: CandidateSet = candidate_set
        self.query_set: QuerySet = query_set
        self.query_selector: QuerySelector = init_query_selector(scenario, info_analyser, candidate_set, query_set)

    def training_job(self):
        # TODO loop
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
