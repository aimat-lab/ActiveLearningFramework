import logging

from al_components.query_selection import QuerySelector
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from al_components.query_selection.query_selector_implementations import MQS_QuerySelector, PbS_QuerySelector, SbS_QuerySelector
from helpers import Scenarios
from workflow_management.database_interfaces import CandidateSet, QuerySet


# TODO will all query selectors have the same arguments? => if not, update
def init_query_selector(scenario: Scenarios, info_analyser: InformativenessAnalyser, candidate_set: CandidateSet, query_set: QuerySet) -> QuerySelector:
    if scenario == Scenarios.MQS:
        logging.info("Initialize MQS query selector")
        return MQS_QuerySelector(info_analyser, candidate_set, query_set)
    elif scenario == Scenarios.PbS:
        logging.info("Initialize PbS query selector")
        return PbS_QuerySelector(info_analyser, candidate_set, query_set)
    else:  # scenario == Scenarios.SbS:
        logging.info("Initialize SbS query selector")
        return SbS_QuerySelector(info_analyser, candidate_set, query_set)