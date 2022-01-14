import logging

from al_components.candidate_update.candidate_updater_implementations import Pool
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from al_components.query_selection.query_selector_implementations import MQS_QuerySelector, PbS_QuerySelector, SbS_QuerySelector
from helpers import Scenarios
from helpers.exceptions import IncorrectParameters
from workflow_management.database_interfaces import CandidateSet


# TODO will all query selectors have the same arguments? => if not, update
def init_query_selector(scenario: Scenarios, info_analyser: InformativenessAnalyser, candidate_set: CandidateSet):
    if scenario == Scenarios.MQS:
        logging.info("Initialize MQS query selector")
        return MQS_QuerySelector(info_analyser, candidate_set)
    elif scenario == Scenarios.PbS:
        logging.info("Initialize PbS query selector")
        if isinstance(candidate_set, Pool):
            return PbS_QuerySelector(info_analyser, candidate_set)
        else:
            raise IncorrectParameters("Query selector in PbS scenario needs a Pool as the candidate_set")
    else:  # scenario == Scenarios.SbS:
        logging.info("Initialize SbS query selector")
        return SbS_QuerySelector(info_analyser, candidate_set)
