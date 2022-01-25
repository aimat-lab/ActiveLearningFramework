import logging

from al_components.candidate_update.candidate_updater_implementations import Pool
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from al_components.query_selection.query_selector_implementations import MQS_QuerySelector, PbS_QuerySelector, SbS_QuerySelector
from helpers import Scenarios
from workflow_management.database_interfaces import CandidateSet

query_selector = {
    Scenarios.PbS: PbS_QuerySelector,
    Scenarios.SbS: SbS_QuerySelector,
    Scenarios.MQS: MQS_QuerySelector
}


# TODO will all query selectors have the same arguments? => if not, update
def init_query_selector(scenario: Scenarios, info_analyser: InformativenessAnalyser, candidate_set: CandidateSet or Pool):
    logging.info(f"Initialize {scenario.name} query selector")
    return query_selector[scenario](info_analyser=info_analyser, candidate_set=candidate_set)
