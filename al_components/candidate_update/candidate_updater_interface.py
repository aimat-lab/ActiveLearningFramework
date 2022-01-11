import logging

from al_components import PbS_CandidateUpdater
from al_components.candidate_update.candidate_updater_implementations import MQS_CandidateUpdater
from helpers import Scenarios


class CandidateUpdater:

    def update_candidate_set(self):
        raise NotImplementedError


def init_candidate_updater(scenario: Scenarios, **kwargs):
    """
    Initialize the candidate updater, dependent on scenario


    **Arguments**:

    - *SbS*: source_stream: Stream (source for new candidates), candidate_set: CandidateSet (where new candidates get entered to), pl: PassiveLearner
    - *PbS*: candidate_set: CandidateSet (will function as source for candidates and as candidate target), pl: PassiveLearner
    - *MQS*: not implemented  # TODO: if MQS candidate updater is implemented, add description

    :param scenario: the selected scenario
    :param kwargs: arguments depending on the scenario (see description arguments)
    :return: the scenario dependent candidate updater
    """
    if scenario == Scenarios.MQS:
        logging.info("Initialize MQS candidate updater")
        logging.warning("MQS candidate updater is not yet implemented!!")  # TODO: if MQS candidate updater is implemented, remove warning
        return MQS_CandidateUpdater()
    elif scenario == Scenarios.PbS:
        logging.info("Initialize PbS candidate updater")
        candidate_set = kwargs.get("candidate_set")
        pl = kwargs.get("pl")
        return PbS_CandidateUpdater(candidate_set, pl)
    else:  # scenario == Scenarios.SbS:
        logging.info("Initialize SbS query selector")
        return SbS_QuerySelector(info_analyser, candidate_set, query_set)
