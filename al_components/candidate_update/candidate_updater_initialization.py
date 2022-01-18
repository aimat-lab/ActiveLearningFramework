import logging

from al_components.candidate_update import CandidateInformationCreator
from al_components.candidate_update.candidate_updater_implementations import MQS_CandidateUpdater, PbS_CandidateUpdater, SbS_CandidateUpdater
from helpers import Scenarios


def init_candidate_updater(scenario: Scenarios, info_creator: CandidateInformationCreator, **kwargs):
    """
    Initialize the candidate updater, dependent on scenario


    **Arguments**:

    - *SbS*: source_stream: Stream (source for new candidates), candidate_set: CandidateSet (where new candidates get entered to), pl: PassiveLearner
    - *PbS*: candidate_set: Pool (and also CandidateSet, will function as source for candidates and as candidate target), pl: PassiveLearner
    - *MQS*: not implemented  # TODO: if MQS candidate updater is implemented, add description

    :param scenario: the selected scenario
    :param info_creator: class that can generate the stored information for candidates provided the instance, prediction and additional_prediction_info
    :param kwargs: arguments depending on the scenario (see description arguments)
    :return: the scenario dependent candidate updater
    """
    if scenario == Scenarios.MQS:
        logging.info("Initialize MQS candidate updater")
        logging.warning("MQS candidate updater is not yet implemented!!")  # TODO: if MQS candidate updater is implemented, remove warning
        return MQS_CandidateUpdater(info_creator)
    elif scenario == Scenarios.PbS:
        logging.info("Initialize PbS candidate updater")
        candidate_set = kwargs.get("candidate_set")
        pl = kwargs.get("pl")
        return PbS_CandidateUpdater(info_creator, candidate_set, pl)
    else:  # scenario == Scenarios.SbS:
        logging.info("Initialize SbS candidate updater")
        candidate_set = kwargs.get("candidate_set")
        pl = kwargs.get("pl")
        source_stream = kwargs.get("candidate_source")
        return SbS_CandidateUpdater(info_creator, candidate_set, source_stream, pl)
