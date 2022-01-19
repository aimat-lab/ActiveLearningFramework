import logging
from typing import Callable

from al_components.candidate_update import CandidateUpdater
from al_components.candidate_update.candidate_updater_implementations import MQS_CandidateUpdater, PbS_CandidateUpdater, SbS_CandidateUpdater, Generator, Stream, Pool
from helpers import Scenarios, X, Y, AddInfo_Y, CandInfo


def get_candidate_additional_information(x: X, prediction: Y, additional_prediction_info: AddInfo_Y) -> CandInfo:
    """
    To be implemented function (case dependent) => use this function as orientation

     => will take the values provided through the prediction of the PL and convert it into the information stored for the candidates (additionally to the input)

    :param x: input of the future candidate
    :param prediction: the predicted output
    :param additional_prediction_info: the additional information provided from the PL about its predictions (e.g., uncertainty about prediction)
    :return: the information for the candidate stored in the candidate set additionally to the raw input (e.g., the uncertainty)
    """
    # TODO case implementation: implement this function
    raise NotImplementedError


def get_candidate_source_type(scenario: Scenarios) -> type:
    """
    Get the type of the candidate source needed for an implementation of the provided scenario
        - PbS: Pool
        - SbS: Stream
        - MQS: Generator

    :param scenario: of the current AL project
    :return: the correct type
    """

    if scenario == Scenarios.PbS:
        return Pool
    elif scenario == Scenarios.SbS:
        return Stream
    else:  # scenario == Scenarios.MQS
        return Generator


def init_candidate_updater(scenario: Scenarios, info_creator: Callable[[X, Y, AddInfo_Y], CandInfo], **kwargs) -> CandidateUpdater:
    """
    Initialize the candidate updater, dependent on scenario


    **Arguments**:

    - *SbS*: source_stream: Stream (source for new candidates), candidate_set: CandidateSet (where new candidates get entered to), pl: PassiveLearner
    - *PbS*: candidate_set: Pool (and also CandidateSet, will function as source for candidates and as candidate target), pl: PassiveLearner
    - *MQS*: not implemented  # TODO: if MQS candidate updater is implemented, add description

    :param scenario: the selected scenario
    :param info_creator: see get_candidate_additional_information
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
