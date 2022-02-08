import logging
from typing import Callable

from al_components.candidate_update import CandidateUpdater
from al_components.candidate_update.candidate_updater_implementations import MQS_CandidateUpdater, PbS_CandidateUpdater, SbS_CandidateUpdater, Generator, Stream, Pool
from helpers import Scenarios, X, Y, AddInfo_Y, CandInfo

candidate_source_type = {
    Scenarios.PbS: Pool,
    Scenarios.SbS: Stream,
    Scenarios.MQS: Generator
}

candidate_updater = {
    Scenarios.PbS: PbS_CandidateUpdater,
    Scenarios.SbS: SbS_CandidateUpdater,
    Scenarios.MQS: MQS_CandidateUpdater
}


# noinspection PyUnusedLocal
def get_candidate_additional_information(x: X, prediction: Y, additional_prediction_info: AddInfo_Y) -> CandInfo:
    """
    To be implemented function (case dependent) => use this function as orientation

     => will take the values provided through the prediction of the PL and convert it into the information stored for the candidates (additionally to the input)

    :param x: input of the future candidate
    :param prediction: the predicted output
    :param additional_prediction_info: the additional information provided from the PL about its predictions (e.g., uncertainty about prediction)
    :return: the information for the candidate stored in the candidate set additionally to the raw input (e.g., the uncertainty)
    """
    # case implementation: implement this function
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
    return candidate_source_type[scenario]


def init_candidate_updater(scenario: Scenarios, cand_info_mapping: Callable[[X, Y, AddInfo_Y], CandInfo], **kwargs) -> CandidateUpdater:
    """
    Initialize the candidate updater, dependent on scenario


    **Arguments**:

    - *SbS*: candidate_source: Stream (source for new candidates), candidate_set: CandidateSet (where new candidates get entered to), ro_pl: ReadOnlyPassiveLearner
    - *PbS*: candidate_set: Pool (and also CandidateSet, will function as source for candidates and as candidate target), ro_pl: ReadOnlyPassiveLearner
    - *MQS*: not implemented  # TODO: if MQS candidate updater is implemented, add description

    :param scenario: the selected scenario
    :param cand_info_mapping: see get_candidate_additional_information
    :param kwargs: arguments depending on the scenario (see description arguments)
    :return: the scenario dependent candidate updater
    """

    logging.info(f"Initialize {scenario.name} candidate updater")
    if scenario == Scenarios.MQS:
        logging.warning("MQS candidate updater is not yet implemented!!")  # TODO: if MQS candidate updater is implemented, remove warning

    # get all possible arguments
    candidate_set = kwargs.get("candidate_set")
    ro_pl = kwargs.get("ro_pl")
    candidate_source = kwargs.get("candidate_source")

    return candidate_updater[scenario](cand_info_mapping=cand_info_mapping, candidate_set=candidate_set, ro_pl=ro_pl, candidate_source=candidate_source)
