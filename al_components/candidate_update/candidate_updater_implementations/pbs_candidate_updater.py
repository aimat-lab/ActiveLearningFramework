import logging
from typing import Tuple, List, Optional, Callable

from numpy import ndarray

from additional_component_interfaces import PassiveLearner
from al_components.candidate_update import CandidateUpdater, CandidateInformationCreator
from helpers import CandInfo, X, Y, AddInfo_Y
from helpers.exceptions import IncorrectParameters, NoMoreCandidatesException, NoNewElementException
from workflow_management.database_interfaces import CandidateSet


# TODO: implement alternative pool/pbs candidate updater => only part of pool at a time updates (maybe extra scenario)

class Pool(CandidateSet):
    """
    Datasource for the candidate updater in a PbS scenario => extends the candidate set (whole pool represents the candidates)
        - initialized with unlabelled instances (fetched from 'natural distribution')
    """

    def get_first_instance(self) -> Tuple[X, CandInfo]:
        raise NotImplementedError

    def get_instance(self, x: X) -> Tuple[X, CandInfo]:
        raise NotImplementedError

    def remove_instance(self, x: X) -> None:
        raise NotImplementedError

    def add_instance(self, x: X, additional_info: CandInfo = None) -> None:
        raise NotImplementedError

    def is_empty(self) -> bool:
        raise NotImplementedError

    def initiate_pool(self, x_initial: List[X] or ndarray) -> None:
        """
        Pool needs to be initialized with a list of input instances

        :param x_initial: array of inputs (array of array)
        """
        raise NotImplementedError

    def update_instances(self, xs: List[X] or ndarray, new_additional_infos: List[CandInfo] or ndarray = None) -> None:
        """
        alter the prediction and uncertainty for the provided candidates (identified by provided input)

        :param xs: array of input values
        :param new_additional_infos: array containing new additional information (optional => if candidate set doesn't include additional information, can be None)

        :raises NoSuchElement: if an instance within xs does not exist
        """
        # TODO: if no such element maybe just ignore update for this instance?
        raise NotImplementedError

    def retrieve_all_instances(self) -> Tuple[List[X] or ndarray, Optional[List[CandInfo] or ndarray]]:
        """
        retrieves all candidates from database (database is left unchanged)

        :return: tuple of lists [x] (array of inputs), [addition_info] (array of additional information)
        :raises NoNewElementException: if no instance is in database
        """
        raise NotImplementedError


# noinspection PyPep8Naming
class PbS_CandidateUpdater(CandidateUpdater):
    # TODO: logging, documentation

    def __init__(self, cand_info_mapping: Callable[[X, Y, AddInfo_Y], CandInfo], candidate_set: Pool, pl: PassiveLearner):
        if (candidate_set is None) or (not isinstance(candidate_set, Pool)) or (pl is None) or (not isinstance(pl, PassiveLearner)):
            raise IncorrectParameters("PbS_CandidateUpdater needs to be initialized with a candidate_set (of type Pool) and pl (of type PassiveLearner)")
        else:
            self.candidate_set = candidate_set
            self.pl = pl
            self.info_creator = cand_info_mapping

    def update_candidate_set(self):
        # noinspection PyUnusedLocal
        xs = None
        try:
            (xs, _) = self.candidate_set.retrieve_all_instances()
        except NoNewElementException:
            raise NoMoreCandidatesException()
        if len(xs) == 0:
            raise NoMoreCandidatesException()

        candidate_information = []
        for x in xs:
            prediction, additional_information = self.pl.predict(x)
            # TODO: how to set additional information/candidate information => extra class with method
            candidate_information.append(self.info_creator(x, prediction, additional_information))

        self.candidate_set.update_instances(xs, candidate_information)
        logging.info("updated whole candidate pool")
