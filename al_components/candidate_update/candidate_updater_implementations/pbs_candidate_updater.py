import logging
from typing import Tuple, List, Optional, Callable

from numpy import ndarray

from additional_component_interfaces import ReadOnlyPassiveLearner
from al_components.candidate_update import CandidateUpdater
from helpers import CandInfo, X, Y, AddInfo_Y
from helpers.exceptions import IncorrectParameters, NoMoreCandidatesException, NoNewElementException
from workflow_management.database_interfaces import CandidateSet

pbs_cand_updater_logging_prefix = "PbS Candidate Updater: "


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

        - if an instance x doesn't exist: it will be added with the according information

        :param xs: array of input values
        :param new_additional_infos: array containing new additional information (optional => if candidate set doesn't include additional information, can be None)
        """
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
    """
    The candidate updater within a PbS scenario => whole pool (= candidate set) will be updated with new predictions/information
    """

    # noinspection PyUnusedLocal
    def __init__(self, cand_info_mapping: Callable[[X, Y, AddInfo_Y], CandInfo], candidate_set: Pool, ro_pl: ReadOnlyPassiveLearner, **kwargs):
        if (candidate_set is None) or (not isinstance(candidate_set, Pool)) or (pl is None) or (not isinstance(ro_pl, ReadOnlyPassiveLearner)):
            raise IncorrectParameters("PbS_CandidateUpdater needs to be initialized with a candidate_set (of type Pool) and ro_pl (of type ReadOnlyPassiveLearner)")
        else:
            self.candidate_set = candidate_set
            self.ro_pl = ro_pl
            self.cand_info_mapping = cand_info_mapping
            logging.info(f"{pbs_cand_updater_logging_prefix} successfully initiated the candidate updater")

    def update_candidate_set(self) -> None:
        """
        Update of the whole pool/candidate set

        Add new predictions/information to every instance within pool

        :raise NoMoreCandidatesException if pool is empty
        """

        # noinspection PyUnusedLocal
        xs = None
        try:
            (xs, _) = self.candidate_set.retrieve_all_instances()
        except NoNewElementException:
            raise NoMoreCandidatesException()
        if len(xs) == 0:
            raise NoMoreCandidatesException()

        logging.info(f"{pbs_cand_updater_logging_prefix} retrieved all instances from pool => now add information")
        candidate_information = []
        for x in xs:
            prediction, additional_information = self.ro_pl.predict(x)
            candidate_information.append(self.cand_info_mapping(x, prediction, additional_information))

        logging.info(f"{pbs_cand_updater_logging_prefix} added information to all instances => now load new information into pool/candidate set")
        self.candidate_set.update_instances(xs, candidate_information)

        logging.info("updated whole candidate pool")
