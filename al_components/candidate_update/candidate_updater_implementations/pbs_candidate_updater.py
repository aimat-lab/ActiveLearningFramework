import logging
from typing import Tuple, Callable, Iterable, Sequence

from tqdm import tqdm

from additional_component_interfaces import ReadOnlyPassiveLearner
from al_components.candidate_update import CandidateUpdater
from helpers import CandInfo, X, Y, AddInfo_Y
from helpers.exceptions import IncorrectParameters, NoMoreCandidatesException, NoNewElementException
from workflow_management.database_interfaces import CandidateSet


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

    def initiate_pool(self, x_initial: Sequence[X]) -> None:
        """
        Pool needs to be initialized with a list of input instances

        :param x_initial: array of inputs (array of array)
        """
        raise NotImplementedError

    def update_instances(self, xs: Iterable[X], new_additional_infos: Sequence[CandInfo] = None) -> None:
        """
        alter the prediction and uncertainty for the provided candidates (identified by provided input)

        - if an instance x doesn't exist: it will be added with the according information

        :param xs: array of input values
        :param new_additional_infos: array containing new additional information (optional => if candidate set doesn't include additional information, can be None)
        """
        raise NotImplementedError

    def retrieve_all_instances(self) -> Tuple[Sequence[X], Sequence[CandInfo]]:
        """
        retrieves all candidates from database (database is left unchanged)

        :return: tuple of lists [x] (array of inputs), [addition_info] (array of additional information)
        :raises NoNewElementException: if no instance is in database
        """
        raise NotImplementedError


log = logging.getLogger("PbS candidate updater")


# noinspection PyPep8Naming
class PbS_CandidateUpdater(CandidateUpdater):
    """
    The candidate updater within a PbS scenario => whole pool (= candidate set) will be updated with new predictions/information
    """

    # noinspection PyUnusedLocal
    def __init__(self, cand_info_mapping: Callable[[X, Y, AddInfo_Y], CandInfo], candidate_set: Pool, ro_pl: ReadOnlyPassiveLearner, **kwargs):
        log.debug(f"""Initialize PbS candidate updater => should have the following parameters:\
                        candidate_set of type Pool ({candidate_set}, has correct type? {isinstance(candidate_set, Pool)})\
                        ro_pl of type ReadOnlyPassiveLearner ({ro_pl}, has correct type? {isinstance(ro_pl, ReadOnlyPassiveLearner)})\
                        cand_info_mapping as a function, mapping the information provided through predictions to the candidate information ({cand_info_mapping}, for correct type: debug)""")

        if (cand_info_mapping is None) or (candidate_set is None) or (not isinstance(candidate_set, Pool)) or (ro_pl is None) or (not isinstance(ro_pl, ReadOnlyPassiveLearner)):
            raise IncorrectParameters("PbS_CandidateUpdater needs to be initialized with cand_info_mapping (function), a candidate_set (of type Pool) and ro_pl (of type ReadOnlyPassiveLearner)")
        else:
            self.candidate_set = candidate_set
            self.ro_pl = ro_pl
            self.cand_info_mapping = cand_info_mapping
            log.info("Successfully initiated the PbS candidate updater")

    def update_candidate_set(self) -> None:
        """
        Update of the whole pool/candidate set

        Add new predictions/information to every instance within pool

        :raise NoMoreCandidatesException if pool is empty
        """
        log.info("Start candidate update")

        # noinspection PyUnusedLocal
        xs = None
        try:
            (xs, _) = self.candidate_set.retrieve_all_instances()
        except NoNewElementException:
            raise NoMoreCandidatesException()
        if len(xs) == 0:
            raise NoMoreCandidatesException()

        log.info("Retrieved all instances from pool => now add information")
        candidate_information = []
        with tqdm(total=len(xs), position=0, desc="Update of candidate pool", ascii=True) as progress:
            predictions, additional_info = self.ro_pl.predict_set(xs)
            for i in range(len(xs)):
                candidate_information.append(self.cand_info_mapping(xs[i], predictions[i], additional_info[i]))
                progress.update(1)

        log.info("Added information to all instances => now load new information into pool/candidate set")

        self.candidate_set.update_instances(xs, candidate_information)

        log.info("Updated whole candidate pool")
