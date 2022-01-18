from typing import Tuple, List

from al_components.candidate_update.candidate_updater_implementations import Pool
from helpers import X, Y, CandInfo
from workflow_management.database_interfaces import TrainingSet, QuerySet


class TrainingSetHouses(TrainingSet):

    def append_labelled_instance(self, x: X, y: Y) -> None:
        pass

    def retrieve_labelled_instance(self) -> Tuple[X, Y]:
        pass

    def retrieve_all_labelled_instances(self) -> Tuple[List[X], List[Y]]:
        pass

    def remove_labelled_instance(self, x: X) -> None:
        pass

    def clear(self) -> None:
        pass


class CandidateSetHouses(Pool):
    def get_first_instance(self) -> Tuple[X, CandInfo]:
        pass

    def get_instance(self, x: X) -> Tuple[X, CandInfo]:
        pass

    def remove_instance(self, x: X) -> None:
        pass

    def add_instance(self, x: X, additional_info: CandInfo = None) -> None:
        pass

    def is_empty(self) -> bool:
        pass

    def initiate_pool(self, x_initial: List[X]) -> None:
        pass

    def update_instances(self, xs: List[X], new_additional_infos: List[CandInfo] = None) -> None:
        pass

    def retrieve_all_instances(self) -> Tuple[List[X], List[CandInfo]]:
        pass


class QuerySetHouses(QuerySet):
    def add_instance(self, x: X) -> None:
        pass

    def get_instance(self) -> X:
        pass

    def remove_instance(self, x: X) -> None:
        pass
