from dataclasses import dataclass
from typing import Callable

from al_components.candidate_update import CandidateUpdater
from helpers import X, CandInfo, AddInfo_Y, Y


class Generator:

    def generate_instance(self) -> X:
        raise NotImplementedError


# TODO: implement the mqs candidate update (including logging)
# noinspection PyPep8Naming
@dataclass()
class MQS_CandidateUpdater(CandidateUpdater):
    info_creator: Callable[[X, Y, AddInfo_Y], CandInfo]

    def update_candidate_set(self):
        raise NotImplementedError
