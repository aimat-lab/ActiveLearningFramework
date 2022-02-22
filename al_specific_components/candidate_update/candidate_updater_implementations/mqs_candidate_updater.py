import logging
from dataclasses import dataclass
from typing import Callable

from al_specific_components.candidate_update import CandidateUpdater
from helpers import X, CandInfo, AddInfo_Y, Y


class Generator:

    def generate_instance(self) -> X:
        raise NotImplementedError


log = logging.getLogger("MQS candidate updater")


# TODO: implement the mqs candidate update (including logging)
# noinspection PyPep8Naming
@dataclass()
class MQS_CandidateUpdater(CandidateUpdater):
    cand_info_mapping: Callable[[X, Y, AddInfo_Y], CandInfo]

    def __init__(self, **kwargs):
        log.error("MQS candidate updater not implemented yet")
        raise NotImplementedError

    def update_candidate_set(self):
        log.error("MQS candidate updater not implemented yet")
        raise NotImplementedError
