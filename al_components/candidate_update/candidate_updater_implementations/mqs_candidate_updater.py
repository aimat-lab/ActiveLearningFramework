from dataclasses import dataclass

from al_components.candidate_update import CandidateUpdater, CandidateInformationCreator
from helpers import X


class Generator:

    def generate_instance(self) -> X:
        raise NotImplementedError


# TODO: implement the mqs candidate update (including logging)
# noinspection PyPep8Naming
@dataclass()
class MQS_CandidateUpdater(CandidateUpdater):
    info_creator: CandidateInformationCreator

    def update_candidate_set(self):
        raise NotImplementedError
