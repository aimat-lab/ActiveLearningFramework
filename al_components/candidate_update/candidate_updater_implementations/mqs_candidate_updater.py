from al_components.candidate_update import CandidateUpdater
from helpers import X


class Generator:

    def generate_instance(self) -> X:
        raise NotImplementedError


# TODO: implement the mqs candidate update (including logging)
# noinspection PyPep8Naming
class MQS_CandidateUpdater(CandidateUpdater):

    def update_candidate_set(self):
        raise NotImplementedError
