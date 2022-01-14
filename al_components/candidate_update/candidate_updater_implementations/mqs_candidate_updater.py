from al_components.candidate_update import CandidateUpdater


class Generator:

    def generate_instance(self):
        raise NotImplementedError


# TODO: implement the mqs candidate update (including logging)
class MQS_CandidateUpdater(CandidateUpdater):

    def update_candidate_set(self):
        raise NotImplementedError
