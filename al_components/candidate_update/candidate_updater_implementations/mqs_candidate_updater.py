from al_components.candidate_update import CandidateUpdater


class Generator:

    def generate_instance(self):
        raise NotImplementedError


# TODO: implement the mqs candidate update
class MQS_CandidateUpdater(CandidateUpdater):

    def update_candidate_set(self):
        raise NotImplementedError
