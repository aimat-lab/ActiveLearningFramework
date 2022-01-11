from additional_component_interfaces import PassiveLearner
from al_components.candidate_update import CandidateUpdater
from workflow_management import CandidateSet
from dataclasses import dataclass


class Stream:

    def get_next_element(self):
        raise NotImplementedError


@dataclass()
class SbS_CandidateUpdater(CandidateUpdater):
    candidate_set: CandidateSet
    source_stream: Stream
    pl: PassiveLearner

    def update_candidate_set(self):
        # TODO: implement (retrieve all instances, add predictions to them, update them)
        raise NotImplementedError
