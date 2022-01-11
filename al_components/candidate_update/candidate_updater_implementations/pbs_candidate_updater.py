from additional_component_interfaces import PassiveLearner
from al_components.candidate_update import CandidateUpdater
from workflow_management import CandidateSet
from dataclasses import dataclass


@dataclass()
class PbS_CandidateUpdater(CandidateUpdater):
    candidate_set: CandidateSet
    pl: PassiveLearner

    def update_candidate_set(self):
        # TODO: implement (retrieve all instances, add predictions to them, update them)
        raise NotImplementedError
