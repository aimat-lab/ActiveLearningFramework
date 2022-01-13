from dataclasses import dataclass

from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from workflow_management.database_interfaces import CandidateSet


@dataclass()
class UncertaintyInfoAnalyser(InformativenessAnalyser):
    candidate_set: CandidateSet

    def get_informativeness(self, x):
        _, _, uncertainty = self.candidate_set.get_instance(x)
        return uncertainty
