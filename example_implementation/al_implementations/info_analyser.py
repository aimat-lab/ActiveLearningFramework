import numpy as np

from al_specific_components.query_selection import InformativenessAnalyser
from helpers import X
from workflow_management.database_interfaces import CandidateSet


class EverythingIsInformativeAnalyser(InformativenessAnalyser):
    def get_informativeness(self, x: X) -> float:
        return 1


class UncertaintyInfoAnalyser(InformativenessAnalyser):

    def __init__(self, candidate_set: CandidateSet):
        self.candidate_set = candidate_set
        self.uncertainty_history = []

    def get_informativeness(self, x):
        _, variance = self.candidate_set.get_instance(x)
        self.uncertainty_history.append(variance[0])

        # normalization: if certainty equals the mean of the last uncertainties: informativeness = 0.5
        normalized_uncertainty = variance[0] / (2 * np.mean(np.array(self.uncertainty_history)))

        if len(self.uncertainty_history) >= 16:
            self.uncertainty_history = []
        return normalized_uncertainty
