import numpy as np

from al_specific_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import X
from workflow_management.database_interfaces import CandidateSet


class EverythingIsInformativeAnalyser(InformativenessAnalyser):

    def get_informativeness(self, x: X) -> float:
        return 1


def _calculate_uncertainty(variance):
    uncertainty = np.mean(np.var(variance[:2])) * 1 + np.mean(np.var(variance[2:])) * 5
    return uncertainty


class UncertaintyInfoAnalyser(InformativenessAnalyser):

    def __init__(self, candidate_set: CandidateSet):
        self.candidate_set = candidate_set
        self.mean = 0.5
        self.amount_instances = 0

    def get_informativeness(self, x):
        _, variance = self.candidate_set.get_instance(x)
        uncertainty = _calculate_uncertainty(np.asarray(variance))
        self.mean = (self.mean * self.amount_instances + uncertainty) / (self.amount_instances + 1)
        if self.amount_instances > 3000:
            self.amount_instances = 50
        else:
            self.amount_instances += 1
        # normalization: if certainty equals the mean of the last uncertainties: informativeness = 0.5
        normalized_uncertainty = uncertainty / (2 * self.mean)
        return normalized_uncertainty
