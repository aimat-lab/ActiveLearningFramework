from additional_component_interfaces import PassiveLearner
from al_components.perfomance_evaluation import PerformanceEvaluator


class DefaultPerformanceEvaluator(PerformanceEvaluator):
    def __init__(self, pl: PassiveLearner, **kwargs):
        self.pl = pl

    def pl_satisfies_evaluation(self) -> bool:
        return False
