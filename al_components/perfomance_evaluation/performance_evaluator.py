from additional_component_interfaces import PassiveLearner


class PerformanceEvaluator:
    """
    Evaluate the performance of the pl => of the SL model/whole model

    - Evaluation is **not part of the PL** training **, but** part of the **AL training** (to check if more training data is necessary)
    """
    pl: PassiveLearner

    def __init__(self, pl: PassiveLearner, **kwargs):
        """
        :param pl: the evaluated passive learner
        :param kwargs: additional properties can be provided, e.g., a set of test data for the evaluation
        """
        raise NotImplementedError

    def pl_satisfies_evaluation(self) -> bool:
        """
        Evaluates the performance of the current SL model (pl) => decides if pl is trained enough (satisfies performance acceptance criterion)
            - can be based on history (performance of pl doesn't get significantly better/gets worse)
            - can be based on threshold => predictions are accurate enough

        :return: whether pl is trained well enough
        """
        raise NotImplementedError
