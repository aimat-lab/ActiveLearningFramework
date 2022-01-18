from numpy import ndarray


class InformativenessAnalyser:
    """
    Analyser of informativeness

    Can use the following sources to evaluate the informativeness (needs to be initialized with the appropriate parameters):
        1. Input data
            - influence on SL model/components/error => need information about the PL, ...
            - relation to underlying distribution of input => need information about input space/distribution, or needs to obtain it itself (e.g. database storing information about queried input instances)
        2. Predictions of PL
            - needs access to predictions, uncertainties => access to candidate set
        3. History of queries:
            - keep track of queries
    """

    def get_informativeness(self, x: ndarray) -> float:
        """
        Evaluate informativeness of a single input instance => gather necessary information for evaluation

        :param x: the input data
        :return: number measuring the informativeness => must be larger than 0 (ideally normalized to number between 0, 1)
        """
        raise NotImplementedError
