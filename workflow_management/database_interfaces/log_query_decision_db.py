from helpers import X, CandInfo


class LogQueryDecisionDB:
    """
    Database to log the informativeness measure and query decision process

    - database columns:

        - input (x, usually a list of properties, items are uniquely identifiable by input)
        - informativeness value (input evaluated by informativeness analyser, float value)
        - queried? (boolean, telling whether the instance was queried or discarded)
        - optional additional information from candidate set (e.g. prediction, uncertainty)
    """

    def add_instance(self, x: X, info_value: float, queried: bool, additional_info: CandInfo) -> None:
        """
        add log entry about evaluated candidate

        :param x: input values
        :param info_value: informativeness according to evaluator
        :param queried: decision, whether candidate is actually queried or discarded
        :param additional_info: optional information about the candidate => most often prediction and uncertainty of current PL model
        """
        raise NotImplementedError
