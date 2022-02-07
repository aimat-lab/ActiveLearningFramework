from helpers import X


class QuerySelector:
    """
    Responsible for selecting the best candidate out of the candidate set for querying
    """

    def select_query_instance(self) -> (X, float, bool):
        """
        Evaluate the candidates (direct access to the candidate set), select the next instance for potential querying, return instance and information to query/just discard

        :return (x, query_instance): tuple with input of an instance, the informativeness value, and boolean telling whether to query the instance or not
        :raises NoNewElementException: if the candidate set is empty
        """
        raise NotImplementedError
