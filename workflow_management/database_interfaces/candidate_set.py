class CandidateSet:
    """
    database interface for candidates # TODO candidate set needs to be database?

    - will be evaluated by AL to measure their informativeness (and possibly queried)
    - database columns: input (x, usually a list of properties, items are uniquely identifiable by input), prediction, uncertainty

    **Communication** between PL (provides information for informativeness analyser) and AL (selects query instance from candidates)
    """

    def add_instance(self, x, y_prediction, uncertainty):
        """
        adds new instance (new last entry) into the candidate database (can be selected for query)

        :param x: input values
        :param y_prediction: predicted output value
        :param uncertainty: of current model about prediction
        """
        raise NotImplementedError

    def get_first_instance(self):
        """
        retrieve first instance from candidates

        :return: first instance (input values x, prediction y, uncertainty)
        :raises NoNewElementException: if no instance is in database
        """
        raise NotImplementedError

    def get_instance(self, x):
        """
        retrieve instance identified by x

        :param x: input values identifying the instance
        :return: the input x, the prediction y, the uncertainty about the prediction
        :raises NoSuchElementException: if instance identified by x doesn't exist
        """
        raise NotImplementedError

    def remove_instance(self, x):
        """
        remove a candidate based on the input values (if instance doesn't exist, counts as removed as well)

        :param x: input values
        """
        raise NotImplementedError
