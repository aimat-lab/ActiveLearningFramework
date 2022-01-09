class CandidateSet:
    """
    database interface for candidates
        => will be evaluated by AL for their informativeness (and possibly queried)
        => database columns: input (x, usually a list of properties, items are uniquely identifiable by input), prediction, certainty
    """

    # TODO candidate set needs to be database?

    def add_instance(self, x, y_prediction, certainty):
        """
        adds new instance (new last entry) into the candidate database (can be selected for query)

        :param x: input values
        :param y_prediction: predicted output value
        :param certainty: of current model about prediction
        """
        raise NotImplementedError

    def retrieve_all_instances(self):
        """
        retrieves all candidates from database (database is left unchanged)

        :return tuple of numpy arrays: [x], [prediction], [certainty]
        """
        raise NotImplementedError

    def remove_instance(self, x):
        """
        remove an candidate based on the input values

        :param x: input values
        """
        raise NotImplementedError

    def update_instance(self, x, new_y_prediction, new_certainty):
        """
        alter the prediction certainty for a candidate (identified by provided input)

        :param x: input values
        :param new_y_prediction: the new prediction
        :param new_certainty: the new certainty about predicition
        """
        raise NotImplementedError

    def get_instance(self):
        """
        retrieve first instance from candidates

        :return first instance
        """
        raise NotImplementedError
