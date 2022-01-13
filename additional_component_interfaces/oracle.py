class Oracle:
    """
    Interface for the oracle (provider of ground truth)
    """

    def query(self, x):
        """
        Query an unlabelled instance to get the correct output (according to ground truth)

        :param x: unlabelled instance (input values)
        :return: label y
        """
        raise NotImplementedError
