class QuerySet:
    """
    database interface for to be queried instances
        => identified by AL to be informative
        => to be labelled by oracle (=> then added to training set of PL)
        => database columns: input (x, list of properties)
    """

    def add_instance(self, x):
        """
        add to be queried instance to set (if instance is already in database: will keep original position)

        :param x: input parameters of the unlabelled instance
        """
        raise NotImplementedError

    def get_instance(self):
        """
        get the first instance in the database

        :raises NoNewElementException: if database is empty
        :return: input parameters of first unlabelled instance
        """
        raise NotImplementedError

    def remove_instance(self, x):
        raise NotImplementedError
