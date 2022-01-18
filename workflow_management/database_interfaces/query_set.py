from numpy import ndarray


class QuerySet:
    """
    database interface for to be queried instances

    - identified by AL to be informative
    - to be labelled by oracle (=> then added to training set of PL)
    - database columns: input (x, list of properties)

    **Communication** between AL (adding most informative instance for querying) and oracle (retrieving unlabelled instances and labelling them)
    """

    def add_instance(self, x: ndarray) -> None:
        """
        add to be queried instance to set (if instance is already in database: will keep original position)

        :param x: input parameters of the unlabelled instance
        """
        raise NotImplementedError

    # TODO: pop instead of get/remove?
    def get_instance(self) -> ndarray:
        """
        get the first instance in the database

        :raises NoNewElementException: if database is empty
        :return input parameters of first unlabelled instance: x
        """
        raise NotImplementedError

    def remove_instance(self, x: ndarray) -> None:
        """
        ensures the provided x is not in database

        :param x: input of removed instance
        """
        raise NotImplementedError
