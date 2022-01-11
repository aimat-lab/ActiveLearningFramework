class TrainingSet:
    """
    Labelled database containing instances, the passive learner will be trained with

    - instances are correctly labelled by the oracle (should only be the most informative instances, selected with AL)
    - database columns: input (x, usually a list of properties, items are uniquely identifiable by input), label (y, usually one value)

    **Communication** between oracle (providing labelled instances) and PL (retrieving instances for training)
    """

    def append_labelled_instance(self, x, y):
        """
        Add new labelled instance to end of database (new last entry)

        If instance already exists: will keep original position (and label => should be the same anyway)

        :param x: input of instance (identifies instance)
        :param y: label for instance (correct label => assigned by oracle)
        """
        raise NotImplementedError

    def retrieve_labelled_instance(self):
        """
        Get first instance from database

        :return: tuple of numpy arrays [x], [y]
        :raises NoNewElementException: if no instance is in database
        """
        raise NotImplementedError

    def retrieve_all_labelled_instances(self):
        """
        Get all instances from database

        :return: tuple of input and label: x, y
        :raises NoNewElementException: if no instance is in database
        """
        raise NotImplementedError

    def remove_labelled_instance(self, x):
        """
        Ensures the instance identified through input x is not in the database (either remove if existing)

        :param x: input of the instance
        """
        raise NotImplementedError

    def clear(self):
        """
        Remove all instances from database
        """
        raise NotImplementedError