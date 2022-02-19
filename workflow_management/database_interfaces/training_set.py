from typing import Tuple, Sequence

from helpers import X, Y


class TrainingSet:
    """
    Labelled database containing instances, the passive learner will be trained with

    - instances are correctly labelled by the oracle (should only be the most informative instances, selected with AL)
    - database columns: input (x, usually a list of properties, items are uniquely identifiable by input), label (y, usually one value), use for training (bool)

    **Communication** between oracle (providing labelled instances) and PL (retrieving instances for training)
    """

    def append_labelled_instance(self, x: X, y: Y) -> None:
        """
        Add new labelled instance to end of database (new last entry)

        If instance already exists: will keep original position (and label => should be the same anyway)

        :param x: input of instance (identifies instance)
        :param y: label for instance (correct label => assigned by oracle)
        """
        raise NotImplementedError

    def retrieve_labelled_training_instance(self) -> Tuple[X, Y]:
        """
        Get first instance from database. that should be used for training of PL

        :return: tuple of input and label: x, y
        :raises NoNewElementException: if no instance is in database
        """
        raise NotImplementedError

    def retrieve_all_training_instances(self) -> Tuple[Sequence[X], Sequence[Y]]:
        """
        Get all instances from database that are still used for training of PL (use for training = true)

        :return: tuple of numpy arrays [x] (array of input), [y] (array of outputs)
        :raises NoNewElementException: if no instance is in database
        """
        raise NotImplementedError

    def retrieve_all_labelled_instances(self) -> Tuple[Sequence[X], Sequence[Y]]:
        """
        Get all instances from database

        :return: tuple of numpy arrays [x] (array of input), [y] (array of outputs)
        :raises NoNewElementException: if no instance is in database
        """
        raise NotImplementedError

    def set_instance_not_use_for_training(self, x: X) -> None:
        """
        Ensures the instance identified through input x is not in the database (either remove if existing)

        :param x: input of the instance
        """
        raise NotImplementedError

    def clear(self) -> None:
        """
        Remove all instances from database
        """
        raise NotImplementedError
