from typing import Tuple, List

from nparray import ndarray

from helpers import X, Y


class StoredLabelledSetDB:
    """
    Database with all training instances labelled by oracle

    - instances are correctly labelled by the oracle (should only be the most informative instances, selected with AL)
    - database columns: input (x, usually a list of properties, items are uniquely identifiable by input), label (y, usually one value)

    Labelled set with very informative instances can be used for other training procedures
    """

    def add_labelled_instance(self, x: X, y: Y) -> None:
        """
        Add new labelled instance to end of database (new last entry)

        If instance already exists: will keep original position (and label => should be the same anyway)

        :param x: input of instance (identifies instance)
        :param y: label for instance (correct label => assigned by oracle)
        """
        raise NotImplementedError

    def retrieve_all_labelled_instances(self) -> Tuple[List[X] or ndarray, List[Y] or ndarray]:
        """
        Get all instances from database

        :return: tuple of numpy arrays [x] (array of input), [y] (array of outputs)
        :raises NoNewElementException: if no instance is in database
        """
        raise NotImplementedError
