from typing import Tuple

from helpers import X, CandInfo


class CandidateSet:
    """
    database interface for candidates

    - will be evaluated by AL to measure their informativeness (and possibly queried)
    - database columns:

        - input (x, usually a list of properties, items are uniquely identifiable by input)
        - optional additional information (e.g. prediction, uncertainty)

    **Communication** between PL (provides information for informativeness analyser) and AL (selects query instance from candidates)
    """

    def add_instance(self, x: X, additional_info: CandInfo) -> None:
        """
        adds new instance (new last entry) into the candidate database (can be selected for query)

        :param x: input values
        :param additional_info: optional information about the candidate => most often prediction and uncertainty of current PL model
        """
        raise NotImplementedError

    def get_first_instance(self) -> Tuple[X, CandInfo]:
        """
        retrieve first instance from candidates

        :return: first instance (input values x, Optional[additional information])
        :raises NoNewElementException: if no instance is in database
        """
        raise NotImplementedError

    def get_instance(self, x: X) -> Tuple[X, CandInfo]:
        """
        retrieve instance identified by x

        :param x: input values identifying the instance
        :return: the input x, and any stored additional information
        :raises NoSuchElementException: if instance identified by x doesn't exist
        """
        raise NotImplementedError

    def remove_instance(self, x: X) -> None:
        """
        remove a candidate based on the input values (if instance doesn't exist, counts as removed as well)

        :param x: input values
        """
        raise NotImplementedError

    def is_empty(self) -> bool:
        """
        checks if any elements are in the candidate set

        :return: true if empty, false if not
        """
        raise NotImplementedError
