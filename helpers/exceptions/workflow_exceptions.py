import logging
import string

from helpers import X


class EndTrainingException(Exception):
    def __init__(self, cause: string):
        logging.error(f"Active training process has been stopped/finished, due to {cause}")
        pass


class NoMoreCandidatesException(Exception):
    def __init__(self):
        logging.error("No more candidates available in candidate source => finish training process")
        pass


class CantResolveQueryException(Exception):
    def __init__(self, x: X):
        logging.error(f"Oracle can't resolve query for the instance x: x={x}")
        pass


class ALSystemError(Exception):
    def __init__(self):
        logging.error("Fatal error => system failed")
        pass
