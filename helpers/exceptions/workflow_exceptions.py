import logging
import string


class EndTrainingException(Exception):
    def __init__(self, cause: string):
        logging.error(f"Active training process has been stopped/finished, due to {cause}")
        pass


class NoMoreCandidatesException(Exception):
    def __init__(self):
        logging.error("No more candidates available in candidate source => finish training process")
        pass


class ALSystemError(Exception):
    def __init__(self):
        logging.error("Fatal error => system failed")
        pass
