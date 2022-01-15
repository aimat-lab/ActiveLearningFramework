import logging
import string


class EndTrainingException(Exception):
    def __init__(self, cause: string):
        logging.info(f"Active training process has been stopped/finished, due to {cause}")
        pass


class NoMoreCandidatesException(Exception):
    def __init__(self):
        logging.info("No more candidates available in candidate source => finish training process")
        pass
