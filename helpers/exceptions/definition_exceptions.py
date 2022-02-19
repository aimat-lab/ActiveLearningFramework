import logging
import string


class IncorrectParameters(Exception):
    def __init__(self, message: string):
        logging.error(f"Incorrect parameters are provided: {message}")
        pass


class IncorrectScenarioImplementation(Exception):
    def __init__(self, cause: string):
        logging.error(f"Exception due to incorrect implementation for the selected scenario. Concrete cause: {cause}")
        pass


class InvalidTyping(Exception):
    def __init__(self, message: string):
        logging.error(f"Incorrect typing for parameter: {message}")
