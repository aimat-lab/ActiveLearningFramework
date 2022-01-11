import logging
import string


class IncorrectParameters(Exception):

    def __init__(self, message: string):
        logging.error(f"Incorrect parameters are provided: {message}")
        pass
