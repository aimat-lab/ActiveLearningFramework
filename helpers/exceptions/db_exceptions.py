import logging
import string


class NoNewElementException(Exception):
    """
    Thrown, when no new entry can be found in the queried database
    """

    def __init__(self, database_name: string):
        logging.error(f"No new element could be found in database `{database_name}`")
        pass


class NoSuchElementException(Exception):
    """
    Thrown, if a specified element can't be identified in the queried database
    """

    def __init__(self, database_name: string, x):
        logging.error(f"No element with the following properties could be found in database `{database_name}`: {x}")
        pass
