import logging
import string


class NoNewElementException(Exception):
    def __init__(self, database_name: string):
        logging.error(f"No new element could be found in database `{database_name}`")
        pass


class NoSuchElement(Exception):
    def __init__(self, database_name: string, x):
        logging.error(f"No element with the following properties could be found in database `{database_name}`: {x}")
        pass
