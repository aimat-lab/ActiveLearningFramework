import logging
import string


class LoadingModelException(Exception):
    def __init__(self, component_context: string):
        logging.error(f"Error while loading the SL model from the component {component_context}")
        pass


class StoringModelException(Exception):
    def __init__(self, component_context: string):
        logging.error(f"Error while storing the SL model from the component {component_context}")
        pass


class ClosingModelException(Exception):
    def __init__(self, component_context: string):
        logging.error(f"Error while closing the SL model/connection from the component {component_context}")
        pass
