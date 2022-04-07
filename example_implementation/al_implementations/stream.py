from numpy import ndarray

from al_specific_components.candidate_update.candidate_updater_implementations import Stream
from helpers import X
from helpers.exceptions import NoNewElementException


class HousingStream(Stream):
    def __init__(self, xs):
        self._xs: ndarray = xs

    def get_element(self) -> X:
        if len(self._xs) == 0:
            raise NoNewElementException
        x = self._xs[0]
        self._xs = self._xs[1:]
        return x
