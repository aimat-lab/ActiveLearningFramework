import logging

from additional_component_interfaces import PassiveLearner
from al_components.candidate_update import CandidateUpdater
from helpers.exceptions import IncorrectParameters, NoNewElementException, NoMoreCandidatesException
from workflow_management.database_interfaces import CandidateSet


class Stream:
    """
    Datasource for the candidate updater in a SbS scenario => fetch candidates (unlabelled instance) one at a time
    """

    def get_next_element(self):
        """
        Get one (next) element from the stream/natural distribution/input space

        :raise NoNewElementException if no more element are available from stream
        :return: the properties describing one unlabelled instance (from input space)
        """
        raise NotImplementedError


# noinspection PyPep8Naming
class SbS_CandidateUpdater(CandidateUpdater):
    # TODO: logging, documentation

    def __init__(self, candidate_set: CandidateSet, source_stream: Stream, pl: PassiveLearner):
        if (candidate_set is None) or (not isinstance(candidate_set, CandidateSet)) or (pl is None) or (not isinstance(source_stream, Stream)) or (source_stream is None) or (not isinstance(pl, PassiveLearner)):
            raise IncorrectParameters("SbS_CandidateUpdater needs to be initialized with a candidate_set (of type CandidateSet), a source_stream (of type Stream), and pl (of type PassiveLearner)")
        else:
            self.candidate_set = candidate_set
            self.source = source_stream
            self.pl = pl

    def update_candidate_set(self):
        # TODO: should instances already in candidate set be updated (new predictions) => currently no
        x = None
        try:
            x = self.source.get_next_element()
        except NoNewElementException:
            raise NoMoreCandidatesException()

        prediction, uncertainty = self.pl.predict(x)
        self.candidate_set.add_instance(x, prediction, uncertainty)
        logging.info("added new instance to candidates set => fetched from stream, predicted")
