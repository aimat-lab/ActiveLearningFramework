import logging
from typing import Callable

from additional_component_interfaces import ReadOnlyPassiveLearner
from al_components.candidate_update import CandidateUpdater
from helpers import CandInfo, AddInfo_Y, Y, X
from helpers.exceptions import IncorrectParameters, NoNewElementException, NoMoreCandidatesException
from workflow_management.database_interfaces import CandidateSet

sbs_cand_updater_logging_prefix = "SbS Candidate Updater: "


class Stream:
    """
    Datasource for the candidate updater in a SbS scenario => fetch candidates (unlabelled instance) one at a time
    """

    def get_element(self):
        """
        Get one (next) element from the stream/natural distribution/input space

        :raise NoNewElementException if no more element are available from stream
        :return: the properties describing one unlabelled instance (from input space)
        """
        raise NotImplementedError


# noinspection PyPep8Naming
class SbS_CandidateUpdater(CandidateUpdater):
    """
    Candidate updater within a SbS scenario => will fetch instance from stream, add information/predictions, and insert instance into candidate set
    """

    # noinspection PyUnusedLocal
    def __init__(self, cand_info_mapping: Callable[[X, Y, AddInfo_Y], CandInfo], candidate_set: CandidateSet, candidate_source: Stream, ro_pl: ReadOnlyPassiveLearner, **kwargs):
        if (candidate_set is None) or (not isinstance(candidate_set, CandidateSet)) or (ro_pl is None) or (not isinstance(candidate_source, Stream)) or (candidate_source is None) or (not isinstance(ro_pl, ReadOnlyPassiveLearner)):
            raise IncorrectParameters("SbS_CandidateUpdater needs to be initialized with an cand_info_mapping (of type CandidateInformationCreator), a candidate_set (of type CandidateSet), a candidate_source (of type Stream), and ro_pl (of type ReadOnlyPassiveLearner)")
        else:
            self.candidate_set = candidate_set
            self.source = candidate_source
            self.ro_pl = ro_pl
            self.cand_info_mapping = cand_info_mapping
            logging.info(f"{sbs_cand_updater_logging_prefix} successfully initiated the sbs candidate updater")

    def update_candidate_set(self) -> None:
        """
        Will update the candidate set by adding one newly fetched instance (augmented with additional information based on prediction)

        :raise NoMoreCandidatesException if stream doesn't provide new instances
        """
        # TODO: should instances that are already stored in candidate set be updated with new info/predictions => currently no (only add new)
        # noinspection PyUnusedLocal
        x = None
        try:
            x = self.source.get_element()
        except NoNewElementException:
            raise NoMoreCandidatesException()
        logging.info(f"{sbs_cand_updater_logging_prefix} fetched new instance from stream")

        prediction, additional_information = self.ro_pl.predict(x)
        logging.info(f"{sbs_cand_updater_logging_prefix} added information to the instance")

        self.candidate_set.add_instance(x, self.cand_info_mapping(x, prediction, additional_information))
        logging.info(f"{sbs_cand_updater_logging_prefix} inserted new candidate into set")
