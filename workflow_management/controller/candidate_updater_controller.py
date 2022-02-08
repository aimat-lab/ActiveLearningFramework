import logging
from multiprocessing.managers import ValueProxy
from typing import Callable

from additional_component_interfaces import ReadOnlyPassiveLearner
from al_components.candidate_update import CandidateUpdater, init_candidate_updater
from helpers import Scenarios, SystemStates, X, Y, AddInfo_Y, CandInfo
from helpers.exceptions import NoMoreCandidatesException
from workflow_management.database_interfaces import CandidateSet

candidate_updater_controller_logging_prefix = "AL_cand_updater_controller: "


class CandidateUpdaterController:
    """
    Controls the workflow for AL components working with the (stored) SL model

    => candidate update (needs SL model for prediction) and performance evaluation

    - does not update the SL model (only reads the model)
    """

    def __init__(self, ro_pl: ReadOnlyPassiveLearner, candidate_set: CandidateSet, cand_info_mapping: Callable[[X, Y, AddInfo_Y], CandInfo], scenario: Scenarios, **kwargs):
        """
        Set the pl arguments, initializes the candidate updater

        :param ro_pl: read only view on sl model (implemented interface, including performance evaluation)
        :param candidate_set: dataset containing the candidates (gets updated by candidate_updater)
        :param cand_info_mapping: can generate the information stored for every candidate out of the information provided from the prediction
        :param scenario: determines the candidate updater implementation
        :param kwargs: depends on the scenario/needed arguments for candidate updater (see documentation for init_candidate_updater)
        """

        logging.info(f"{candidate_updater_controller_logging_prefix} Init candidate updater controller => set ro_pl, candidate set, performance evaluator, init candidate_updater")

        self.scenario = scenario
        self.ro_pl = ro_pl
        self.candidate_set = candidate_set

        candidate_source = kwargs.get("candidate_source")  # not needed in PbS scenario (candidate_set = source)
        self.candidate_updater: CandidateUpdater = init_candidate_updater(scenario, cand_info_mapping, ro_pl=ro_pl, candidate_set=candidate_set, candidate_source=candidate_source)

    def init_candidates(self):
        """
        Initial candidate update (e.g. add additional information to candidates after initial training of PL)
        """

        logging.info(f"{candidate_updater_controller_logging_prefix} Initial predictions for candidate set")

        self.ro_pl.load_model()
        self.candidate_updater.update_candidate_set()
        self.ro_pl.close_model()

    def training_job(self, system_state: ValueProxy):
        """
        The actual training job for candidate update and performance evaluation => should run in separate process
        Also including the soft training end job

        Job sequence:
            1. update candidates (if this provides new information)
            2. evaluate performance of pl
                1. if performance satisfies evaluation criterion:
                    1. set system state to terminate training, return
                2. else:
                    1. keep training

        :param system_state: The current system state (shared over all controllers, values align with enum SystemStates)
        :return: if the process should end => indicated by system_state
        """

        if system_state.value >= int(SystemStates.FINISH_TRAINING__INFO):
            logging.warning(f"{candidate_updater_controller_logging_prefix} Training process was terminated => end training job (system_state: {SystemStates(system_state.value).name})")
            return

        self.ro_pl.load_model()
        logging.info(f"{candidate_updater_controller_logging_prefix} Update candidate set")

        try:
            self.candidate_updater.update_candidate_set()
        except NoMoreCandidatesException:
            system_state.set(int(SystemStates.FINISH_TRAINING__INFO))

            logging.warning(f"{candidate_updater_controller_logging_prefix} Initiate finishing of training process, no more candidates => soft end (set system_state: {SystemStates(system_state.value).name})")
            return

        if self.ro_pl.pl_satisfies_evaluation():
            logging.warning(f"{candidate_updater_controller_logging_prefix} PL is trained well enough => terminate training process")
            system_state.set(int(SystemStates.TERMINATE_TRAINING))
            return

        self.training_job(system_state)
        return
