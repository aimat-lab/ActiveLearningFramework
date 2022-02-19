import logging
from multiprocessing import synchronize
from multiprocessing.managers import ValueProxy
from typing import Callable

from additional_component_interfaces import ReadOnlyPassiveLearner
from al_components.candidate_update import CandidateUpdater, init_candidate_updater
from helpers import Scenarios, SystemStates, X, Y, AddInfo_Y, CandInfo
from helpers.exceptions import NoMoreCandidatesException, LoadingModelException, ClosingModelException
from workflow_management.database_interfaces import CandidateSet

log = logging.getLogger("AL candidate updater controller")


class CandidateUpdaterController:
    """
    Controls the workflow for AL components working with the (stored) SL model

    => candidate update (needs SL model for prediction) and performance evaluation

    - does not update the SL model (only reads the model)
    """

    def __init__(self, ro_pl: ReadOnlyPassiveLearner, candidate_set: CandidateSet, cand_info_mapping: Callable[[X, Y, AddInfo_Y], CandInfo], scenario: Scenarios, **kwargs):
        """
        Initializes the candidate updater, set the read only Passive Learner

        :param ro_pl: read only view on sl model (implemented interface, including performance evaluation)
        :param candidate_set: dataset containing the candidates (gets updated by candidate_updater)
        :param cand_info_mapping: can generate the information stored for every candidate out of the information provided from the prediction
        :param scenario: determines the candidate updater implementation
        :param kwargs: depends on the scenario/needed arguments for candidate updater (see documentation for init_candidate_updater)
        """
        log.info("Initialize candidate updater controller")

        log.debug("Set scenario, read only passive learner (including performance evaluation) and candidate set")
        self.scenario = scenario
        self.ro_pl = ro_pl
        self.candidate_set = candidate_set

        candidate_source = kwargs.get("candidate_source")  # not needed in PbS scenario (candidate_set = source) => candidate_source = None
        self.candidate_updater: CandidateUpdater = init_candidate_updater(scenario, cand_info_mapping, ro_pl=ro_pl, candidate_set=candidate_set, candidate_source=candidate_source)
        log.debug(f"Initialized the candidate updater, scenario: {scenario} => candidate updater type: {type(self.candidate_updater)}")

    def close_sl_connection(self):
        try:
            log.debug("Close connection to read only SL model")
            self.ro_pl.close_model()
        except Exception as e:
            log.error("During closing of model, an error occurred", e)
            raise ClosingModelException("Read Only Passive learner (within candidate updater)")

    def init_candidates(self):
        """
        Initial candidate update (e.g. add additional information to candidates after initial training of PL)
        """

        log.info("Initial update of candidate set => insert first candidate/add initial additional information after first training of PL")

        try:
            log.debug("Load the read only SL model")
            self.ro_pl.load_model()
        except Exception as e:
            log.error("During loading of model, an error occurred", e)
            raise LoadingModelException("Read Only Passive Learner (withing candidate updater)")
        self.candidate_updater.update_candidate_set()
        self.close_sl_connection()

    def training_job(self, system_state: ValueProxy, sl_model_gets_stored: synchronize.Lock):
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
        :param sl_model_gets_stored: Check if SL model storage is currently in process => then not loading
        :return: if the process should end => indicated by system_state
        """
        try:

            if system_state.value >= int(SystemStates.FINISH_TRAINING__INFO):
                log.warning(f"Training process was terminated => end training job (system_state: {SystemStates(system_state.value).name})")
                return

            sl_model_gets_stored.acquire()
            try:
                log.debug("Load the read only SL model")
                self.ro_pl.load_model()
            except Exception as e:
                sl_model_gets_stored.release()
                log.error("During loading of model, an error occurred", e)
                raise LoadingModelException("Read Only Passive Learner (withing candidate updater)")
            sl_model_gets_stored.release()

            try:
                self.candidate_updater.update_candidate_set()
            except NoMoreCandidatesException:
                system_state.set(int(SystemStates.FINISH_TRAINING__INFO))
                log.warning(f"Initiate finishing of training process, no more candidates => soft end (set system_state: {SystemStates(system_state.value).name})")
                self.close_sl_connection()
                return

            if self.scenario == Scenarios.PbS:  # reload model, because candidate update can take a long time => maybe SL model updated meanwhile
                sl_model_gets_stored.acquire()
                self.close_sl_connection()
                try:
                    log.debug("Load the read only SL model")
                    self.ro_pl.load_model()
                except Exception as e:
                    sl_model_gets_stored.release()
                    log.error("During loading of model, an error occurred", e)
                    raise LoadingModelException("Read Only Passive Learner (withing candidate updater)")
                sl_model_gets_stored.release()

            if self.ro_pl.pl_satisfies_evaluation():
                log.warning("PL is trained well enough => terminate training process")
                system_state.set(int(SystemStates.TERMINATE_TRAINING))
                self.close_sl_connection()
                return

            self.close_sl_connection()
            self.training_job(system_state, sl_model_gets_stored)
            return

        except Exception as e:
            log.error("An error occurred during the execution of candidate updater training job => terminate system", e)
            system_state.set(int(SystemStates.ERROR))
            return
