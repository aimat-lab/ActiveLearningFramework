import logging
import time
from multiprocessing.managers import ValueProxy
from typing import List, Callable

from additional_component_interfaces import PassiveLearner
from al_components.candidate_update import CandidateUpdater, init_candidate_updater
from al_components.perfomance_evaluation import PerformanceEvaluator
from helpers import Scenarios, SystemStates, X, Y, AddInfo_Y, CandInfo
from helpers.exceptions import NoNewElementException, NoMoreCandidatesException, IncorrectParameters
from workflow_management.database_interfaces import TrainingSet, CandidateSet

pl_controller_logging_prefix = "PL_controller: "


class PassiveLearnerController:
    """
    Controls the workflow for components working with the PL

    => the actual SL model (including performance evaluation) and the candidate updater (needs PL for prediction calculation)
    """

    def __init__(self, pl: PassiveLearner, training_set: TrainingSet, candidate_set: CandidateSet, cand_info_mapping: Callable[[X, Y, AddInfo_Y], CandInfo], scenario: Scenarios, pl_evaluator: PerformanceEvaluator, **kwargs):
        """
        Set the pl arguments, initializes the candidate updater

        :param pl: sl model (implemented interface)
        :param training_set: dataset based on which the pl is trained
        :param candidate_set: dataset containing the candidates (gets updated by candidate_updater)
        :param cand_info_mapping: can generate the information stored for every candidate out of the information provided from the prediction
        :param scenario: determines the candidate updater implementation
        :param pl_evaluator: evaluator of the current SL model performance
        :param kwargs: depends on the scenario/needed arguments for candidate updater (see documentation for init_candidate_updater)
        """

        logging.info(f"{pl_controller_logging_prefix} Init passive learner controller => set pl, training set, candidate set, init candidate_updater")

        self.scenario = scenario
        self.pl = pl
        self.training_set = training_set
        self.candidate_set = candidate_set
        self.pl_evaluator = pl_evaluator
        if not self.pl_evaluator.pl == pl:
            raise IncorrectParameters("The pl provided to the pl_controller and the pl of the pl_evaluator need to be the same!")

        candidate_source = kwargs.get("candidate_source")  # not needed in PbS scenario (candidate_set = source)
        self.candidate_updater: CandidateUpdater = init_candidate_updater(scenario, cand_info_mapping, pl=pl, candidate_set=candidate_set, candidate_source=candidate_source)

        self.pl_and_candidates_align = False  # keeps track of whether it makes sense to update candidates => if nothing changed in pl, candidates don't need to be updated

    def init_pl(self, x_train: List[X], y_train: List[Y], **kwargs):
        """
        Initialize the sl model

        - initial training (including setting the scaling for the input)
        - save the model => so it can be loaded in new process environment

        :param x_train: Initial training data input (list of input data, most likely numpy array)
        :param y_train: Initial training data labels (list of assigned output to input data, most likely numpy array)
        :param kwargs: can provide the `batch_size` and `epochs` for initial training
        """

        logging.info(f"{pl_controller_logging_prefix} Initial training of pl")

        batch_size = kwargs.get("batch_size")
        epochs = kwargs.get("epochs")
        self.pl.initial_training(x_train, y_train, batch_size=batch_size, epochs=epochs)
        self.pl.save_model()

        logging.info(f"{pl_controller_logging_prefix} SL model and candidate_set don't align (SL newly trained)")
        self.pl_and_candidates_align = False

    def init_candidates(self):
        """
        Initial candidate update (e.g. add additional information to candidates after initial training of PL)
        """

        logging.info(f"{pl_controller_logging_prefix} Initial predictions for candidate set")

        # TODO: move pl.load/save model into candidate update??
        self.pl.load_model()
        self.candidate_updater.update_candidate_set()

        logging.info(f"{pl_controller_logging_prefix} SL model and candidate_set align (candidates freshly updated)")
        self.pl_and_candidates_align = True
        self.pl.save_model()

    def training_job(self, system_state: ValueProxy):
        """
        The actual training job for the PL components => should run in separate process

        Job sequence:
            1. evaluate performance of pl
                1. if performance satisfies evaluation criterion:
                    1. set system state to terminate training, return
                2. else:
                    1. keep training
            2. update candidates (if this provides new information)
            3. retrieve new training data
                1. if new data available:
                    1. retrain pl
                    2. remove training instance from set
                    3. restart job
                1. else:
                    1. sleep
                    2. restart job

        :param system_state: The current system state (shared over all controllers, values align with enum SystemStates)
        :return: if the process should end => indicated by system_state
        """

        if system_state.value > int(SystemStates.TRAINING):
            logging.warning(f"{pl_controller_logging_prefix} Training process was terminated => end training job (system_state: {SystemStates(system_state.value).name})")
            return

        self.pl.load_model()

        if self.pl_evaluator.pl_satisfies_evaluation():
            logging.warning(f"{pl_controller_logging_prefix} PL is trained well enough => terminate training process")
            system_state.set(int(SystemStates.TERMINATE_TRAINING))
            return

        # check if candidate update is necessary:
        # # for PbS: pl didn't change => no new information provided through candidate update
        # # for SbS/MQS: only if candidate set is not empty assumption form PbS can be made => because it can slow down process: will just always update if scenario is SbS or MQS
        if (self.scenario != Scenarios.PbS) or (not self.pl_and_candidates_align) or (self.candidate_set.is_empty()):
            logging.info(f"{pl_controller_logging_prefix} Update candidate set")

            try:
                self.candidate_updater.update_candidate_set()
            except NoMoreCandidatesException:
                system_state.set(int(SystemStates.FINISH_TRAINING))

                logging.warning(f"{pl_controller_logging_prefix} Initiate finishing of training process, no more candidates => soft end (set system_state: {SystemStates(system_state.value).name})")
                return

            logging.info(f"{pl_controller_logging_prefix} SL model and candidate_set align (candidates freshly updated)")
            self.pl_and_candidates_align = True
        else:
            logging.info(f"{pl_controller_logging_prefix} Candidate set doesn't need to be updated => skipped update step")

        # noinspection PyUnusedLocal
        x_train, y_train = None, None
        try:
            (x_train, y_train) = self.training_set.retrieve_labelled_instance()
        except NoNewElementException:
            logging.info(f"{pl_controller_logging_prefix} Wait for new training data")
            time.sleep(5)

            self.training_job(system_state)
            return

        logging.info(f"{pl_controller_logging_prefix} Train PL with (x, y): x = `{x_train}`, y = `{y_train}`")
        self.pl.train(x_train, y_train)
        self.pl.save_model()

        self.pl_and_candidates_align = False
        logging.info(f"{pl_controller_logging_prefix} SL model and candidate_set don't align (SL newly trained)")

        self.training_set.remove_labelled_instance(x_train)
        logging.info(f"{pl_controller_logging_prefix} Removed (x, y) from the training set => PL already trained with it: x = `{x_train}`, y = `{y_train}`")

        self.training_job(system_state)
        return

    def finish_training(self):
        """
        Soft end for training => get remaining instances from training set and train pl with it

        :return: once the final training is finished
        """

        logging.info(f"{pl_controller_logging_prefix} Soft end of training process => train with remaining instances in training set")
        self.pl.load_model()

        while True:
            # noinspection PyUnusedLocal
            x_train, y_train = None, None
            try:
                (x_train, y_train) = self.training_set.retrieve_labelled_instance()
            except NoNewElementException:
                logging.info(f"{pl_controller_logging_prefix} No more training data available")
                break

            logging.info(f"{pl_controller_logging_prefix} Train PL with (x, y): x = `{x_train}`, y = `{y_train}`")
            self.pl.train(x_train, y_train)
            self.training_set.remove_labelled_instance(x_train)
            logging.info(f"{pl_controller_logging_prefix} Removed (x, y) from the training set => PL already trained with it: x = `{x_train}`, y = `{y_train}`")

        self.pl.save_model()
        logging.info(f"{pl_controller_logging_prefix} Finished soft end of training of SL model => SL model ready for predictions")
