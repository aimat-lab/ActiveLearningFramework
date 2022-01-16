import logging
import time
from multiprocessing.managers import ValueProxy

from additional_component_interfaces import PassiveLearner
from al_components.candidate_update import CandidateUpdater, init_candidate_updater
from helpers import Scenarios, SystemStates
from helpers.exceptions import NoNewElementException, NoMoreCandidatesException
from workflow_management.database_interfaces import TrainingSet, CandidateSet

pl_controller_logging_prefix = "PL_controller: "


class PassiveLearnerController:
    """
    Controls the workflow for components working with the PL (=> the actual SL model and the candidate updater (needs PL for prediction calculation))
    """

    def __init__(self, pl: PassiveLearner, training_set: TrainingSet, candidate_set: CandidateSet, scenario: Scenarios, **kwargs):
        """
        Set the pl arguments, initializes the candidate updater

        :param pl: sl model (implemented interface)
        :param training_set: dataset based on which the pl is trained
        :param candidate_set: dataset containing the candidates (gets updated by candidate_updater)
        :param scenario: determines the candidate updater implementation
        :param kwargs: depends on the scenario/needed arguments for candidate updater (see documentation for init_candidate_updater)
        """

        logging.info(f"{pl_controller_logging_prefix} Init passive learner controller => set pl, training set, candidate set, init candidate_updater")

        self.pl = pl
        self.training_set = training_set
        self.candidate_set = candidate_set

        candidate_source = kwargs.get("candidate_source")  # not needed in PbS scenario (candidate_set = source)
        self.candidate_updater: CandidateUpdater = init_candidate_updater(scenario, pl=pl, candidate_set=candidate_set, candidate_source=candidate_source)

        self.pl_and_candidates_align = False  # keeps track of whether it makes sense to update candidates => if nothing changed in pl, candidates don't need to be updated

    def init_pl(self, x_train, y_train, **kwargs):
        """
        Initialize the sl model

        - initial training (including setting the scaling for the input)
        - save the model => so it can be loaded in new process environment

        :param x_train: Initial training data input
        :param y_train: Initial training data labels
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
        Add initial predictions for candidates (after initial training of pl)
        """

        logging.info(f"{pl_controller_logging_prefix} Initial predictions for candidate set")

        self.pl.load_model()
        self.candidate_updater.update_candidate_set()

        logging.info(f"{pl_controller_logging_prefix} SL model and candidate_set align (candidates freshly updated)")
        self.pl_and_candidates_align = True
        self.pl.save_model()

    def training_job(self, system_state: ValueProxy):
        """
        The actual training job for the PL components => should be in separate process

        Job sequence:
            1. update candidates (if this provides new information)
            2. retrieve new training data
                1. if new data available:
                    1. retrain pl
                    2. remove training instance from set
                    3. restart job
                1. else:
                    1. sleep
                    2. restart job

        :param system_state: The current system state (shared over all controllers)
        """

        if system_state.value > int(SystemStates.TRAINING):
            logging.warning(f"{pl_controller_logging_prefix} Training process was terminated => end training job (system_state: {SystemStates(system_state.value).name})")
            return

        # check if candidate update is necessary:
        # # for PbS: pl didn't change => no new information provided through candidate update
        # # for SbS/MQS: only if candidate set is not empty assumption form PbS can be made  # TODO: maybe remove check for SbS/MQS
        self.pl.load_model()
        if not self.pl_and_candidates_align or self.candidate_set.is_empty():
            logging.info(f"{pl_controller_logging_prefix} Update candidate set")

            try:
                self.candidate_updater.update_candidate_set()
            except NoMoreCandidatesException:
                system_state.set(int(SystemStates.FINISH_TRAINING))

                logging.warning(f"{pl_controller_logging_prefix} Terminate training process, no more candidates => soft end (set system_state: {SystemStates(system_state.value).name})")
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
