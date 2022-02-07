import logging
import time
from multiprocessing.managers import ValueProxy
from typing import List

from additional_component_interfaces import PassiveLearner
from helpers import Scenarios, SystemStates, X, Y
from helpers.exceptions import NoNewElementException
from workflow_management.database_interfaces import TrainingSet

pl_controller_logging_prefix = "PL_controller: "


class PassiveLearnerController:
    """
    Controls the workflow for the actual SL model
    """

    def __init__(self, pl: PassiveLearner, training_set: TrainingSet, scenario: Scenarios):
        """
        Set the pl arguments, initializes the candidate updater

        :param pl: sl model (implemented interface)
        :param training_set: dataset based on which the pl is trained
        :param scenario: determines the candidate updater implementation
        """

        logging.info(f"{pl_controller_logging_prefix} Init passive learner controller => set pl, training set")

        self.scenario = scenario
        self.pl = pl
        self.training_set = training_set

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

    def training_job(self, system_state: ValueProxy):
        """
        The actual training job for the PL components => should run in separate process
        Also including the soft training end job

        Job sequence:
            1. retrieve new training data
                1. if new data available:
                    1. retrain pl
                    2. remove training instance from set
                    3. restart job
                1. else:
                    1. sleep (or return if system state is FINISH_TRAINING__PL)
                    2. restart job

        :param system_state: The current system state (shared over all controllers, values align with enum SystemStates)
        :return: if the process should end => indicated by system_state
        """

        if system_state.value >= int(SystemStates.TERMINATE_TRAINING):
            logging.warning(f"{pl_controller_logging_prefix} Training process was terminated => end training job (system_state: {SystemStates(system_state.value).name})")
            return

        # noinspection PyUnusedLocal
        x_train, y_train = None, None
        try:
            (x_train, y_train) = self.training_set.retrieve_labelled_instance()
        except NoNewElementException:
            if system_state.value == int(SystemStates.FINISH_TRAINING__PL):
                logging.info(f"{pl_controller_logging_prefix} Training database empty, slow end of training finished")
                return

            else:
                logging.info(f"{pl_controller_logging_prefix} Wait for new training data")
                time.sleep(5)

                self.training_job(system_state)
                return

        logging.info(f"{pl_controller_logging_prefix} Train PL with (x, y): x = `{x_train}`, y = `{y_train}`")
        self.pl.load_model()
        self.pl.train(x_train, y_train)
        self.pl.save_model()

        self.training_set.remove_labelled_instance(x_train)
        logging.info(f"{pl_controller_logging_prefix} Removed (x, y) from the training set => PL already trained with it: x = `{x_train}`, y = `{y_train}`")

        self.training_job(system_state)
        return
