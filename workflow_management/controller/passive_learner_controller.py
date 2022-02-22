import logging
import time
from multiprocessing import synchronize
from multiprocessing.managers import ValueProxy
from typing import Sequence

from basic_sl_component_interfaces import PassiveLearner
from helpers import SystemStates, X, Y
from helpers.exceptions import NoNewElementException, StoringModelException, LoadingModelException
from workflow_management.database_interfaces import TrainingSet

log = logging.getLogger("Passive Learner controller")


class PassiveLearnerController:
    """
    Controls the workflow for the actual SL model
    """

    def __init__(self, pl: PassiveLearner, training_set: TrainingSet):
        """
        Set the pl arguments, initializes the candidate updater

        :param pl: sl model (implemented interface)
        :param training_set: dataset based on which the pl is trained
        """
        log.info("Init passive learner controller")

        log.debug("Set pl and training set")
        self.pl = pl
        self.training_set = training_set

    def save_sl_model(self):
        try:
            log.debug("Store the updated SL model")
            self.pl.save_model()
        except Exception as e:
            log.error("During saving of model, an error occurred", e)
            raise StoringModelException("Passive learner (within pl controller)")

    def init_pl(self, x_train: Sequence[X], y_train: Sequence[Y], **kwargs):
        """
        Initialize the sl model

        - initial training (including setting the scaling for the input)
        - save the model => so it can be loaded in new process environment

        :param x_train: Initial training data input (list of input data, most likely numpy array)
        :param y_train: Initial training data labels (list of assigned output to input data, most likely numpy array)
        :param kwargs: can provide the `batch_size` and `epochs` for initial training
        """

        log.info("Initial training of pl")

        batch_size = kwargs.get("batch_size")
        epochs = kwargs.get("epochs")
        self.pl.initial_training(x_train, y_train, batch_size=batch_size, epochs=epochs)
        self.save_sl_model()

    def training_job(self, system_state: ValueProxy, sl_model_gets_stored: synchronize.Lock):
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
            2. evaluate performance of pl
                1. if performance satisfies evaluation criterion:
                    1. set system state to terminate training, return
                2. else:
                    1. keep training

        :param system_state: The current system state (shared over all controllers, values align with enum SystemStates)
        :param sl_model_gets_stored: Set, if SL model storage is currently in process
        :return: if the process should end => indicated by system_state
        """
        try:
            if system_state.value >= int(SystemStates.TERMINATE_TRAINING):
                log.warning(f"Training process was terminated => end training job (system_state: {SystemStates(system_state.value).name})")
                return

            # noinspection PyUnusedLocal
            x_train, y_train = None, None
            try:
                (x_train, y_train) = self.training_set.retrieve_labelled_training_instance()
            except NoNewElementException:
                if system_state.value == int(SystemStates.FINISH_TRAINING__PL):
                    log.info("Training database empty, slow end of training finished")
                    return

                else:
                    log.info("Wait for new training data")
                    time.sleep(5)

                    self.training_job(system_state, sl_model_gets_stored)
                    return

            log.info(f"Train PL with (x, y): x = `{x_train}`, y = `{y_train}`")

            sl_model_gets_stored.acquire()
            try:
                log.debug("Load the read only SL model")
                self.pl.load_model()
            except Exception as e:
                sl_model_gets_stored.release()
                log.error("During loading of model, an error occurred", e)
                raise LoadingModelException("Passive Learner (withing pl controller)")
            sl_model_gets_stored.release()

            self.pl.train(x_train, y_train)
            sl_model_gets_stored.acquire()
            self.save_sl_model()
            sl_model_gets_stored.release()

            self.training_set.set_instance_not_use_for_training(x_train)
            log.info(f"Set (x, y) to not be part of active training any more => PL already trained with it: x = `{x_train}`, y = `{y_train}`")

            if self.pl.sl_model_satisfies_evaluation():
                log.warning("PL is trained well enough => terminate training process")
                system_state.set(int(SystemStates.TERMINATE_TRAINING))
                return

            self.training_job(system_state, sl_model_gets_stored)
            return
        except Exception as e:
            log.error("An error occurred during the execution of pl training job => terminate system", e)
            system_state.set(int(SystemStates.ERROR))
            return
