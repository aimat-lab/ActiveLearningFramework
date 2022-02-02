import logging
import time
from multiprocessing.managers import ValueProxy

from additional_component_interfaces import Oracle
from helpers import SystemStates
from helpers.exceptions import NoNewElementException
from workflow_management.database_interfaces import TrainingSet, QuerySet

oracle_controller_logging_prefix = "oracle_controller: "


class OracleController:
    """
    Controls the oracle workflow (manages query process)
    """

    def __init__(self, o: Oracle, training_set: TrainingSet, query_set: QuerySet):
        """
        Set the arguments for the oracle workflow

        :param o: the actual oracle
        :param training_set: dataset where the resolved query are inserted to
        :param query_set: dataset providing outstanding queries
        """

        logging.info(f"{oracle_controller_logging_prefix} Init oracle controller => set oracle, training set, query set")

        self.o = o
        self.training_set = training_set
        self.query_set = query_set

    def training_job(self, system_state: ValueProxy):
        """
        Actual training job for the oracle component => should run in separate process
        Also including the soft training end job

        Job sequence:
            1. retrieve a query request
                1. if unresolved query request exists:
                    1. resolve the query
                    2. add the labelled instance to the training set
                    3. restart job
                2. else:
                    1. sleep (or set state to FINISH_TRAINING__PL and return if state is FINISH_TRAINING__ORACLE)
                    2. restart job

        :param system_state: Shared variable over all parallel training processes; shows the state of the whole AL system (values align with enum SystemStates)
        :return: if the process should end => indicated by system_state
        """

        if system_state.value >= int(SystemStates.TERMINATE_TRAINING):
            logging.warning(f"{oracle_controller_logging_prefix} Training process was terminated => end training job (system_state: {SystemStates(system_state.value).name})")
            return
        try:
            # noinspection PyUnusedLocal
            query_instance = None
            try:
                query_instance = self.query_set.get_instance()
            except NoNewElementException:
                if system_state.value == int(SystemStates.FINISH_TRAINING__ORACLE):
                    logging.info(f"{oracle_controller_logging_prefix} Query database empty, all queries resolved => only PL needs to softly end training")

                    system_state.set(int(SystemStates.FINISH_TRAINING__PL))
                    logging.warning(f"{oracle_controller_logging_prefix} Enter PL finish training state => soft end (set system_state: {SystemStates(system_state.value).name})")

                    return

                else:
                    logging.info(f"{oracle_controller_logging_prefix} Wait for new queries")
                    time.sleep(5)

                    self.training_job(system_state)
                    return

            logging.info(f"{oracle_controller_logging_prefix} Retrieve unresolved query => will add label")
            label = self.o.query(query_instance)
            self.query_set.remove_instance(query_instance)
            self.training_set.append_labelled_instance(query_instance, label)
            logging.info(f"{oracle_controller_logging_prefix} Query for instance x resolved with label y, added to training set for PL; x = `{query_instance}`, y = `{label}`")
        except Exception as e:
            logging.error(f"{oracle_controller_logging_prefix} Exception during query job: {e}")
            system_state.set(int(SystemStates.FINISH_TRAINING__AL))
            return

        self.training_job(system_state)
        return
