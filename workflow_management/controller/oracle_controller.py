import logging
import time
from dataclasses import dataclass

from additional_component_interfaces import Oracle
from helpers.exceptions import NoNewElementException
from workflow_management.database_interfaces import TrainingSet, QuerySet


@dataclass()
class OracleController:
    """
    Controls the oracle workflow (manages query process)

    Arguments for initiation
        - *o: Oracle* - the actual oracle
        - *training_set: TrainingSet* - the set the resolved query is inserted to
        - *query_set: QuerySet* - the set providing outstanding queries
    """
    o: Oracle
    training_set: TrainingSet
    query_set: QuerySet

    def training_job(self):
        query_instance = None
        try:
            query_instance = self.query_set.get_instance()
        except NoNewElementException:
            logging.info("Wait for new queries")
            time.sleep(5)
            self.training_job()

        label = self.o.query(query_instance)
        self.query_set.remove_instance(query_instance)
        self.training_set.append_labelled_instance(query_instance, label)
        logging.info(f"Query for instance x resolved with label y, added to training set for PL; x = `{query_instance}`, y = `{label}`")

        self.training_job()

    def finish_training(self):
        while True:
            # noinspection PyUnusedLocal
            query_instance = None
            try:
                query_instance = self.query_set.get_instance()
            except NoNewElementException:
                logging.info("Finished handling every outstanding query")
                break
            label = self.o.query(query_instance)
            self.query_set.remove_instance(query_instance)
            self.training_set.append_labelled_instance(query_instance, label)
            logging.info(f"Query for instance x resolved with label y, added to training set for PL; x = `{query_instance}`, y = `{label}`")
