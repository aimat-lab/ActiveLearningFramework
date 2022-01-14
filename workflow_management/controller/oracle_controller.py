import logging
from dataclasses import dataclass

from additional_component_interfaces import Oracle
from helpers.exceptions import NoNewElementException
from workflow_management.database_interfaces import TrainingSet, QuerySet


@dataclass()
class OracleController:
    o: Oracle
    training_set: TrainingSet
    query_set: QuerySet

    def training_job(self):
        # TODO loop
        query_instance = None
        try:
            query_instance = self.query_set.get_instance()
        except NoNewElementException:
            logging.info("Wait for new candidates")

        label = self.o.query(query_instance)
        self.query_set.remove_instance(query_instance)
        self.training_set.append_labelled_instance(query_instance, label)
        logging.info(f"Query for instance x resolved with label y, added to training set for PL; x = `{query_instance}`, y = `{label}`")

        # TODO loop => currently not active, fist: multiprocessing
        # self.training_job()
