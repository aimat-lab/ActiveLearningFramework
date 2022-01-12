from dataclasses import dataclass

from additional_component_interfaces import Oracle
from workflow_management.database_interfaces import TrainingSet, QuerySet


@dataclass()
class OracleController:
    o: Oracle
    training_set: TrainingSet
    query_set: QuerySet

    def training_job(self):
        # TODO loop
        query_instance = self.query_set.get_instance()
        label = self.o.query(query_instance)
        self.query_set.remove_instance(query_instance)
        self.training_set.append_labelled_instance(query_instance, label)
