from dataclasses import dataclass

from additional_component_interfaces import Oracle
from workflow_management.database_interfaces import TrainingSet, QuerySet


@dataclass()
class ActiveLearnerController:
    o: Oracle
    training_set: TrainingSet
    query_set: QuerySet

    def training_job(self):
        query_instance = self.unlabelled_set.get_instance()
        label = self.o.query(query_instance)
        self.labelled_set.append_labelled_instance(query_instance, label)
