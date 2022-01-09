from dataclasses import dataclass

from scenario_dependend_interfaces import Oracle, TrainingSet, QuerySet


@dataclass()
class OracleController:
    o: Oracle
    training_set: TrainingSet
    query_set: QuerySet

    def training_job(self):
        query_instance = self.unlabelled_set.get_instance()
        label = self.o.query(query_instance)
        self.labelled_set.append_labelled_instance(query_instance, label)
