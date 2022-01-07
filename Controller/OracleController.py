from Interfaces import Oracle, TrainingSet, QuerySet


class OracleController:

    def __init__(self, o: Oracle, training_set: TrainingSet, query_set: QuerySet):
        self.o = o
        self.labelled_set = training_set
        self.unlabelled_set = query_set

    def training_job(self):
        query_instance = self.unlabelled_set.pop_instance()
        label = self.o.query(query_instance)
        self.labelled_set.append_labelled_instance(query_instance, label)
