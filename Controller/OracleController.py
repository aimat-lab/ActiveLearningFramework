from Interfaces import Oracle, TrainingSet, QuerySet


class OracleController:

    def __init__(self, o: Oracle, training_set: TrainingSet, query_set: QuerySet):
        self.o = o
        self.labelled_set = training_set
        self.unlabelled_set = query_set

    def training_job(self):
        self.unlabelled_set
