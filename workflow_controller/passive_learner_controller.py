from scenario_dependend_interfaces import PassiveLearner, TrainingSet, CandidateSet

from dataclasses import dataclass


@dataclass()
class PassiveLearnerController:
    pl: PassiveLearner
    training_set: TrainingSet
    candidate_set: CandidateSet

    def init_pl(self, batch_size, epochs):
        (x_train, y_train) = self.labelled_set.pop_all_labelled_instances()
        self.pl.initial_training(x_train, y_train, batch_size=batch_size, epochs=epochs)

    def training_job(self):
        (x_train, y_train) = self.labelled_set.pop_labelled_instance()
        self.pl.train(x_train, y_train)
