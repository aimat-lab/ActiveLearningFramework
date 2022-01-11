from dataclasses import dataclass

from additional_component_interfaces import PassiveLearner
from workflow_management import TrainingSet, CandidateSet


@dataclass()
class PassiveLearnerController:
    pl: PassiveLearner
    candidate_updater: CandidateUpdater
    training_set: TrainingSet
    candidate_set: CandidateSet

    def init_pl(self, batch_size, epochs):
        (x_train, y_train) = self.labelled_set.pop_all_labelled_instances()
        self.pl.initial_training(x_train, y_train, batch_size=batch_size, epochs=epochs)

    def training_job(self):
        (x_train, y_train) = self.labelled_set.pop_labelled_instance()
        self.pl.train(x_train, y_train)
