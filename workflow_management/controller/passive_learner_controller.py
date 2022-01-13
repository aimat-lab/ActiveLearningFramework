from additional_component_interfaces import PassiveLearner
from al_components.candidate_update import CandidateUpdater, init_candidate_updater
from helpers import Scenarios
from workflow_management.database_interfaces import TrainingSet, CandidateSet


class PassiveLearnerController:

    def __init__(self, pl: PassiveLearner, training_set: TrainingSet, candidate_set: CandidateSet, scenario: Scenarios, **kwargs):
        # TODO: documentation (especially kwargs)
        self.pl = pl
        self.labelled_set = training_set
        self.candidate_set = candidate_set
        source_stream = kwargs.get("source_stream")  # only needed if scenario = SbS
        self.candidate_updater: CandidateUpdater = init_candidate_updater(scenario, pl=pl, candidate_set=candidate_set, source_stream=source_stream)

    def init_pl(self, x_train, y_train, **kwargs):
        batch_size = kwargs.get("batch_size")
        epochs = kwargs.get("epochs")
        self.pl.initial_training(x_train, y_train, batch_size=batch_size, epochs=epochs)
        self.labelled_set.clear()

    def training_job(self):
        # TODO loop
        self.candidate_updater.update_candidate_set()
        (x_train, y_train) = self.labelled_set.retrieve_labelled_instance()
        self.pl.train(x_train, y_train)
        self.labelled_set.remove_labelled_instance(x_train)
