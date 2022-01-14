import logging

from additional_component_interfaces import PassiveLearner
from al_components.candidate_update import CandidateUpdater, init_candidate_updater
from helpers import Scenarios
from helpers.exceptions import NoNewElementException
from workflow_management.database_interfaces import TrainingSet, CandidateSet


class PassiveLearnerController:

    def __init__(self, pl: PassiveLearner, training_set: TrainingSet, candidate_set: CandidateSet, scenario: Scenarios, **kwargs):
        # TODO: documentation (especially kwargs)
        self.pl = pl
        self.training_set = training_set
        self.candidate_set = candidate_set
        candidate_source = kwargs.get("candidate_source")  # not needed in PbS scenario (candidate_set = source)

        self.candidate_updater: CandidateUpdater = init_candidate_updater(scenario, pl=pl, candidate_set=candidate_set, candidate_source=candidate_source)

        self.pl_and_candidates_align = False  # keeps track of whether it makes sense to update candidates => if nothing changed in pl, candidates don't need to be updated

    def init_pl(self, x_train, y_train, **kwargs):
        batch_size = kwargs.get("batch_size")
        epochs = kwargs.get("epochs")
        self.pl.initial_training(x_train, y_train, batch_size=batch_size, epochs=epochs)
        self.training_set.clear()
        self.pl_and_candidates_align = False

    def init_candidates(self):
        self.candidate_updater.update_candidate_set()
        self.pl_and_candidates_align = True

    def training_job(self):
        # TODO: check if assumption "pl doesn't change => candidates don't need to be updated" works
        # # for PbS definitely
        # # for SbS/MQS: only if candidate set is not empty => bring into calculation?
        if not self.pl_and_candidates_align:
            self.candidate_updater.update_candidate_set()
            self.pl_and_candidates_align = True

        (x_train, y_train) = None, None
        try:
            (x_train, y_train) = self.training_set.retrieve_labelled_instance()
        except NoNewElementException:
            logging.info("Wait for new training data")

        logging.info(f"Train PL with (x, y): x = `{x_train}`, y = `{y_train}`")
        self.pl.train(x_train, y_train)
        self.pl_and_candidates_align = False
        self.training_set.remove_labelled_instance(x_train)
        logging.info(f"Removed (x, y) from the training set => PL already trained with it: x = `{x_train}`, y = `{y_train}`")

        # TODO loop => currently not active, fist: multiprocessing
        # self.training_job()

