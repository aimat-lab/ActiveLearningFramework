from Interfaces import PassiveLearner, TrainingSet, CandidateSet


class PassiveLearnerController:

    def __init__(self, pl: PassiveLearner, training_set: TrainingSet, candidate_set: CandidateSet):
        self.pl = pl
        self.labelled_set = training_set
        self.unlabelled_set = candidate_set

    def init_pl(self):
        # check for instances in labelled set => need to provide some initial instances as a starting point
        self.pl.initial_training(None, None)

    def training_job(self):
        self.labelled_set
