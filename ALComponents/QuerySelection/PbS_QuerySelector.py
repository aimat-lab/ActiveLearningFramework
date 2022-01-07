from ALComponents.QuerySelection import QuerySelector, InformativenessAnalyser
from Interfaces import CandidateSet, TrainingSet


class PbS_QuerySelector(QuerySelector):

    def __init__(self, info_analyser: InformativenessAnalyser, candidate_set: CandidateSet, labelled_set: TrainingSet):
        self.info_analyser = info_analyser
        self.candidate_set = candidate_set
        self.labelled_set = labelled_set

    def select_query_instance(self):
        self.candidate_set.retrieve_all_instances()
