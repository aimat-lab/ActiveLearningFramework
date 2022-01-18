from al_components.candidate_update import CandidateInformationCreator
from helpers import X, Y, AddInfo_Y, CandInfo


class DefaultCandidateInformationCreator(CandidateInformationCreator):

    def get_candidate_additional_information(self, x: X, prediction: Y, additional_prediction_info: AddInfo_Y) -> CandInfo:
        return prediction, additional_prediction_info
