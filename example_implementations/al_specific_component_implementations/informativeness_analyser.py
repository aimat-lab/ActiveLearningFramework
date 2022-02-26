from al_specific_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import X


class EverythingIsInformativeAnalyser(InformativenessAnalyser):

    def get_informativeness(self, x: X) -> float:
        return 1
