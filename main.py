from keras.datasets import boston_housing

from al_components.candidate_update.candidate_updater_implementations import Pool
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from example__house_pricing import SimpleRegressionHousing, CandidateSetHouses, OracleHouses, QuerySetHouses, TrainingSetHouses
from helpers import Scenarios
from workflow_management.controller import PassiveLearnerController, OracleController, ActiveLearnerController
from workflow_management.database_interfaces import TrainingSet, CandidateSet, QuerySet

if __name__ == '__main__':
    # WORKFLOW: Initialization
    scenario = Scenarios.PbS

    # type of candidate_source depends on the scenario:
    #   - PbS: Pool => should be the candidate_set
    #   - MQS: Generator
    #   - SbS: Stream
    candidate_source: Pool = CandidateSetHouses()

    # init databases (usually empty)
    training_set: TrainingSet = TrainingSetHouses()
    candidate_set: CandidateSet = candidate_source  # only in case of PbS same as candidate_source
    query_set: QuerySet = QuerySetHouses()

    (x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.9)  # in this case: load data from existing set

    class DefaultInfoAnalyser(InformativenessAnalyser):

        def get_informativeness(self, x):
            return 1


    # init components (workflow controller)
    pl = PassiveLearnerController(pl=SimpleRegressionHousing(), training_set=training_set, candidate_set=candidate_set, scenario=scenario)
    o = OracleController(o=OracleHouses(x_test, y_test), training_set=training_set, query_set=query_set)
    al = ActiveLearnerController(candidate_set=candidate_set, query_set=query_set, info_analyser=DefaultInfoAnalyser(), scenario=scenario)

    # initial training, data source
    pl.init_pl(x_train, y_train, batch_size=8, epochs=10)  # training with initial training data
    candidate_source.initiate_pool(x_test)

    # WORKFLOW: Training
    al.training_job()
    o.training_job()
    pl.training_job()
