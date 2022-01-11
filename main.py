from keras.datasets import boston_housing

from al_components import init_candidate_updater
from example__house_pricing import SimpleRegressionHousing, OracleHouses
from example__house_pricing import TrainingSetHouses, QuerySetHouses, CandidateSetHouses
from helpers import Scenarios
from workflow_management import PassiveLearnerController, OracleController, TrainingSet, CandidateSet, QuerySet

if __name__ == '__main__':

    print("TEEST")
    init_candidate_updater(Scenarios.PbS)
    # WORKFLOW: Initialization

    # init databases (usually empty)
    training_set: TrainingSet = TrainingSetHouses()
    candidate_set: CandidateSet = CandidateSetHouses()
    query_set: QuerySet = QuerySetHouses()

    (x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.9)  # in this case: load data from existing set

    # init components (workflow controller)
    pl = PassiveLearnerController(pl=SimpleRegressionHousing(), training_set=training_set, candidate_set=candidate_set)
    o = OracleController(o=OracleHouses(x_test, y_test), training_set=training_set, query_set=query_set)
    # al = ActiveLearnerController()

    pl.init_pl(8, 10)

    for x in x_test:
        query_set.add_instance(x)

    for i in range(len(x_test)):
        o.training_job()
        pl.training_job()
