from keras.datasets import boston_housing

from workflow_controller import PassiveLearnerController, OracleController
from example__house_pricing import SimpleRegressionHousing, OracleHouses
from example__house_pricing import TrainingSetHouses, QuerySetHouses, CandidateSetHouses

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.9)

    training_set = TrainingSetHouses(x_train, y_train)
    candidate_set = CandidateSetHouses()
    query_set = QuerySetHouses()

    pl = PassiveLearnerController(pl=SimpleRegressionHousing(), training_set=training_set, candidate_set=candidate_set)
    pl.init_pl(8, 10)

    o = OracleController(o=OracleHouses(x_test, y_test), training_set=training_set, query_set=query_set)

    for x in x_test:
        query_set.add_instance(x)

    for i in range(len(x_test)):
        o.training_job()
        pl.training_job()
