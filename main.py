from keras.datasets import boston_housing

from Controller import PassiveLearnerController, OracleController
from HousePricingExample import SimpleRegression_Housing
from HousePricingExample import TrainingSetHouses, QuerySetHouses, CandidateSetHouses

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.9)

    training_set = TrainingSetHouses(x_train, y_train)
    candidate_set = CandidateSetHouses()
    query_set = QuerySetHouses()

    pl = PassiveLearnerController(pl=SimpleRegression_Housing(), training_set=training_set, candidate_set=candidate_set)
    # pl.init_pl(8, 10)

    query_set.add_instance(x_test[0])
    query_set.pop_instance()


