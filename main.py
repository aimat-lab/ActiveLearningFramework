import logging

from keras.datasets import boston_housing

from al_components.candidate_update.candidate_updater_implementations import Pool
from example__house_pricing import SimpleRegressionHousing, CandidateSetHouses, OracleHouses, QuerySetHouses, TrainingSetHouses, UncertaintyInfoAnalyser
from helpers import Scenarios
from workflow_management.controller import PassiveLearnerController, OracleController, ActiveLearnerController
from workflow_management.database_interfaces import TrainingSet, CandidateSet, QuerySet

if __name__ == '__main__':
    # noinspection SpellCheckingInspection
    logging.basicConfig(format='LOGGING:  %(levelname)s:%(message)s', level=logging.DEBUG)

    # set scenario
    scenario = Scenarios.PbS
    logging.info(f"Start of AL framework, chosen scenario: {scenario.name}")

    # WORKFLOW: Initialization
    logging.info("------Initialize AL framework------")

    # in this case: load data from existing set
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.9)
    x_test = x_test[:50]
    y_test = y_test[:50]

    logging.info("Initialize datasets")
    # type of candidate_source depends on the scenario:
    #   - PbS: Pool => should be the candidate_set as well
    #   - MQS: Generator
    #   - SbS: Stream
    candidate_source: Pool = CandidateSetHouses()

    # init databases (usually empty)
    training_set: TrainingSet = TrainingSetHouses()
    candidate_set: CandidateSet = candidate_source  # only in case of PbS same as candidate_source
    query_set: QuerySet = QuerySetHouses()

    candidate_source.initiate_pool(x_test)

    logging.info("Initialize components")
    # init components (workflow controller)
    pl = PassiveLearnerController(pl=SimpleRegressionHousing(), training_set=training_set, candidate_set=candidate_set, scenario=scenario)
    o = OracleController(o=OracleHouses(x_test, y_test), training_set=training_set, query_set=query_set)
    al = ActiveLearnerController(candidate_set=candidate_set, query_set=query_set, info_analyser=UncertaintyInfoAnalyser(candidate_set), scenario=scenario)

    logging.info("Initial training and first candidate update")
    # initial training, data source
    pl.init_pl(x_train, y_train, batch_size=8, epochs=10)  # training with initial training data
    pl.init_candidates()

    logging.info("------Active Training------")
    # WORKFLOW: Training
    al.training_job()
    o.training_job()
    pl.training_job()
