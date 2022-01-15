import logging
from multiprocessing import Process

from keras.datasets import boston_housing

from al_components.candidate_update.candidate_updater_implementations import Pool
from example__house_pricing import SimpleRegressionHousing, CandidateSetHouses, OracleHouses, QuerySetHouses, TrainingSetHouses, UncertaintyInfoAnalyser
from helpers import Scenarios
from helpers.exceptions import NoMoreCandidatesException
from workflow_management.controller import PassiveLearnerController, OracleController, ActiveLearnerController
from workflow_management.database_interfaces import TrainingSet, CandidateSet, QuerySet

if __name__ == '__main__':
    # noinspection SpellCheckingInspection
    logging.basicConfig(format='LOGGING:  %(levelname)s:%(message)s :END LOGGING', level=logging.DEBUG)

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
    # WORKFLOW: Training in parallel processes

    # create processes
    al_process = Process(target=al.training_job, name="Process-AL")
    o_process = Process(target=o.training_job, name="Process-Oracle")
    # pl_process = Process(target=pl.training_job, name="Process-PL")

    try:
        # actually start the processes
        al_process.start()
        o_process.start()
        # pl_process.start()
        pl.training_job()
    except NoMoreCandidatesException:
        al_process.kill()
        o_process.kill()
        o.finish_training()


    # collect the processes
    al_process.join()
    o_process.join()
    # pl_process.join()
