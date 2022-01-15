import logging
from multiprocessing import Process, Manager

from keras.datasets import boston_housing

from al_components.candidate_update.candidate_updater_implementations import Pool
from example__house_pricing import SimpleRegressionHousing, CandidateSetHouses, OracleHouses, QuerySetHouses, TrainingSetHouses, UncertaintyInfoAnalyser
from helpers import Scenarios, SystemStates
from workflow_management.controller import PassiveLearnerController, OracleController, ActiveLearnerController
from workflow_management.database_interfaces import TrainingSet, CandidateSet, QuerySet

logging.basicConfig(format='LOGGING:  %(levelname)s:%(message)s :END LOGGING', level=logging.DEBUG)

if __name__ == '__main__':
    # set scenario
    scenario = Scenarios.PbS
    logging.info(f"Start of AL framework, chosen scenario: {scenario.name}")

    state_manager = Manager()

    # WORKFLOW: Initialization
    logging.info("------Initialize AL framework------")
    system_state = state_manager.Value('i', int(SystemStates.Init))

    # in this case: load data from existing set
    (x_train, y_train), (x_test_ori, y_test_ori) = boston_housing.load_data(test_split=0.9)
    x_test = x_test_ori
    y_test = y_test_ori

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
    system_state = state_manager.Value('i', int(SystemStates.Training))
    # WORKFLOW: Training in parallel processes

    # create processes
    al_process = Process(target=al.training_job, args=(system_state,), name="Process-AL")
    o_process = Process(target=o.training_job, args=(system_state,), name="Process-Oracle")
    pl_process = Process(target=pl.training_job, args=(system_state,), name="Process-PL")

    # actually start the processes
    al_process.start()
    o_process.start()
    pl_process.start()
    # pl.training_job()

    # collect the processes
    al_process.join()
    o_process.join()
    pl_process.join()

    if system_state.value == int(SystemStates.FinishTraining):
        o.finish_training()
        pl.finish_training()

    system_state.set(int(SystemStates.Predict))
    logging.info("Finished training => PREDICT")

    pl.pl.load_model()
    predict_40 = pl.pl.predict(x_test_ori[40])
    print(f"prediction: {predict_40}, {y_test_ori[40]}")
    predict_50 = pl.pl.predict(x_test_ori[50])
    print(f"prediction: {predict_50}, {y_test_ori[50]}")
    predict_60 = pl.pl.predict(x_test_ori[60])
    print(f"prediction: {predict_60}, {y_test_ori[60]}")
