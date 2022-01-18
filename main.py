import logging
from multiprocessing import Process, Manager

from keras.datasets import boston_housing

from al_components.candidate_update.candidate_updater_implementations import Pool, Stream, Generator
from example__house_pricing import SimpleRegressionHousing, CandidateSetHouses, OracleHouses, QuerySetHouses, TrainingSetHouses, UncertaintyInfoAnalyser
from helpers import Scenarios, SystemStates
from helpers.exceptions import IncorrectScenarioImplementation, ALSystemError
from workflow_management.controller import PassiveLearnerController, OracleController, ActiveLearnerController
from workflow_management.database_interfaces import TrainingSet, CandidateSet, QuerySet

logging.basicConfig(format='LOGGING:  %(levelname)s:%(message)s :END LOGGING', level=logging.INFO)

if __name__ == '__main__':
    # set scenario
    scenario = Scenarios.PbS
    logging.info(f"Start of AL framework, chosen scenario: {scenario.name}")

    # WORKFLOW: Initialization
    state_manager = Manager()
    system_state = state_manager.Value('i', int(SystemStates.INIT))
    logging.info(f"------ Initialize AL framework ------  => system_state={SystemStates(system_state.value).name}")

    # in boston housing case: load data from existing set
    (x_train, y_train), (x_test_ori, y_test_ori) = boston_housing.load_data(test_split=0.9)
    x_test = x_test_ori[:50]
    y_test = y_test_ori[:50]

    # initialize candidate source
    # type of candidate_source depends on the scenario:
    #   - PbS: Pool => should be the candidate_set as well
    #   - MQS: Generator
    #   - SbS: Stream
    candidate_source_type = None
    if scenario == Scenarios.PbS:
        candidate_source_type = Pool
    elif scenario == Scenarios.SbS:
        candidate_source_type = Stream
    else:  # scenario == Scenarios.MQS
        candidate_source_type = Generator
    logging.info(f"Initialize datasource => type: {candidate_source_type}")

    candidate_source: candidate_source_type = CandidateSetHouses()
    if not isinstance(candidate_source, candidate_source_type):
        raise IncorrectScenarioImplementation(f"candidate_source needs to be of type {candidate_source_type}, but is of type {candidate_source.__class__}")

    candidate_source.initiate_pool(x_test)  # load data into candidate_source

    # init databases (usually empty)
    logging.info("Initialize datasets")
    training_set: TrainingSet = TrainingSetHouses()
    candidate_set: CandidateSet = candidate_source  # only in case of PbS same as candidate_source
    query_set: QuerySet = QuerySetHouses()

    # init components (workflow controller)
    logging.info("Initialize components")
    pl = PassiveLearnerController(pl=SimpleRegressionHousing(), training_set=training_set, candidate_set=candidate_set, scenario=scenario)
    o = OracleController(o=OracleHouses(x_test, y_test), training_set=training_set, query_set=query_set)
    al = ActiveLearnerController(candidate_set=candidate_set, query_set=query_set, info_analyser=UncertaintyInfoAnalyser(candidate_set), scenario=scenario)

    logging.info("Initial training and first candidate update")
    # initial training, data source
    pl.init_pl(x_train, y_train, batch_size=8, epochs=10)  # training with initial training data
    pl.init_candidates()

    # WORKFLOW: Training in parallel processes
    if system_state == int(SystemStates.INIT):
        system_state = state_manager.Value('i', int(SystemStates.TRAINING))
    else:
        logging.error("An error occurred during initiation => system failed")
        raise ALSystemError()

    logging.info(f"------ Active Training ------ => system_state={SystemStates(system_state.value).name}")

    # create processes
    al_process = Process(target=al.training_job, args=(system_state,), name="Process-AL")
    o_process = Process(target=o.training_job, args=(system_state,), name="Process-Oracle")
    pl_process = Process(target=pl.training_job, args=(system_state,), name="Process-PL")

    logging.info(f"Start every controller process: al - {al_process.name}, oracle - {o_process.name}, pl - {pl_process.name}")
    # actually start the processes
    al_process.start()
    o_process.start()
    pl_process.start()

    # collect the processes
    al_process.join()
    o_process.join()
    pl_process.join()
    logging.info(f"Every controller process has finished => system_state={SystemStates(system_state.value).name}")

    # TODO: implement terminate training case => if convergence
    if system_state.value == int(SystemStates.FINISH_TRAINING):
        logging.info("Soft end for training process => empty query set and training set")
        o.finish_training()
        pl.finish_training()

    elif system_state.value == int(SystemStates.ERROR):
        logging.error("A fatal error occurred => model training has failed")
        raise ALSystemError()

    # WORKFLOW: Prediction
    logging.info("Finished training process")
    system_state.set(int(SystemStates.PREDICT))
    logging.info(f"----- Prediction ------- => system_state={SystemStates(system_state.value).name}")

    pl.pl.load_model()
    predict_40 = pl.pl.predict(x_test_ori[40])
    print(f"prediction: {predict_40}, {y_test_ori[40]}")
    predict_50 = pl.pl.predict(x_test_ori[50])
    print(f"prediction: {predict_50}, {y_test_ori[50]}")
    predict_60 = pl.pl.predict(x_test_ori[60])
    print(f"prediction: {predict_60}, {y_test_ori[60]}")
