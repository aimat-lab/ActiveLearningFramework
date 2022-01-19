import logging
from multiprocessing import Process, Manager
from typing import Callable

from additional_component_interfaces import PassiveLearner, Oracle
from al_components.candidate_update import get_candidate_source_type, get_candidate_additional_information
from al_components.perfomance_evaluation import PerformanceEvaluator
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import SystemStates, CandInfo, AddInfo_Y, Y, X
from helpers.exceptions import IncorrectScenarioImplementation, ALSystemError
from workflow_management.controller import PassiveLearnerController, OracleController, ActiveLearnerController
from workflow_management.database_interfaces import TrainingSet, CandidateSet, QuerySet

logging.basicConfig(format='LOGGING:  %(levelname)s:%(message)s :END LOGGING', level=logging.INFO)

if __name__ == '__main__':
    # set scenario
    scenario = None  # TODO case implementation: set scenario
    logging.info(f"Start of AL framework, chosen scenario: {scenario.name}")

    # WORKFLOW: Initialization
    state_manager = Manager()
    system_state = state_manager.Value('i', int(SystemStates.INIT))
    logging.info(f"------ Initialize AL framework ------  => system_state={SystemStates(system_state.value).name}")

    # initialize candidate source
    # type of candidate_source depends on the scenario:
    #   - PbS: Pool => should be the candidate_set as well
    #   - MQS: Generator
    #   - SbS: Stream
    candidate_source_type = get_candidate_source_type()
    logging.info(f"Initialize datasource => type: {candidate_source_type}")

    candidate_source: candidate_source_type = None  # TODO case implementation: implement concrete candidate source => initialize accordingly
    if not isinstance(candidate_source, candidate_source_type):
        raise IncorrectScenarioImplementation(f"candidate_source needs to be of type {candidate_source_type}")

    # init databases (usually empty)
    logging.info("Initialize datasets")  # TODO case implementation: implement concrete datasets
    training_set: TrainingSet = None
    candidate_set: CandidateSet = None
    query_set: QuerySet = None

    # init components (workflow controller)
    logging.info("Initialize components")

    sl_model: PassiveLearner = None  # TODO case implementation: implement concrete sl model (passive learner)
    info_creator: Callable[[X, Y, AddInfo_Y], CandInfo] = get_candidate_additional_information  # TODO case implementation: implement concrete candidate information creation function
    pl_performance_evaluator: PerformanceEvaluator = None  # TODO case implementation: implement concrete sl performance evaluator
    pl = PassiveLearnerController(pl=sl_model, training_set=training_set, candidate_set=candidate_set, scenario=scenario, info_creator=info_creator, pl_evaluator=pl_performance_evaluator)

    oracle: Oracle = None  # TODO case implementation: implement concrete oracle (with knowledge about ground truth)
    o = OracleController(o=oracle, training_set=training_set, query_set=query_set)

    info_analyser: InformativenessAnalyser = None  # TODO case implementation: implement concrete informativeness analyser => base for query selection
    al = ActiveLearnerController(candidate_set=candidate_set, query_set=query_set, info_analyser=info_analyser, scenario=scenario)

    logging.info("Initial training and first candidate update")
    # initial training, data source update
    x_train, y_train = None, None  # TODO case implementation: set the initial training data for the sl model
    pl.init_pl(x_train, y_train, batch_size=8, epochs=10)  # training with initial training data
    pl.init_candidates()

    # WORKFLOW: Training in parallel processes
    if system_state.value == int(SystemStates.INIT):
        system_state.set(int(SystemStates.TRAINING))
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
    # TODO: how should prediction be performed???
