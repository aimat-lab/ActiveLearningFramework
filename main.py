import logging
from multiprocessing import Process, Manager
from typing import Callable

from additional_component_interfaces import PassiveLearner, Oracle, ReadOnlyPassiveLearner
from al_components.candidate_update import get_candidate_source_type
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import SystemStates, CandInfo, AddInfo_Y, Y, X, Scenarios
from helpers.exceptions import IncorrectScenarioImplementation, ALSystemError
from helpers.system_initiator import InitiationHelper
from workflow_management.controller import PassiveLearnerController, OracleController, CandidateUpdaterController, QuerySelectionController

logging.basicConfig(format='LOGGING:  %(levelname)s:%(message)s :END LOGGING', level=logging.INFO)

if __name__ == '__main__':

    init_helper: InitiationHelper = InitiationHelper()  # case implementation: implement initiation helper => rest of training/workflow management/... is done by the framework

    # set scenario
    scenario: Scenarios = init_helper.get_scenario()
    logging.info(f"Start of AL framework, chosen scenario: {scenario.name}")

    # WORKFLOW: Initialization
    state_manager = Manager()
    system_state = state_manager.Value('i', int(SystemStates.INITIALIZATION))

    logging.info(f"------ Initialize AL framework ------  => system_state={SystemStates(system_state.value).name}")

    # initialize candidate source
    # type of candidate_source depends on the scenario
    candidate_source_type = get_candidate_source_type(scenario)
    logging.info(f"Initialize datasource => type: {candidate_source_type}")
    candidate_source: candidate_source_type = init_helper.get_candidate_source()
    if not isinstance(candidate_source, candidate_source_type):
        system_state.set(int(SystemStates.ERROR))
        raise IncorrectScenarioImplementation(f"candidate_source needs to be of type {candidate_source_type}")

    # init databases (usually empty)
    logging.info("Initialize datasets")
    (training_set, stored_labelled_set_db, candidate_set, log_query_decision_db, query_set) = init_helper.get_datasets()

    # init components (workflow controller)
    logging.info("Initialize components")

    # init passive learner
    sl_model: PassiveLearner = init_helper.get_sl_model()
    pl = PassiveLearnerController(pl=sl_model, training_set=training_set, scenario=scenario)

    # init oracle
    oracle: Oracle = init_helper.get_oracle()
    # set the oracle controller
    o = OracleController(o=oracle, training_set=training_set, stored_labelled_set=stored_labelled_set_db, query_set=query_set)

    # init candidate updater
    cand_info_mapping: Callable[[X, Y, AddInfo_Y], CandInfo] = init_helper.get_mapper_function_prediction_to_candidate_info()
    ro_pl: ReadOnlyPassiveLearner = init_helper.get_ro_sl_model()
    cand_up = CandidateUpdaterController(ro_pl=ro_pl, candidate_set=candidate_set, cand_info_mapping=cand_info_mapping, scenario=scenario, candidate_source=candidate_source)

    # init info analyser (query selection)
    info_analyser: InformativenessAnalyser = init_helper.get_informativeness_analyser()
    query_select = QuerySelectionController(candidate_set=candidate_set, log_query_decision_db=log_query_decision_db, query_set=query_set, scenario=scenario, info_analyser=info_analyser)

    # initial training, data source update
    logging.info("Initial training and first candidate update")
    x_train, y_train, epochs, batch_size = init_helper.get_initial_training_data()
    pl.init_pl(x_train, y_train, batch_size=batch_size, epochs=epochs)  # training with initial training data
    cand_up.init_candidates()

    # WORKFLOW: Training in parallel processes
    # from here on out, no further case dependent implementation necessary => just in initiation phase

    if system_state.value == int(SystemStates.INITIALIZATION):
        system_state.set(int(SystemStates.TRAINING))
    else:
        logging.error("An error occurred during initiation => system failed")
        raise ALSystemError()

    logging.info(f"------ Active Training ------ => system_state={SystemStates(system_state.value).name}")

    # create processes
    cand_updater_process = Process(target=cand_up.training_job, args=(system_state,), name="Process-AL-candidate-update")
    query_selection_process = Process(target=query_select.training_job, args=(system_state,), name="Process-AL-query-selection")
    o_process = Process(target=o.training_job, args=(system_state,), name="Process-Oracle")
    pl_process = Process(target=pl.training_job, args=(system_state,), name="Process-PL")

    logging.info(f"Start every controller process: al candidate update - {cand_updater_process.name}, al query selection - {query_selection_process.name}, oracle - {o_process.name}, pl - {pl_process.name}")
    # actually start the processes
    query_selection_process.start()
    logging.info(f"al query selection process ({query_selection_process.name}) started")
    o_process.start()
    logging.info(f"oracle process ({o_process.name}) started")
    pl_process.start()
    logging.info(f"pl process ({pl_process.name}) started")
    cand_updater_process.start()
    logging.info(f"al candidate update process ({cand_updater_process.name}) started")

    # collect the processes
    cand_updater_process.join()
    query_selection_process.join()
    o_process.join()
    pl_process.join()
    logging.info(f"Every controller process has finished => system_state={SystemStates(system_state.value).name}")

    if system_state.value == int(SystemStates.ERROR):
        logging.error("A fatal error occurred => model training has failed")
        raise ALSystemError()

    # WORKFLOW: Prediction
    logging.info("Finished training process")
    system_state.set(int(SystemStates.PREDICT))
    logging.info(f"----- Prediction ------- => system_state={SystemStates(system_state.value).name}")

    # case implementation: results are available (use the stored SL model for predictions or use the stored labelled set for further training)
