import logging
import time
from multiprocessing import Process, Manager, Lock, synchronize
from multiprocessing.managers import ValueProxy
from typing import Callable

from al_specific_components.candidate_update import get_candidate_source_type
from al_specific_components.query_selection import InformativenessAnalyser
from basic_sl_component_interfaces import PassiveLearner, Oracle, ReadOnlyPassiveLearner
from example_implementations.initiator import ButeneEnergyForceInitiator
from helpers import SystemStates, CandInfo, AddInfo_Y, Y, X, Scenarios
from helpers.exceptions import IncorrectScenarioImplementation, ALSystemError
from helpers.system_initiator import InitiationHelper
from try_out.loaded_train_energy_force_butene import load
from workflow_management.controller import PassiveLearnerController, OracleController, CandidateUpdaterController, QuerySelectionController

logging.basicConfig(format='\nLOGGING: %(name)s, %(levelname)s: %(message)s :END LOGGING', level=logging.INFO)
log = logging.getLogger("Main logger")

if __name__ == '__main__':
    #load()
#else:
    state_manager = Manager()
    system_state: ValueProxy = state_manager.Value('i', int(SystemStates.INITIALIZATION))
    sl_model_gets_stored: synchronize.Lock = Lock()
    cand_up: CandidateUpdaterController
    query_select: QuerySelectionController
    o: OracleController
    pl: PassiveLearnerController
    try:

        init_helper: InitiationHelper = ButeneEnergyForceInitiator()  # case implementation: implement initiation helper => rest of training/workflow management/... is done by the framework

        # set scenario
        scenario: Scenarios = init_helper.get_scenario()
        log.info(f"Start of AL framework, chosen scenario: {scenario.name}")

        # WORKFLOW: Initialization
        log.info(f"------ Initialize AL framework ------  => system_state={SystemStates(system_state.value).name}")

        # initialize candidate source
        # type of candidate_source depends on the scenario
        candidate_source_type = get_candidate_source_type(scenario)
        log.info(f"Initialize datasource => type: {candidate_source_type}")
        candidate_source: candidate_source_type = init_helper.get_candidate_source()
        if not isinstance(candidate_source, candidate_source_type):
            system_state.set(int(SystemStates.ERROR))
            raise IncorrectScenarioImplementation(f"candidate_source needs to be of type {candidate_source_type}")

        # init databases (usually empty)
        log.info("Initialize datasets")
        (training_set, candidate_set, log_query_decision_db, query_set) = init_helper.get_datasets()

        # init components (workflow controller)
        log.info("Initialize components")

        # init passive learner
        sl_model: PassiveLearner = init_helper.get_pl()
        pl = PassiveLearnerController(pl=sl_model, training_set=training_set)

        # init oracle
        oracle: Oracle = init_helper.get_oracle()
        # set the oracle controller
        o = OracleController(o=oracle, training_set=training_set, query_set=query_set)

        # init candidate updater
        cand_info_mapping: Callable[[X, Y, AddInfo_Y], CandInfo] = init_helper.get_mapper_function_prediction_to_candidate_info()
        ro_pl: ReadOnlyPassiveLearner = init_helper.get_ro_pl()
        cand_up = CandidateUpdaterController(ro_pl=ro_pl, candidate_set=candidate_set, cand_info_mapping=cand_info_mapping, scenario=scenario, candidate_source=candidate_source)

        # init info analyser (query selection)
        info_analyser: InformativenessAnalyser = init_helper.get_informativeness_analyser()
        query_select = QuerySelectionController(candidate_set=candidate_set, log_query_decision_db=log_query_decision_db, query_set=query_set, scenario=scenario, info_analyser=info_analyser)

        # initial training, data source update
        log.info("Initial training and first candidate update")
        x_train, y_train = init_helper.get_initial_training_data()
        pl.init_pl(x_train, y_train)  # training with initial training data
        cand_up.init_candidates()
        for i in range(len(x_train)):
            training_set.append_labelled_instance(x_train[i], y_train[i])  # add initial training data to stored labelled set # TODO: correct to do this?
            training_set.set_instance_not_use_for_training(x_train[i])

    except Exception as e:
        log.error("During initialization, an unexpected error occurred", e)
        system_state.set(int(SystemStates.ERROR))

    # WORKFLOW: Training in parallel processes
    # from here on out, no further case dependent implementation necessary => just in initiation phase

    if system_state.value == int(SystemStates.INITIALIZATION):
        system_state.set(int(SystemStates.TRAINING))
    else:
        log.error("An error occurred during initiation => system failed")
        raise ALSystemError()

    log.info(f"------ Active Training ------ => system_state={SystemStates(system_state.value).name}")

    # create processes
    cand_updater_process = Process(target=cand_up.training_job, args=(system_state, sl_model_gets_stored), name="Process-AL-candidate-update")
    query_selection_process = Process(target=query_select.training_job, args=(system_state,), name="Process-AL-query-selection")
    o_process = Process(target=o.training_job, args=(system_state,), name="Process-Oracle")
    pl_process = Process(target=pl.training_job, args=(system_state, sl_model_gets_stored), name="Process-PL")

    log.info(f"Start every controller process: al candidate update - {cand_updater_process.name}, al query selection - {query_selection_process.name}, oracle - {o_process.name}, pl - {pl_process.name}")
    # actually start the processes
    query_selection_process.start()
    log.info(f"al query selection process ({query_selection_process.name}) started")
    o_process.start()
    log.info(f"oracle process ({o_process.name}) started")
    pl_process.start()
    log.info(f"pl process ({pl_process.name}) started")
    cand_updater_process.start()
    log.info(f"al candidate update process ({cand_updater_process.name}) started")

    # collect the processes
    cand_updater_process.join()
    query_selection_process.join()
    o_process.join()
    pl_process.join()

    log.info(f"Every controller process has finished => system_state={SystemStates(system_state.value).name}")

    if system_state.value == int(SystemStates.ERROR):
        log.error("A fatal error occurred => model training has failed")
        raise ALSystemError()

    # WORKFLOW: Prediction
    log.info("Finished training process")
    system_state.set(int(SystemStates.PREDICT))
    log.info(f"----- Prediction ------- => system_state={SystemStates(system_state.value).name}")

    # case implementation: results are available (use the stored SL model for predictions or use the stored labelled set for further training)

    load()
