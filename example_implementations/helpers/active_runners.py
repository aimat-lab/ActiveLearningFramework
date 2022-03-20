import logging
import os
import time
from multiprocessing import Process, Manager, Lock, synchronize
from multiprocessing.managers import ValueProxy
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from pyNNsMD.plots.pred import plot_scatter_prediction

from al_specific_components.candidate_update import get_candidate_source_type
from al_specific_components.query_selection import InformativenessAnalyser
from basic_sl_component_interfaces import PassiveLearner, Oracle, ReadOnlyPassiveLearner
from example_implementations.helpers import properties
from example_implementations.helpers.mapper import map_shape_input_to_flat, map_flat_output_to_shape, map_shape_output_to_flat
from example_implementations.helpers.metrics import calc_final_evaluation
from example_implementations.initiator import ButeneEnergyForceInitiator
from example_implementations.ua_initiator import ButeneEnergyForceInitiatorUnfiltered
from helpers import SystemStates, CandInfo, AddInfo_Y, Y, X, Scenarios
from helpers.exceptions import IncorrectScenarioImplementation, ALSystemError
from helpers.system_initiator import InitiationHelper
from workflow_management.controller import PassiveLearnerController, OracleController, CandidateUpdaterController, QuerySelectionController


def run_al(x, x_test, y, y_test):
    t0 = time.time()

    state_manager = Manager()
    system_state: ValueProxy = state_manager.Value('i', int(SystemStates.INITIALIZATION))
    sl_model_gets_stored: synchronize.Lock = Lock()
    cand_up: CandidateUpdaterController
    query_select: QuerySelectionController
    o: OracleController
    pl: PassiveLearnerController
    try:

        init_helper: InitiationHelper = ButeneEnergyForceInitiator(x, x_test, y, y_test)  # case implementation: implement initiation helper => rest of training/workflow management/... is done by the framework

        # set scenario
        scenario: Scenarios = init_helper.get_scenario()
        logging.info(f"Start of AL framework, chosen scenario: {scenario.name}")

        # WORKFLOW: Initialization
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
        (training_set, candidate_set, logging_query_decision_db, query_set) = init_helper.get_datasets()

        # init components (workflow controller)
        logging.info("Initialize components")

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
        query_select = QuerySelectionController(candidate_set=candidate_set, log_query_decision_db=logging_query_decision_db, query_set=query_set, scenario=scenario, info_analyser=info_analyser)

        # initial training, data source update
        logging.info("Initial training and first candidate update")
        x_train, y_train = init_helper.get_initial_training_data()
        pl.init_pl(x_train, y_train)  # training with initial training data
        cand_up.init_candidates()
        for i in range(len(x_train)):
            training_set.append_labelled_instance(x_train[i], y_train[i])  # add initial training data to stored labelled set # TODO: correct to do this?
            training_set.set_instance_not_use_for_training(x_train[i])

    except Exception as e:
        logging.error("During initialization, an unexpected error occurred", e)
        system_state.set(int(SystemStates.ERROR))

    # WORKFLOW: Training in parallel processes
    # from here on out, no further case dependent implementation necessary => just in initiation phase

    if system_state.value == int(SystemStates.INITIALIZATION):
        system_state.set(int(SystemStates.TRAINING))
    else:
        logging.error("An error occurred during initiation => system failed")
        raise ALSystemError()

    t1 = time.time()
    logging.info(f"------ Active Training ------ => system_state={SystemStates(system_state.value).name}")

    # create processes
    cand_updater_process = Process(target=cand_up.training_job, args=(system_state, sl_model_gets_stored), name="Process-AL-candidate-update")
    query_selection_process = Process(target=query_select.training_job, args=(system_state,), name="Process-AL-query-selection")
    o_process = Process(target=o.training_job, args=(system_state,), name="Process-Oracle")
    pl_process = Process(target=pl.training_job, args=(system_state, sl_model_gets_stored), name="Process-PL")

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

    t2 = time.time()
    logging.info(f"Every controller process has finished => system_state={SystemStates(system_state.value).name}")

    if system_state.value == int(SystemStates.ERROR):
        logging.error("A fatal error occurred => model training has failed")
        raise ALSystemError()

    # WORKFLOW: Prediction
    logging.info("Finished training process")
    system_state.set(int(SystemStates.PREDICT))
    logging.info(f"----- Prediction ------- => system_state={SystemStates(system_state.value).name}")

    pl.pl.load_model()
    pred_train, pred_test = pl.pl.predict_set(xs=map_shape_input_to_flat(x))[0], pl.pl.predict_set(xs=map_shape_input_to_flat(x_test))[0]
    _, mae_test, r2_test = calc_final_evaluation(pred_test, map_shape_output_to_flat(y_test), "IA test set", properties.entities["ia"] + "_test" + properties.prediction_image_suffix)
    _, mae_train, r2_train = calc_final_evaluation(pred_train, map_shape_output_to_flat(y), "IA train set", properties.entities["ia"] + "_train" + properties.prediction_image_suffix)

    reduced_x = np.load(os.path.join(properties.al_training_data_storage_location, properties.al_training_data_storage_x))

    return {
        "time_init": t1-t0,
        "time_training": t2-t1,
        "size_training_set": len(reduced_x),
        "mae_test": mae_test,
        "r2_test": r2_test,
        "mae_train": mae_train,
        "r2_train": r2_train
    }


def run_al_unfiltered(x, x_test, y, y_test):
    t0 = time.time()

    state_manager = Manager()
    system_state: ValueProxy = state_manager.Value('i', int(SystemStates.INITIALIZATION))
    sl_model_gets_stored: synchronize.Lock = Lock()
    cand_up: CandidateUpdaterController
    query_select: QuerySelectionController
    o: OracleController
    pl: PassiveLearnerController
    try:

        init_helper: InitiationHelper = ButeneEnergyForceInitiatorUnfiltered(x, x_test, y, y_test)  # case implementation: implement initiation helper => rest of training/workflow management/... is done by the framework

        # set scenario
        scenario: Scenarios = init_helper.get_scenario()
        logging.info(f"Start of AL framework, chosen scenario: {scenario.name}")

        # WORKFLOW: Initialization
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
        (training_set, candidate_set, logging_query_decision_db, query_set) = init_helper.get_datasets()

        # init components (workflow controller)
        logging.info("Initialize components")

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
        query_select = QuerySelectionController(candidate_set=candidate_set, log_query_decision_db=logging_query_decision_db, query_set=query_set, scenario=scenario, info_analyser=info_analyser)

        # initial training, data source update
        logging.info("Initial training and first candidate update")
        x_train, y_train = init_helper.get_initial_training_data()
        pl.init_pl(x_train, y_train)  # training with initial training data
        cand_up.init_candidates()
        for i in range(len(x_train)):
            training_set.append_labelled_instance(x_train[i], y_train[i])  # add initial training data to stored labelled set # TODO: correct to do this?
            training_set.set_instance_not_use_for_training(x_train[i])

    except Exception as e:
        logging.error("During initialization, an unexpected error occurred", e)
        system_state.set(int(SystemStates.ERROR))

    # WORKFLOW: Training in parallel processes
    # from here on out, no further case dependent implementation necessary => just in initiation phase

    if system_state.value == int(SystemStates.INITIALIZATION):
        system_state.set(int(SystemStates.TRAINING))
    else:
        logging.error("An error occurred during initiation => system failed")
        raise ALSystemError()

    t1 = time.time()
    logging.info(f"------ Active Training ------ => system_state={SystemStates(system_state.value).name}")

    # create processes
    cand_updater_process = Process(target=cand_up.training_job, args=(system_state, sl_model_gets_stored), name="Process-AL-candidate-update")
    query_selection_process = Process(target=query_select.training_job, args=(system_state,), name="Process-AL-query-selection")
    o_process = Process(target=o.training_job, args=(system_state,), name="Process-Oracle")
    pl_process = Process(target=pl.training_job, args=(system_state, sl_model_gets_stored), name="Process-PL")

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

    t2 = time.time()
    logging.info(f"Every controller process has finished => system_state={SystemStates(system_state.value).name}")

    if system_state.value == int(SystemStates.ERROR):
        logging.error("A fatal error occurred => model training has failed")
        raise ALSystemError()

    # WORKFLOW: Prediction
    logging.info("Finished training process")
    system_state.set(int(SystemStates.PREDICT))
    logging.info(f"----- Prediction ------- => system_state={SystemStates(system_state.value).name}")

    logging.info(f"time al model: initialisation {t1 - t0}, training {t2 - t1}, whole {(t2 - t0)}")
    reduced_x = np.load(os.path.join(properties.al_training_data_storage_location, properties.al_training_data_storage_x))

    pl.pl.load_model()
    pred_train, pred_test = pl.pl.predict_set(xs=map_shape_input_to_flat(x))[0], pl.pl.predict_set(xs=map_shape_input_to_flat(x_test))[0]
    _, mae_test, r2_test = calc_final_evaluation(pred_test, map_shape_output_to_flat(y_test), "UA test set", properties.entities["ua"] + "_test" + properties.prediction_image_suffix)
    _, mae_train, r2_train = calc_final_evaluation(pred_train, map_shape_output_to_flat(y), "UA train set", properties.entities["ua"] + "_train" + properties.prediction_image_suffix)

    return {
        "time_init": t1 - t0,
        "time_training": t2 - t1,
        "size_training_set": len(reduced_x),
        "mae_test": mae_test,
        "r2_test": r2_test,
        "mae_train": mae_train,
        "r2_train": r2_train
    }
