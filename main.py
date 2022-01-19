import logging
from multiprocessing import Process, Manager
from typing import Callable, Tuple, List, Optional

from nparray import ndarray

from additional_component_interfaces import PassiveLearner, Oracle
from al_components.candidate_update import get_candidate_source_type, get_candidate_additional_information
from al_components.candidate_update.candidate_updater_implementations import Generator, Stream, Pool
from al_components.perfomance_evaluation import PerformanceEvaluator
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import SystemStates, CandInfo, AddInfo_Y, Y, X, Scenarios
from helpers.exceptions import IncorrectScenarioImplementation, ALSystemError, IncorrectParameters
from workflow_management.controller import PassiveLearnerController, OracleController, ActiveLearnerController
from workflow_management.database_interfaces import TrainingSet, CandidateSet, QuerySet

logging.basicConfig(format='LOGGING:  %(levelname)s:%(message)s :END LOGGING', level=logging.INFO)


class InitiationHelper:

    def get_scenario(self) -> Scenarios:
        """
        Determines the scenario for the whole AL process

        :return: the selected scenario
        """
        # TODO case implementation: set scenario
        raise NotImplementedError

    def get_candidate_source(self) -> Pool or Stream or Generator:
        """
        Determines the source for the candidates => concrete type is scenario dependent (see get_candidate_source_type)

        :return: the candidate source
        """
        # TODO case implementation: implement concrete candidate source => initialize accordingly
        raise NotImplementedError

    def get_datasets(self) -> Tuple[TrainingSet, CandidateSet, QuerySet]:
        """
        Returns all implemented datasets

        - if PbS: candidate set must match candidate source

        :return: the datasets needed for the workflow
        """
        # TODO case implementation: implement concrete datasets
        raise NotImplementedError

    def get_sl_model(self) -> PassiveLearner:
        """
        Returns the extended sl/pl model (need to implement the PassiveLearner interface)

        :return: the model
        """
        # TODO case implementation: implement concrete sl model (passive learner)
        raise NotImplementedError

    def get_initial_training_data(self) -> Tuple[List[X] or ndarray, List[Y] or ndarray, Optional, Optional]:
        """
        Returns labelled data for the initial training episode of the pl

        can return additional properties for this training:
            - epochs
            - batch_size

        :return: input data, correct output, optional epochs, optional batch_size
        """
        # TODO case implementation: set the initial training data for the sl model
        raise NotImplementedError

    def get_mapper_function_prediction_to_candidate_info(self) -> Callable[[X, Y, AddInfo_Y], CandInfo]:
        """
        Return a function => see *al_components.candidate_update.get_candidate_additional_information* for more information

        :return: the mapper function
        """
        function = get_candidate_additional_information  # TODO case implementation: implement concrete candidate information creation function
        raise NotImplementedError

    def get_pl_performance_evaluator(self) -> PerformanceEvaluator:
        """
        Return a performance evaluator (evaluate performance of pl form perspective of the AL project => performance of AL model = performance of PL model)

        :return: the evaluator
        """
        # TODO case implementation: implement concrete sl performance evaluator
        raise NotImplementedError

    def get_oracle(self) -> Oracle:
        """
        Return the instance correctly labelling the unlabelled data then to be added to the training data

        :return: the oracle instance
        """
        # TODO case implementation: implement concrete oracle (with knowledge about ground truth)
        raise NotImplementedError

    def get_informativeness_analyser(self) -> InformativenessAnalyser:
        """
        Return the instance evaluating the performance of a candidate

        :return: the informativeness analyser
        """
        # TODO case implementation: implement concrete informativeness analyser => foundation for query selection
        raise NotImplementedError


if __name__ == '__main__':

    init_helper: InitiationHelper = InitiationHelper()  # TODO case implementation: implement initiation helper => rest of training/workflow management/... is done by the framework

    # set scenario
    scenario: Scenarios = init_helper.get_scenario()
    logging.info(f"Start of AL framework, chosen scenario: {scenario.name}")

    # WORKFLOW: Initialization
    state_manager = Manager()
    system_state = state_manager.Value('i', int(SystemStates.INIT))

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
    (training_set, candidate_set, query_set) = init_helper.get_datasets()

    # init components (workflow controller)
    logging.info("Initialize components")

    # init passive learner and every component needed for pl workflow
    sl_model: PassiveLearner = init_helper.get_sl_model()
    cand_info_mapping: Callable[[X, Y, AddInfo_Y], CandInfo] = init_helper.get_mapper_function_prediction_to_candidate_info()
    pl_performance_evaluator: PerformanceEvaluator = init_helper.get_pl_performance_evaluator()
    if pl_performance_evaluator.pl != sl_model:
        system_state.set(int(SystemStates.ERROR))
        raise IncorrectParameters("The pl provided to the pl_controller and the pl of the pl_evaluator need to be the same!")
    # set the passive learner controller
    pl = PassiveLearnerController(pl=sl_model, training_set=training_set, candidate_set=candidate_set, scenario=scenario, cand_info_mapping=cand_info_mapping, pl_evaluator=pl_performance_evaluator)

    # init oracle
    oracle: Oracle = init_helper.get_oracle()
    # set the oracle controller
    o = OracleController(o=oracle, training_set=training_set, query_set=query_set)

    # init the components needed for the al workflow => info_analyser
    info_analyser: InformativenessAnalyser = init_helper.get_informativeness_analyser()
    # set the al controller
    al = ActiveLearnerController(candidate_set=candidate_set, query_set=query_set, info_analyser=info_analyser, scenario=scenario)

    # initial training, data source update
    logging.info("Initial training and first candidate update")
    x_train, y_train, epochs, batch_size = init_helper.get_initial_training_data()
    pl.init_pl(x_train, y_train, batch_size=batch_size, epochs=epochs)  # training with initial training data
    pl.init_candidates()

    # WORKFLOW: Training in parallel processes
    # from here on out, no further case dependent implementation necessary => just in initiation phase

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
