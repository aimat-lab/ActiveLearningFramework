

# noinspection PyUnusedLocal
from typing import Tuple, List, Optional, Callable

from nparray import ndarray

from additional_component_interfaces import PassiveLearner, Oracle
from al_components.candidate_update.candidate_updater_implementations import Pool, Stream, Generator
from al_components.perfomance_evaluation import PerformanceEvaluator
from al_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import Scenarios, X, Y, AddInfo_Y, CandInfo
from workflow_management.database_interfaces import CandidateSet, QuerySet, TrainingSet, StoredLabelledSetDB, LogQueryDecisionDB


class InitiationHelper:

    def get_scenario(self) -> Scenarios:
        """
        Determines the scenario for the whole AL process

        :return: the selected scenario
        """
        # case implementation: set scenario
        raise NotImplementedError

    def get_candidate_source(self) -> Pool or Stream or Generator:
        """
        Determines the source for the candidates => concrete type is scenario dependent (see get_candidate_source_type)

        :return: the candidate source
        """
        # case implementation: implement concrete candidate source => initialize accordingly
        raise NotImplementedError

    def get_datasets(self) -> Tuple[TrainingSet, StoredLabelledSetDB, CandidateSet, LogQueryDecisionDB, QuerySet]:
        """
        Returns all implemented datasets

        - if PbS: candidate set must match candidate source

        :return: the datasets needed for the workflow and for additional information storage
        """
        # case implementation: implement concrete datasets
        raise NotImplementedError

    def get_sl_model(self) -> PassiveLearner:
        """
        Returns the extended sl/pl model (need to implement the PassiveLearner interface)

        :return: the model
        """
        # case implementation: implement concrete sl model (passive learner)
        raise NotImplementedError

    def get_initial_training_data(self) -> Tuple[List[X] or ndarray, List[Y] or ndarray, Optional[int], Optional[int]]:
        """
        Returns labelled data for the initial training episode of the pl

        can return additional properties for this training:
            - epochs
            - batch_size

        :return: input data, correct output, optional epochs, optional batch_size
        """
        # case implementation: set the initial training data for the sl model
        raise NotImplementedError

    # noinspection PyUnreachableCode,PyTypeChecker
    def get_mapper_function_prediction_to_candidate_info(self) -> Callable[[X, Y, AddInfo_Y], CandInfo]:
        """
        Return a function => see *al_components.candidate_update.get_candidate_additional_information* for more information

        :return: the mapper function
        """
        raise NotImplementedError
        function = get_candidate_additional_information  # case implementation: implement concrete candidate information creation function

    def get_pl_performance_evaluator(self) -> PerformanceEvaluator:
        """
        Return a performance evaluator (evaluate performance of pl form perspective of the AL project => performance of AL model = performance of PL model)

        :return: the evaluator
        """
        # case implementation: implement concrete sl performance evaluator
        raise NotImplementedError

    def get_oracle(self) -> Oracle:
        """
        Return the instance correctly labelling the unlabelled data then to be added to the training data

        :return: the oracle instance
        """
        # case implementation: implement concrete oracle (with knowledge about ground truth)
        raise NotImplementedError

    def get_informativeness_analyser(self) -> InformativenessAnalyser:
        """
        Return the instance evaluating the performance of a candidate

        :return: the informativeness analyser
        """
        # case implementation: implement concrete informativeness analyser => foundation for query selection
        raise NotImplementedError
