from typing import Tuple, Optional, Callable, Sequence

from basic_sl_component_interfaces import PassiveLearner, Oracle, ReadOnlyPassiveLearner
from al_specific_components.candidate_update.candidate_updater_implementations import Pool, Stream, Generator
from al_specific_components.query_selection.informativeness_analyser import InformativenessAnalyser
from helpers import Scenarios, X, Y, AddInfo_Y, CandInfo
from helpers.database_helper.default_database_initiator import get_default_databases
from workflow_management.database_interfaces import CandidateSet, QuerySet, TrainingSet, LogQueryDecisionDB


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

    def get_datasets(self) -> Tuple[TrainingSet, CandidateSet, LogQueryDecisionDB, QuerySet]:
        """
        Returns all implemented datasets => can use the default implementation

        - if PbS: candidate set must match candidate source

        :return: the datasets needed for the workflow and for additional information storage
        """
        # case implementation: implement concrete datasets or use default datasets and only insert host, ...
        # IMPORTANT: first call of function if using the default dataset after the initialization of scenario, candidate source, passive learner and mapper function (prediction to candidate info)

        # access to database => MySQL
        host, user, password, database = None, None, None, None
        if (host is None) or (user is None) or (password is None) or (database is None):
            raise NotImplementedError

        return get_default_databases(self.get_scenario(), self.get_candidate_source(), self.get_pl(), self.get_mapper_function_prediction_to_candidate_info(), host, user, password, database)

    def get_pl(self) -> PassiveLearner:
        """
        Returns the extended sl/pl model (need to implement the PassiveLearner interface)

        should be an extension to the read only view (ensure the objects are separate object, otherwise conflicts can arise due to parallelization)

        :return: the model
        """
        # case implementation: implement concrete sl model (passive learner)
        raise NotImplementedError

    def get_ro_pl(self) -> ReadOnlyPassiveLearner:
        """
        Return the read only version/view of the sl model

        :return: the RO passive learner
        """
        # case implementation: implement concrete read only sl model
        raise NotImplementedError

    def get_initial_training_data(self) -> Tuple[Sequence[X], Sequence[Y], Optional[int], Optional[int]]:
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
        Return a function => see *al_specific_components.candidate_update.get_candidate_additional_information* for more information

        :return: the mapper function
        """
        raise NotImplementedError
        # noinspection PyUnusedLocal
        function = get_candidate_additional_information  # case implementation: implement concrete candidate information creation function

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
