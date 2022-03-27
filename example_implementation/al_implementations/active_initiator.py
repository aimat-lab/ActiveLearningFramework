import math
from typing import Tuple, Sequence, Callable

import numpy as np

from al_specific_components.candidate_update.candidate_updater_implementations import Pool, Stream, Generator
from al_specific_components.query_selection import InformativenessAnalyser
from basic_sl_component_interfaces import Oracle, ReadOnlyPassiveLearner, PassiveLearner
from helpers import X, Y, Scenarios, AddInfo_Y, CandInfo
from helpers.database_helper.default_datasets import get_default_databases, DefaultTrainingSet
from helpers.system_initiator import InitiationHelper
from example_implementation.al_implementations.butene_pool import ButenePool
from example_implementation.al_implementations.informativeness_analyser import UncertaintyInfoAnalyser, EverythingIsInformativeAnalyser
from example_implementation.al_implementations.oracle import ButeneOracle
from example_implementation.al_implementations.passive_learner import ButenePassiveLearner
from example_implementation.helpers import properties
from workflow_management.database_interfaces import TrainingSet, CandidateSet, LogQueryDecisionDB, QuerySet


# noinspection PyUnusedLocal
def get_candidate_additional_information(x: X, prediction: Y, additional_prediction_info: AddInfo_Y) -> CandInfo:
    if any([math.isnan(elem) for elem in prediction]) or any([math.isnan(elem) for elem in additional_prediction_info]):
        raise Exception
    uncertainty = np.mean(np.var(additional_prediction_info[:2])) * 1 + np.mean(np.var(additional_prediction_info[2:])) * 5
    return uncertainty,


class ButeneInitiator(InitiationHelper):

    def __init__(self, x, x_test, y, y_test, entity=properties.entities["ia"]):
        self._scenario = Scenarios.PbS
        self._entity = entity

        host, user, password, database = "localhost", "root", "toor", properties.RUN_NUMBER + "__butene_energy_force"
        if entity == properties.entities["ua"]:
            database = properties.RUN_NUMBER + "__butene_energy_force__ua"

        initial_data_size = properties.al_training_params["initial_set_size"]
        self._x_train_init, x = x[:initial_data_size], x[initial_data_size:]
        self._y_train_init, y = y[:initial_data_size], y[initial_data_size:]

        self._pl: PassiveLearner = ButenePassiveLearner(x_test, y_test, eval_entity=entity)
        self._ro_pl: ReadOnlyPassiveLearner = ButenePassiveLearner(x_test, y_test, eval_entity=entity)

        self._mapper_function_prediction_to_candidate_info = get_candidate_additional_information

        example_x = x[0]
        example_y, example_add_info = self._pl.predict(example_x)
        example_cand_info = self._mapper_function_prediction_to_candidate_info(example_x, example_y, example_add_info)

        self._candidate_set = ButenePool(host, user, password, database, example_x, example_cand_info)
        self._candidate_set.initiate_pool(x)

        self._training_set, self._candidate_set, self._log_qd_db, self._query_set = get_default_databases(self._scenario, self._candidate_set, self._pl, self._mapper_function_prediction_to_candidate_info, host, user, password, database)

        if entity == properties.entities["ia"]:
            self._info_analyser = UncertaintyInfoAnalyser(candidate_set=self._candidate_set)
        else:
            self._info_analyser = EverythingIsInformativeAnalyser()

        assert isinstance(self._training_set, DefaultTrainingSet)
        self._oracle: Oracle = ButeneOracle(host, user, password, database, self._training_set.database_info.input_definition, self._training_set.database_info.output_definition, xs=x, ys=y)

    def get_scenario(self) -> Scenarios:
        return self._scenario

    def get_candidate_source(self) -> Pool or Stream or Generator:
        return self._candidate_set

    def get_datasets(self) -> Tuple[TrainingSet, CandidateSet, LogQueryDecisionDB, QuerySet]:
        return self._training_set, self._candidate_set, self._log_qd_db, self._query_set

    def get_pl(self) -> PassiveLearner:
        return self._pl

    def get_ro_pl(self) -> ReadOnlyPassiveLearner:
        return self._ro_pl

    def get_initial_training_data(self) -> Tuple[Sequence[X], Sequence[Y]]:
        return self._x_train_init, self._y_train_init

    def get_mapper_function_prediction_to_candidate_info(self) -> Callable[[X, Y, AddInfo_Y], CandInfo]:
        return self._mapper_function_prediction_to_candidate_info

    def get_oracle(self) -> Oracle:
        return self._oracle

    def get_informativeness_analyser(self) -> InformativenessAnalyser:
        return self._info_analyser
