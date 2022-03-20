import math
from typing import Tuple, Sequence, Callable

import numpy as np

from al_specific_components.candidate_update.candidate_updater_implementations import Pool, Stream, Generator
from al_specific_components.query_selection import InformativenessAnalyser
from basic_sl_component_interfaces import Oracle, ReadOnlyPassiveLearner, PassiveLearner
from example_implementations.al_specific_component_implementations import ButenePool, EverythingIsInformativeAnalyser
from example_implementations.basic_sl_component_implementations import ButeneOracle
from example_implementations.basic_sl_component_implementations.ua_passive_learner import UAButenePassiveLearner
from example_implementations.helpers import properties
from example_implementations.helpers.mapper import map_shape_input_to_flat, map_shape_output_to_flat
from helpers import X, Y, Scenarios, AddInfo_Y, CandInfo
from helpers.database_helper.default_datasets import get_default_databases, DefaultTrainingSet
from helpers.system_initiator import InitiationHelper
from workflow_management.database_interfaces import TrainingSet, CandidateSet, LogQueryDecisionDB, QuerySet


# noinspection PyUnusedLocal
def get_candidate_additional_information(x: X, prediction: Y, additional_prediction_info: AddInfo_Y) -> CandInfo:
    if any([math.isnan(elem) for elem in prediction]) or any([math.isnan(elem) for elem in additional_prediction_info]):
        raise Exception
    uncertainty = np.mean(np.var(additional_prediction_info[:2])) * 1 + np.mean(np.var(additional_prediction_info[2:])) * 5
    return uncertainty,


class ButeneEnergyForceInitiator(InitiationHelper):

    def __init__(self, x, x_test, y, y_test):
        self.scenario = Scenarios.PbS

        x_flat = map_shape_input_to_flat(x)
        x_test_flat = map_shape_input_to_flat(x_test)

        y_flat = map_shape_output_to_flat(y)
        y_test_flat = map_shape_output_to_flat(y_test)

        host, user, password, database = "localhost", "root", "toor", properties.RUN_NUMBER + "__butene_energy_force__ua"

        initial_data_size = 4
        self.x_train_init, self.y_train_init, x_flat, y = x_flat[:initial_data_size], y_flat[:initial_data_size], x_flat[initial_data_size:], y_flat[initial_data_size:]

        self.pl: PassiveLearner = UAButenePassiveLearner(x_test_flat, y_test_flat)
        self.ro_pl: ReadOnlyPassiveLearner = UAButenePassiveLearner(x_test_flat, y_test_flat)

        self.mapper_function_prediction_to_candidate_info = get_candidate_additional_information
        example_x = x_flat[0]
        example_y, example_add_info = self.pl.predict(example_x)
        example_cand_info = self.mapper_function_prediction_to_candidate_info(example_x, example_y, example_add_info)

        self.candidate_set = ButenePool(host, user, password, database, example_x, example_cand_info)
        self.candidate_set.initiate_pool(x_flat)

        self.training_set, self.candidate_set, self.log_qd_db, self.query_set = get_default_databases(self.scenario, self.candidate_set, self.pl, self.mapper_function_prediction_to_candidate_info, host, user, password, database)

        self.info_analyser = EverythingIsInformativeAnalyser()

        assert isinstance(self.training_set, DefaultTrainingSet)
        self.oracle: Oracle = ButeneOracle(host, user, password, database, self.training_set.database_info.input_definition, self.training_set.database_info.output_definition, xs=x_flat, ys=y_flat)

    def get_scenario(self) -> Scenarios:
        return self.scenario

    def get_candidate_source(self) -> Pool or Stream or Generator:
        return self.candidate_set

    def get_datasets(self) -> Tuple[TrainingSet, CandidateSet, LogQueryDecisionDB, QuerySet]:
        return self.training_set, self.candidate_set, self.log_qd_db, self.query_set

    def get_mapper_function_prediction_to_candidate_info(self) -> Callable[[X, Y, AddInfo_Y], CandInfo]:
        return self.mapper_function_prediction_to_candidate_info

    def get_pl(self) -> PassiveLearner:
        return self.pl

    def get_ro_pl(self) -> ReadOnlyPassiveLearner:
        return self.ro_pl

    def get_initial_training_data(self) -> Tuple[Sequence[X], Sequence[Y]]:
        return self.x_train_init, self.y_train_init

    def get_oracle(self) -> Oracle:
        return self.oracle

    def get_informativeness_analyser(self) -> InformativenessAnalyser:
        return self.info_analyser
