from typing import Tuple, Sequence, Optional, Callable

import numpy as np

from al_specific_components.candidate_update.candidate_updater_implementations import Pool, Stream, Generator
from al_specific_components.query_selection.informativeness_analyser import InformativenessAnalyser
from basic_sl_component_interfaces import Oracle, ReadOnlyPassiveLearner, PassiveLearner
from example_implementations.al_specific_component_implementations import EverythingIsInformativeAnalyser, ButenePool
from example_implementations.basic_sl_component_implementations import ButenePassiveLearner, ButeneOracle
from helpers import X, Y, Scenarios, AddInfo_Y, CandInfo
from helpers.database_helper.default_database_initiator import get_default_databases
from helpers.database_helper.default_datasets import DefaultTrainingSet
from helpers.system_initiator import InitiationHelper
from workflow_management.database_interfaces import TrainingSet, CandidateSet, LogQueryDecisionDB, QuerySet


# noinspection PyUnusedLocal
def get_candidate_additional_information(x: X, prediction: Y, additional_prediction_info: AddInfo_Y) -> CandInfo:
    return tuple(prediction)


class ButeneEnergyForceInitiator(InitiationHelper):

    def __init__(self):
        self.scenario = Scenarios.PbS

        x = np.array([instance.flatten() for instance in np.load("example_implementations/butene_data/butene_x.npy")])
        eng = np.load("example_implementations/butene_data/butene_energy.npy")
        grads = np.load("example_implementations/butene_data/butene_force.npy")
        y = np.array([np.append(eng[i].flatten(), grads[i].flatten()) for i in range(len(eng))])
        host, user, password, database = "localhost", "root", "toor", "butene_energy_force"

        self.x_train_init, self.y_train_init, x, y = x[:20], y[:20], x[20:], y[20:]

        self.pl: PassiveLearner = ButenePassiveLearner()
        self.ro_pl: ReadOnlyPassiveLearner = ButenePassiveLearner()

        self.mapper_function_prediction_to_candidate_info = get_candidate_additional_information
        example_x = x[0]
        example_y, example_add_info = self.pl.predict(example_x)
        example_cand_info = self.mapper_function_prediction_to_candidate_info(example_x, example_y, example_add_info)

        self.candidate_set = ButenePool(host, user, password, database, example_x, example_cand_info)
        self.candidate_set.initiate_pool(x)

        self.info_analyser = EverythingIsInformativeAnalyser()

        self.training_set, self.candidate_set, self.log_qd_db, self.query_set = get_default_databases(self.scenario, self.candidate_set, self.pl, self.mapper_function_prediction_to_candidate_info, host, user, password, database)

        assert isinstance(self.training_set, DefaultTrainingSet)
        self.oracle: Oracle = ButeneOracle(host, user, password, database, self.training_set.database_info.input_definition, self.training_set.database_info.output_definition, xs=x, ys=y)

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

    def get_initial_training_data(self) -> Tuple[Sequence[X], Sequence[Y], Optional[int], Optional[int]]:
        return self.x_train_init, self.y_train_init, 0, 0

    def get_oracle(self) -> Oracle:
        return self.oracle

    def get_informativeness_analyser(self) -> InformativenessAnalyser:
        return self.info_analyser