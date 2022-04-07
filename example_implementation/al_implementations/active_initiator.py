import string
from numbers import Number
from typing import Tuple, Sequence, Callable

from al_specific_components.candidate_update.candidate_updater_implementations import Pool, Stream, Generator
from al_specific_components.query_selection import InformativenessAnalyser
from basic_sl_component_interfaces import Oracle, ReadOnlyPassiveLearner, PassiveLearner
from example_implementation.al_implementations.info_analyser import EverythingIsInformativeAnalyser, UncertaintyInfoAnalyser
from example_implementation.al_implementations.oracle import HousingOracle
from example_implementation.al_implementations.pl import HousingPL
from example_implementation.al_implementations.stream import HousingStream
from example_implementation.helpers import properties
from helpers import X, Y, Scenarios, AddInfo_Y, CandInfo
from helpers.database_helper import DatabaseInfoStore
from helpers.database_helper.default_datasets import DefaultCandidateSet, DefaultTrainingSet, DefaultLogQueryDecision, DefaultQuerySet, get_default_databases
from helpers.exceptions import InvalidTyping
from helpers.system_initiator import InitiationHelper
from workflow_management.database_interfaces import TrainingSet, CandidateSet, LogQueryDecisionDB, QuerySet


# noinspection PyUnusedLocalß
def get_candidate_additional_information(x: X, pred: Y, add_info: AddInfo_Y):
    return tuple(add_info)


class HousingInitiator(InitiationHelper):

    def __init__(self, x, x_test, y, y_test, entity=properties.eval_entities["ia"]):
        self._scenario = Scenarios.SbS
        self._entity = entity

        host, user, password, database = "localhost", "root", "toor", properties.RUN_NUMBER + "__housing__" + entity

        initial_training_data_size = properties.al_training_params["initial_batch_size"]
        self._x_train_init, x = x[:initial_training_data_size], x[initial_training_data_size:]
        self._y_train_init, y = y[:initial_training_data_size], y[initial_training_data_size:]

        self._pl: PassiveLearner = HousingPL(x_test=x_test, y_test=y_test, entity=entity)
        self._ro_pl: ReadOnlyPassiveLearner = HousingPL(x_test=x_test, y_test=y_test, entity=entity)

        self._stream = HousingStream(x)

        self._mapper_function_prediction_to_candidate_info = get_candidate_additional_information

        self._training_set, self._candidate_set, self._log_qd_da, self._query_set = get_default_databases(self._scenario, self._stream, self._pl, self._mapper_function_prediction_to_candidate_info, host, user, password, database)

        if entity == properties.eval_entities["ua"]:
            self._info_analyser = EverythingIsInformativeAnalyser()
        else:  # entity == properties.eval_entities["ia"]
            self._info_analyser = UncertaintyInfoAnalyser(candidate_set=self._candidate_set)

        self._oracle = HousingOracle(host, user, password, database, self._training_set.database_info.input_definition, self._training_set.database_info.output_definition, xs=x, ys=y)

    def get_scenario(self) -> Scenarios:
        return self._scenario

    def get_candidate_source(self) -> Pool or Stream or Generator:
        return self._stream

    def get_pl(self) -> PassiveLearner:
        return self._pl

    def get_ro_pl(self) -> ReadOnlyPassiveLearner:
        return self._ro_pl

    def get_initial_training_data(self) -> Tuple[Sequence[X], Sequence[Y]]:
        return self._x_train_init, self._y_train_init

    def get_oracle(self) -> Oracle:
        return self._oracle

    def get_informativeness_analyser(self) -> InformativenessAnalyser:
        return self._info_analyser

    def get_mapper_function_prediction_to_candidate_info(self) -> Callable[[X, Y, AddInfo_Y], CandInfo]:
        return self._mapper_function_prediction_to_candidate_info

    def get_datasets(self) -> Tuple[TrainingSet, CandidateSet, LogQueryDecisionDB, QuerySet]:
        return self._training_set, self._candidate_set, self._log_qd_da, self._query_set

