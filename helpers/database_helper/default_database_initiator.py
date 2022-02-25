import string
from numbers import Number
from typing import Callable, Tuple

import mysql.connector

from basic_sl_component_interfaces import PassiveLearner
from al_specific_components.candidate_update.candidate_updater_implementations import Generator, Stream, Pool
from helpers import Scenarios, X, Y, AddInfo_Y, CandInfo
from helpers.database_helper.database_info_store import DefaultDatabaseHelper
from helpers.database_helper.default_datasets import DefaultTrainingSet, DefaultCandidateSet, DefaultLogQueryDecision, DefaultQuerySet
from helpers.exceptions import InvalidTyping
from workflow_management.database_interfaces import TrainingSet, CandidateSet, LogQueryDecisionDB, QuerySet


def connect_to_house_pricing_example_db(host: string, user: string, password: string, database: string):
    return mysql.connector.connect(host=host, user=user, password=password, database=database)


def get_default_databases(
        scenario: Scenarios,
        candidate_source: Pool or Stream or Generator,
        pl: PassiveLearner,
        mapper_function_prediction_to_candidate_info: Callable[[X, Y, AddInfo_Y], CandInfo],
        host: string, user: string, password: string, database: string
) -> Tuple[TrainingSet, CandidateSet, LogQueryDecisionDB, QuerySet]:
    # TODO: documentation
    """
    The default databases assume, that the input, additional information about the prediction and the candidate information are Sequences of numbers and the output is a single number

    :param scenario:
    :param candidate_source:
    :param pl:
    :param mapper_function_prediction_to_candidate_info:
    :param host:
    :param user:
    :param password:
    :param database:
    :return:
    """
    training_set: TrainingSet
    candidate_set: CandidateSet
    log_query_decision_db: LogQueryDecisionDB
    query_set: QuerySet

    example_x: X
    if scenario == Scenarios.PbS:
        assert isinstance(candidate_source, Pool)
        example_x, _ = candidate_source.get_first_instance()
    elif scenario == Scenarios.SbS:
        example_x = candidate_source.get_element()
    else:  # scenario == Scenarios.MQS
        example_x = candidate_source.generate_instance()

    example_y: Y
    example_add_info_y: AddInfo_Y
    example_y, example_add_info_y = pl.predict(example_x)

    example_cand_info: CandInfo
    example_cand_info = mapper_function_prediction_to_candidate_info(example_x, example_y, example_add_info_y)

    try:
        x_part = example_x[0]
        assert isinstance(x_part, Number)
        y_part = example_y[0]
        assert isinstance(y_part, Number)
        assert isinstance(example_cand_info, Tuple)
        assert all([isinstance(example_cand_info[i], Number) for i in range(len(example_cand_info))])
    except Exception:
        raise InvalidTyping("Default database implementation assumes the following types: X (input) - array of numbers (e.g., numpy array); Y (output) - single number; CandInfo (additional information about candidate) - tuple of numbers")

    x_sql_definition = ", ".join(["x_" + str(i) + " double" for i in range(len(example_x))])
    y_sql_definition = ", ".join(["y_" + str(i) + " double" for i in range(len(example_y))])
    cand_info_sql_definition = ", ".join(["cand_info_" + str(i) + " double" for i in range(len(example_cand_info))])
    default_database_helper = DefaultDatabaseHelper(host=host, user=user, password=password, database=database,
                                                    input_definition=x_sql_definition, output_definition=y_sql_definition, additional_candidate_information_definition=cand_info_sql_definition)

    training_set = DefaultTrainingSet(default_database_helper)

    if scenario == Scenarios.PbS:
        candidate_set = candidate_source
    else:
        candidate_set = DefaultCandidateSet(default_database_helper)

    log_query_decision_db = DefaultLogQueryDecision(default_database_helper)
    query_set = DefaultQuerySet(default_database_helper)

    return training_set, candidate_set, log_query_decision_db, query_set
