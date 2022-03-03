import logging
import string
from numbers import Number
from typing import Tuple, Sequence, Callable

import mysql.connector
import numpy as np

from al_specific_components.candidate_update.candidate_updater_implementations import Pool, Stream, Generator
from basic_sl_component_interfaces import PassiveLearner
from helpers import X, Y, CandInfo, Scenarios, AddInfo_Y
from helpers.database_helper import DatabaseInfoStore
from helpers.exceptions import NoNewElementException, NoSuchElementException, InvalidTyping
from workflow_management.database_interfaces import TrainingSet, CandidateSet, LogQueryDecisionDB, QuerySet

log = logging.getLogger("Database information")


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
        raise InvalidTyping("Default database implementation assumes the following types: X (input) - array of numbers (e.g., numpy array); Y (output) - array of numbers (e.g., numpy array); CandInfo (additional information about candidate) - tuple of numbers")

    x_sql_definition = ", ".join(["x_" + str(i) + " double" for i in range(len(example_x))])
    y_sql_definition = ", ".join(["y_" + str(i) + " double" for i in range(len(example_y))])
    cand_info_sql_definition = ", ".join(["cand_info_" + str(i) + " double" for i in range(len(example_cand_info))])
    default_database_helper = DatabaseInfoStore(host=host, user=user, password=password, database=database,
                                                input_definition=x_sql_definition, output_definition=y_sql_definition, additional_candidate_information_definition=cand_info_sql_definition)

    training_set = DefaultTrainingSet(default_database_helper)

    if scenario == Scenarios.PbS:
        candidate_set = candidate_source
    else:
        candidate_set = DefaultCandidateSet(default_database_helper)

    log_query_decision_db = DefaultLogQueryDecision(default_database_helper)
    query_set = DefaultQuerySet(default_database_helper)

    return training_set, candidate_set, log_query_decision_db, query_set


class DefaultTrainingSet(TrainingSet):

    def __init__(self, database_info: DatabaseInfoStore):
        self.database_info = database_info

        training_set_name = self.database_info.training_set_name
        input_definition = self.database_info.input_definition
        output_definition = self.database_info.output_definition
        schema_name = self.database_info.database

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DROP TABLE IF EXISTS {training_set_name}"
        cursor.execute(sql)

        sql = f"CREATE TABLE {training_set_name} (id int AUTO_INCREMENT PRIMARY KEY, use_for_training double, {input_definition}, {output_definition})"
        cursor.execute(sql)

        db.close()

        log.info(f"finished initializing the Training set, database table: '{schema_name}.{training_set_name}'")

    def append_labelled_instance(self, x: X, y: Y) -> None:
        training_set_name = self.database_info.training_set_name
        input_reference = self.database_info.create_reference_from_sql_definition(self.database_info.input_definition)
        output_reference = self.database_info.create_reference_from_sql_definition(self.database_info.output_definition)
        input_placeholders = self.database_info.create_placeholders_from_sql_definition(self.database_info.input_definition)
        output_placeholders = self.database_info.create_placeholders_from_sql_definition(self.database_info.output_definition)

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"""INSERT INTO {training_set_name} (use_for_training, {input_reference}, {output_reference}) VALUES (1, {input_placeholders}, {output_placeholders})"""
        val = self.database_info.x_to_str_tuple(x) + self.database_info.y_to_str_tuple(y)

        cursor.execute(sql, val)
        db.commit()

        db.close()

    def retrieve_all_labelled_instances(self) -> Tuple[Sequence[X], Sequence[Y]]:
        training_set_name = self.database_info.training_set_name
        schema_name = self.database_info.database
        x_size = len(self.database_info.input_definition.split(", "))
        y_size = len(self.database_info.output_definition.split(", "))

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT * FROM {training_set_name}")
        res = cursor.fetchall()
        db.close()

        if (len(res) == 0) or (res[0][0] is None):
            raise NoNewElementException(f"{schema_name}.{training_set_name}")

        xs, ys = np.array([]), np.array([])
        for item in res:
            x = np.array(item[2:x_size + 2])
            y = np.array(item[-y_size:])
            if len(xs) == 0:  # and len(ys) == 0
                xs = np.array([x])
                ys = np.array([y])
            else:
                xs = np.append(xs, [x], axis=0)
                ys = np.append(ys, [y], axis=0)

        return xs, ys

    def remove_labelled_instance(self, x: X) -> None:
        training_set_name = self.database_info.training_set_name
        input_equal_check = self.database_info.create_equal_check_from_sql_definition(self.database_info.input_definition)

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DELETE FROM {training_set_name} WHERE {input_equal_check}"
        val = self.database_info.x_to_str_tuple(x)

        cursor.execute(sql, val)
        db.commit()
        db.close()

    def clear(self) -> None:
        training_set_name = self.database_info.training_set_name

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"DELETE FROM {training_set_name}")
        db.commit()
        db.close()

    def retrieve_labelled_training_instance(self) -> Tuple[X, Y]:
        training_set_name = self.database_info.training_set_name
        input_reference = self.database_info.create_reference_from_sql_definition(self.database_info.input_definition)
        output_reference = self.database_info.create_reference_from_sql_definition(self.database_info.output_definition)
        schema_name = self.database_info.database
        x_size = len(self.database_info.input_definition.split(", "))
        y_size = len(self.database_info.output_definition.split(", "))

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT MIN(id), {input_reference}, {output_reference} FROM {training_set_name} WHERE use_for_training = 1")
        result = cursor.fetchall()
        db.close()

        if (len(result) == 0) or (result[0][0] is None):
            raise NoNewElementException(f"{schema_name}.{training_set_name}")

        x = np.array(result[0][1:x_size + 1])
        y = np.array(result[0][-y_size:])

        return x, y

    def retrieve_all_training_instances(self) -> Tuple[Sequence[X], Sequence[Y]]:
        training_set_name = self.database_info.training_set_name
        schema_name = self.database_info.database
        x_size = len(self.database_info.input_definition.split(", "))
        y_size = len(self.database_info.output_definition.split(", "))

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT * FROM {training_set_name} WHERE use_for_training = 1")
        res = cursor.fetchall()
        db.close()

        if (len(res) == 0) or (res[0][0] is None):
            raise NoNewElementException(f"{schema_name}.{training_set_name}")

        xs, ys = np.array([]), np.array([])
        for item in res:
            x = np.array(item[2:x_size + 2])
            y = np.array(item[-y_size:])
            if len(xs) == 0:  # and len(ys) == 0
                xs = np.array([x])
                ys = np.array([y])
            else:
                xs = np.append(xs, [x], axis=0)
                ys = np.append(ys, [y], axis=0)

        return xs, ys

    def set_instance_not_use_for_training(self, x: X) -> None:
        training_set_name = self.database_info.training_set_name
        input_equal_check = self.database_info.create_equal_check_from_sql_definition(self.database_info.input_definition)
        schema_name = self.database_info.database

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"SELECT * FROM {training_set_name} WHERE {input_equal_check}"
        val = self.database_info.x_to_str_tuple(x)

        cursor.execute(sql, val)
        res = cursor.fetchall()

        if (len(res) == 0) or (res[0][0] is None):
            db.close()
            raise NoSuchElementException(f"{schema_name}.{training_set_name}", x)

        sql = f"UPDATE {training_set_name} SET use_for_training = 0 WHERE {input_equal_check}"
        val = self.database_info.x_to_str_tuple(x)

        cursor.execute(sql, val)
        db.commit()
        db.close()


class DefaultCandidateSet(CandidateSet):

    def __init__(self, database_info: DatabaseInfoStore):
        self.database_info = database_info

        candidate_set_name = self.database_info.candidate_set_name
        input_definition = self.database_info.input_definition
        additional_candidate_information_definition = self.database_info.additional_candidate_information_definition
        schema_name = self.database_info.database

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DROP TABLE IF EXISTS {candidate_set_name}"
        cursor.execute(sql)

        sql = f"CREATE TABLE {candidate_set_name} (id int AUTO_INCREMENT PRIMARY KEY, {input_definition}, {additional_candidate_information_definition})"
        cursor.execute(sql)

        db.close()

        log.info(f"finished initializing the Candidate set, database name: '{schema_name}.{candidate_set_name}'")

    def get_first_instance(self) -> Tuple[X, CandInfo]:
        candidate_set_name = self.database_info.candidate_set_name
        input_reference = self.database_info.create_reference_from_sql_definition(self.database_info.input_definition)
        additional_candidate_information_reference = self.database_info.create_reference_from_sql_definition(self.database_info.additional_candidate_information_definition)
        schema_name = self.database_info.database
        x_size = len(self.database_info.input_definition.split(", "))
        cand_info_size = len(self.database_info.additional_candidate_information_definition.split(", "))

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT MIN(id), {input_reference}, {additional_candidate_information_reference} FROM {candidate_set_name}")
        res = cursor.fetchall()
        db.close()

        if (len(res) == 0) or (res[0][0] is None):
            raise NoNewElementException(f"{schema_name}.{candidate_set_name}")

        x = np.array(res[0][1:x_size + 1])
        additional_info = tuple(res[0][-cand_info_size:])
        return x, additional_info

    def get_instance(self, x: X) -> Tuple[X, CandInfo]:
        candidate_set_name = self.database_info.candidate_set_name
        input_reference = self.database_info.create_reference_from_sql_definition(self.database_info.input_definition)
        input_equal_check = self.database_info.create_equal_check_from_sql_definition(self.database_info.input_definition)
        additional_candidate_information_reference = self.database_info.create_reference_from_sql_definition(self.database_info.additional_candidate_information_definition)
        schema_name = self.database_info.database
        x_size = len(self.database_info.input_definition.split(", "))
        cand_info_size = len(self.database_info.additional_candidate_information_definition.split(", "))

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"SELECT {input_reference}, {additional_candidate_information_reference} FROM {candidate_set_name} WHERE {input_equal_check}"
        val = self.database_info.x_to_str_tuple(x)

        cursor.execute(sql, val)
        res = cursor.fetchall()
        db.close()

        if (len(res) == 0) or (res[0][0] is None):
            raise NoSuchElementException(f"{schema_name}.{candidate_set_name}", x)

        x = np.array(res[0][1:x_size + 1])
        additional_info = tuple(res[0][-cand_info_size:])
        return x, additional_info

    def remove_instance(self, x: X) -> None:
        candidate_set_name = self.database_info.candidate_set_name
        input_equal_check = self.database_info.create_equal_check_from_sql_definition(self.database_info.input_definition)

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DELETE FROM {candidate_set_name} WHERE {input_equal_check}"
        val = self.database_info.x_to_str_tuple(x)

        cursor.execute(sql, val)
        db.commit()

        db.close()

    def add_instance(self, x: X, additional_info: CandInfo) -> None:
        candidate_set_name = self.database_info.candidate_set_name
        input_reference = self.database_info.create_reference_from_sql_definition(self.database_info.input_definition)
        input_placeholders = self.database_info.create_placeholders_from_sql_definition(self.database_info.input_definition)
        additional_candidate_information_reference = self.database_info.create_reference_from_sql_definition(self.database_info.additional_candidate_information_definition)
        additional_candidate_information_placeholders = self.database_info.create_placeholders_from_sql_definition(self.database_info.additional_candidate_information_definition)

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"INSERT INTO {candidate_set_name} ({input_reference}, {additional_candidate_information_reference}) VALUES ({input_placeholders}, {additional_candidate_information_placeholders})"
        val = self.database_info.x_to_str_tuple(x) + self.database_info.additional_candidate_information_to_str_tuple(additional_info)

        cursor.execute(sql, val)
        db.commit()

        db.close()

    def is_empty(self) -> bool:
        try:
            self.get_first_instance()
        except NoNewElementException:
            return True
        return False


class DefaultLogQueryDecision(LogQueryDecisionDB):

    def __init__(self, database_info: DatabaseInfoStore):
        self.database_info = database_info

        log_query_decision_set_name = self.database_info.log_query_decision_set_name
        input_definition = self.database_info.input_definition
        log_query_decision_information_definition = self.database_info.log_query_decision_information_definition
        additional_candidate_information_definition = self.database_info.additional_candidate_information_definition
        schema_name = self.database_info.database

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DROP TABLE IF EXISTS {log_query_decision_set_name}"
        cursor.execute(sql)

        sql = f"CREATE TABLE {log_query_decision_set_name} (id int AUTO_INCREMENT PRIMARY KEY, {input_definition}, {log_query_decision_information_definition}, {additional_candidate_information_definition})"
        cursor.execute(sql)

        db.close()

        log.info(f"finished initializing the Training set, database name: '{schema_name}.{log_query_decision_set_name}'")

    def add_instance(self, x: X, info_value: float, queried: bool, additional_info: CandInfo) -> None:
        log_query_decision_set_name = self.database_info.log_query_decision_set_name
        input_reference = self.database_info.create_reference_from_sql_definition(self.database_info.input_definition)
        log_query_decision_information_reference = self.database_info.create_reference_from_sql_definition(self.database_info.log_query_decision_information_definition)
        additional_candidate_information_reference = self.database_info.create_reference_from_sql_definition(self.database_info.additional_candidate_information_definition)
        input_placeholders = self.database_info.create_placeholders_from_sql_definition(self.database_info.input_definition)
        log_query_decision_information_placeholders = self.database_info.create_placeholders_from_sql_definition(self.database_info.log_query_decision_information_definition)
        additional_candidate_information_placeholders = self.database_info.create_placeholders_from_sql_definition(self.database_info.additional_candidate_information_definition)

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"INSERT INTO {log_query_decision_set_name} ({input_reference}, {log_query_decision_information_reference}, {additional_candidate_information_reference}) VALUES ({input_placeholders}, {log_query_decision_information_placeholders}, {additional_candidate_information_placeholders})"""
        val = self.database_info.x_to_str_tuple(x) + self.database_info.log_query_decision_information_to_str_tuple(info_value, queried) + self.database_info.additional_candidate_information_to_str_tuple(additional_info)

        cursor.execute(sql, val)
        db.commit()

        db.close()


class DefaultQuerySet(QuerySet):

    def __init__(self, database_info: DatabaseInfoStore):
        self.database_info = database_info

        query_set_name = self.database_info.query_set_name
        input_definition = self.database_info.input_definition
        schema_name = self.database_info.database

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DROP TABLE IF EXISTS {query_set_name}"
        cursor.execute(sql)

        sql = f"CREATE TABLE {query_set_name} (id int AUTO_INCREMENT PRIMARY KEY, {input_definition})"
        cursor.execute(sql)

        db.close()
        logging.info(f"finished initializing the Query set, database name: '{schema_name}.{query_set_name}'")

    def add_instance(self, x: X) -> None:
        query_set_name = self.database_info.query_set_name
        input_equal_check = self.database_info.create_equal_check_from_sql_definition(self.database_info.input_definition)
        input_reference = self.database_info.create_reference_from_sql_definition(self.database_info.input_definition)
        input_placeholders = self.database_info.create_placeholders_from_sql_definition(self.database_info.input_definition)

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"SELECT id from {query_set_name} WHERE {input_equal_check}"
        val = self.database_info.x_to_str_tuple(x)

        cursor.execute(sql, val)
        res = cursor.fetchall()
        if len(res) == 0:
            sql = f"INSERT INTO {query_set_name} ({input_reference}) VALUES ({input_placeholders})"
            cursor.execute(sql, val)
            db.commit()

        db.close()

    def get_instance(self) -> X:
        query_set_name = self.database_info.query_set_name
        input_reference = self.database_info.create_reference_from_sql_definition(self.database_info.input_definition)
        schema_name = self.database_info.database

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT MIN(id), {input_reference} from {query_set_name}")
        res = cursor.fetchall()
        db.close()

        if (len(res) == 0) or (res[0][0] is None):
            raise NoNewElementException(f"{schema_name}.{query_set_name}")

        x = np.array(res[0][1:])
        return x

    def remove_instance(self, x: X) -> None:
        query_set_name = self.database_info.query_set_name
        input_equal_check = self.database_info.create_equal_check_from_sql_definition(self.database_info.input_definition)

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()
        sql = f"DELETE from {query_set_name} WHERE {input_equal_check}"
        val = self.database_info.x_to_str_tuple(x)

        cursor.execute(sql, val)
        db.commit()
        db.close()
