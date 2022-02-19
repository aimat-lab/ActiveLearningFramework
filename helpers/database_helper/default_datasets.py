import logging
from typing import Tuple, Sequence

import numpy as np

from helpers import X, Y, CandInfo
from helpers.database_helper.database_info_store import DefaultDatabaseHelper
from helpers.exceptions import NoNewElementException, NoSuchElementException
from workflow_management.database_interfaces import TrainingSet, CandidateSet, LogQueryDecisionDB, QuerySet

log = logging.getLogger("Database information")


class DefaultTrainingSet(TrainingSet):

    def __init__(self, database_info: DefaultDatabaseHelper):
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
            y = np.array(item[-1])
            if len(xs) == 0:  # and len(ys) == 0
                xs = np.array([x])
                ys = np.array(y)
            else:
                xs = np.append(xs, [x], axis=0)
                ys = np.append(ys, y)

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

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT MIN(id), {input_reference}, {output_reference} FROM {training_set_name} WHERE use_for_training = 1")
        result = cursor.fetchall()
        db.close()

        if (len(result) == 0) or (result[0][0] is None):
            raise NoNewElementException(f"{schema_name}.{training_set_name}")

        x = np.array(result[0][1:x_size + 1])
        y = result[0][-1]

        return x, y

    def retrieve_all_training_instances(self) -> Tuple[Sequence[X], Sequence[Y]]:
        training_set_name = self.database_info.training_set_name
        schema_name = self.database_info.database
        x_size = len(self.database_info.input_definition.split(", "))

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
            y = np.array(item[-1])
            if len(xs) == 0:  # and len(ys) == 0
                xs = np.array([x])
                ys = np.array(y)
            else:
                xs = np.append(xs, [x], axis=0)
                ys = np.append(ys, y)

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

    def __init__(self, database_info: DefaultDatabaseHelper):
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

    def __init__(self, database_info: DefaultDatabaseHelper):
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

    def add_instance(self, x: X, info_value: float, queried: bool, additional_info: CandInfo = None) -> None:
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

    def __init__(self, database_info: DefaultDatabaseHelper):
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
