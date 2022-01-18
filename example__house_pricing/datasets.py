import logging
from typing import Tuple, List

import mysql.connector
import numpy as np
from numpy import ndarray

from al_components.candidate_update.candidate_updater_implementations import Pool
from helpers import X, Y, CandInfo
from helpers.exceptions import NoNewElementException, NoSuchElementException
from workflow_management.database_interfaces import TrainingSet, QuerySet


def connect_to_house_pricing_example_db():
    # noinspection SpellCheckingInspection
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="toor",
        database="house_pricing_example")


def x_to_str_tuple(x):
    return str(x[0]), str(x[1]), str(x[2]), str(x[3]), str(x[4]), str(x[5]), str(x[6]), str(x[7]), str(x[8]), str(x[9]), str(x[10]), str(x[11]), str(x[12])


# TODO: ensure unique identification by input
input_definition = "ZERO double, ONE double, TWO double, THREE double, FOUR double, FIVE double, SIX double, SEVEN double, EIGHT double, NINE double, TEN double, ELEVEN double, TWELVE double"
input_reference = "ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, ELEVEN, TWELVE"
input_placeholders = "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s"
input_equal_check = "ZERO = %s AND ONE = %s AND TWO = %s AND THREE = %s AND FOUR = %s AND FIVE = %s AND SIX = %s AND SEVEN = %s AND EIGHT = %s AND NINE = %s AND TEN = %s AND ELEVEN = %s AND TWELVE = %s"


def y_to_str_tuple(y):
    return (str(y[0]),)


output_definition = "PRICE double"
output_reference = "PRICE"
output_placeholders = "%s"
output_equal_check = "PRICE = %s"


def additional_candidate_information_to_str_tuple(additional_candidate_information):
    return str(additional_candidate_information[0]), str(additional_candidate_information[1])


# TODO: how should uncertainty look?????
additional_candidate_information_definition = "predicted_PRICE double, uncertainty double"
additional_candidate_information_reference = "predicted_PRICE, uncertainty"
additional_candidate_information_placeholders = "%s, %s"
additional_candidate_information_equal_check = "predicted_PRICE = %s AND uncertainty = %s"
additional_candidate_information_set = "predicted_PRICE = %s, uncertainty = %s"

schema_name = "house_pricing_example"
candidate_set_name = "predicted_set"
training_set_name = "labelled_set"
query_set_name = "unlabelled_set"


class TrainingSetHouses(TrainingSet):

    def __init__(self):
        logging.info("start initializing the Training set")

        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DROP TABLE IF EXISTS {training_set_name}"
        cursor.execute(sql)

        sql = f"""CREATE TABLE {training_set_name} (
                        id int AUTO_INCREMENT PRIMARY KEY,
                        {input_definition}, {output_definition}
                    )"""
        cursor.execute(sql)

        db.close()

        logging.info(f"finished initializing the Training set, database name: '{schema_name}.{training_set_name}'")

    def append_labelled_instance(self, x: X, y: Y) -> None:
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"""INSERT INTO {training_set_name} (
                                        {input_reference}, {output_reference}
                                     ) VALUES (
                                        {input_placeholders}, {output_placeholders}
                                     )"""
        val = x_to_str_tuple(x) + y_to_str_tuple(y)

        cursor.execute(sql, val)
        db.commit()

        db.close()

    def retrieve_labelled_instance(self) -> Tuple[X, Y]:
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT MIN(id), {input_reference}, {output_reference} FROM {training_set_name}")
        result = cursor.fetchall()

        if (len(result) == 0) or (result[0][0] is None):
            db.close()
            raise NoNewElementException(f"{schema_name}.{training_set_name}")

        x = np.array(result[0][1:-1])
        y = result[0][-1]

        return x, y

    def retrieve_all_labelled_instances(self) -> Tuple[List[X] or ndarray, List[Y] or ndarray]:
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT * FROM {training_set_name}")
        res = cursor.fetchall()

        db.close()

        if (len(res) == 0) or (res[0][0] is None):
            raise NoNewElementException(f"{schema_name}.{training_set_name}")

        xs, ys = None, None
        for item in res:
            if len(xs) == 0:  # and len(ys) == 0
                xs = np.array([np.array(item[1:-1])])
                ys = np.array(np.array(item[-1]))
            else:
                xs = np.append(xs, [np.array(item[1:-1])], axis=0)
                ys = np.append(ys, np.array(item[-1]))

        return xs, ys

    def remove_labelled_instance(self, x: X) -> None:
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DELETE FROM {training_set_name} WHERE {input_equal_check}"
        val = x_to_str_tuple(x)

        cursor.execute(sql, val)
        db.commit()
        db.close()

    def clear(self) -> None:
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"DELETE FROM {training_set_name}")
        db.commit()
        db.close()


class CandidateSetHouses(Pool):

    def __init__(self):
        logging.info("start initializing the Candidate set")

        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DROP TABLE IF EXISTS {candidate_set_name}"
        cursor.execute(sql)

        sql = f"""CREATE TABLE {candidate_set_name} (
                        id int AUTO_INCREMENT PRIMARY KEY,
                        {input_definition}, {additional_candidate_information_definition}
                  )"""
        cursor.execute(sql)

        db.close()

        logging.info(f"finished initializing the Candidate set, database name: '{schema_name}.{candidate_set_name}'")

    def initiate_pool(self, x_initial: List[X] or ndarray) -> None:
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"INSERT INTO {candidate_set_name} ({input_reference}) VALUES ({input_placeholders})"
        val = []
        for i in range(len(x_initial)):
            val.append(x_to_str_tuple(x_initial[i]))

        cursor.executemany(sql, val)
        db.commit()

        db.close()

    def get_first_instance(self) -> Tuple[X, CandInfo]:
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"""SELECT 
                                MIN(id),
                                {input_reference}, {additional_candidate_information_reference}
                           FROM {candidate_set_name}""")
        res = cursor.fetchall()
        db.close()

        if (len(res) == 0) or (res[0][0] is None):
            raise NoNewElementException(f"{schema_name}.{candidate_set_name}")

        x = np.array(res[0][1:-2])
        additional_info = (res[0][-2], res[0][-1])
        return x, additional_info

    def get_instance(self, x: X) -> Tuple[X, CandInfo]:
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"""SELECT {input_reference}, {additional_candidate_information_reference}
                  FROM {candidate_set_name} 
                  WHERE {input_equal_check}"""
        val = x_to_str_tuple(x)

        cursor.execute(sql, val)
        res = cursor.fetchall()

        if (len(res) == 0) or (res[0][0] is None):
            db.close()
            raise NoSuchElementException(f"{schema_name}.{candidate_set_name}", x)

        db.close()

        x = np.array(res[0][1:-2])
        addition_info = (res[0][-2], res[0][-1])
        return x, addition_info

    def remove_instance(self, x: X) -> None:
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DELETE FROM {candidate_set_name} WHERE {input_equal_check}"
        val = x_to_str_tuple(x)

        cursor.execute(sql, val)
        db.commit()

        db.close()

    def add_instance(self, x: X, additional_info: CandInfo = None) -> None:
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"""INSERT INTO {candidate_set_name} (
                                            {input_reference}, {additional_candidate_information_reference} 
                                         ) VALUES (
                                            {input_placeholders}, {additional_candidate_information_placeholders}
                                         )"""
        val = x_to_str_tuple(x) + additional_candidate_information_to_str_tuple(additional_info)

        cursor.execute(sql, val)
        db.commit()

        db.close()

    def is_empty(self) -> bool:
        try:
            self.retrieve_all_instances()
        except NoNewElementException:
            return True
        return False

    def update_instances(self, xs: List[X] or ndarray, new_additional_infos: List[CandInfo] or ndarray = None) -> None:
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"UPDATE {candidate_set_name} SET {additional_candidate_information_set} WHERE {input_equal_check}"
        val = []
        for i in range(len(xs)):
            val.append(additional_candidate_information_to_str_tuple(new_additional_infos[i]) + x_to_str_tuple(xs[i]))

        cursor.executemany(sql, val)
        db.commit()

        db.close()

    def retrieve_all_instances(self) -> Tuple[List[X] or ndarray, List[CandInfo] or ndarray]:
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT {input_reference}, {additional_candidate_information_reference} FROM {candidate_set_name}")
        res = cursor.fetchall()

        if (len(res) == 0) or (res[0][0] is None):
            db.close()
            raise NoNewElementException(f"{schema_name}.{candidate_set_name}")

        db.close()

        xs, add_infos = np.array([]), None
        for item in res:
            if len(xs) == 0:
                xs = np.array([np.array(item[0:-2])])
                add_infos = np.array((item[-2], item[-1]))
            else:
                xs = np.append(xs, [np.array(item[0:-2])], axis=0)
                add_infos = np.append((item[-2], item[-1]))

        return xs, add_infos


class QuerySetHouses(QuerySet):

    def __init__(self):
        logging.info("start initializing the Query set")

        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DROP TABLE IF EXISTS {query_set_name}"
        cursor.execute(sql)

        # TODO: ensure unique identification by input
        sql = f"CREATE TABLE {query_set_name} (id int AUTO_INCREMENT PRIMARY KEY, {input_definition})"
        cursor.execute(sql)

        db.close()
        logging.info(f"finished initializing the Query set, database name: '{schema_name}.{query_set_name}'")

    def add_instance(self, x: X) -> None:
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"SELECT id from {query_set_name} WHERE {input_equal_check}"
        val = x_to_str_tuple(x)

        cursor.execute(sql, val)
        res = cursor.fetchall()
        if len(res) == 0:
            sql = f"INSERT INTO {query_set_name} ({input_reference}) VALUES ({input_placeholders})"

            cursor.execute(sql, val)
            db.commit()

        db.close()

    def get_instance(self) -> X:
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT MIN(id), {input_reference} from {query_set_name}")
        res = cursor.fetchall()
        db.close()

        if (len(res) == 0) or (res[0][0] is None):
            raise NoNewElementException(f"{schema_name}.{query_set_name}")

        x = np.array(res[0][1:])
        return x

    def remove_instance(self, x: X) -> None:
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()
        sql = f"DELETE from {query_set_name} WHERE {input_equal_check}"
        val = x_to_str_tuple(x)

        cursor.execute(sql, val)
        db.commit()

        db.close()
