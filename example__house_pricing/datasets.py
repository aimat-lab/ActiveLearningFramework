import logging

import mysql.connector
import numpy as np

from al_components.candidate_update.candidate_updater_implementations import Pool
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


input_definition = "ZERO double, ONE double, TWO double, THREE double, FOUR double, FIVE double, SIX double, SEVEN double, EIGHT double, NINE double, TEN double, ELEVEN double, TWELVE double"
input_reference = "ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, ELEVEN, TWELVE"
input_placeholders = "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s"
input_equal_check = "ZERO = %s AND ONE = %s AND TWO = %s AND THREE = %s AND FOUR = %s AND FIVE = %s AND SIX = %s AND SEVEN = %s AND EIGHT = %s AND NINE = %s AND TEN = %s AND ELEVEN = %s AND TWELVE = %s"

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

        # TODO: ensure unique identification by input
        sql = f"""CREATE TABLE {training_set_name} (
                        id int AUTO_INCREMENT PRIMARY KEY,
                        {input_definition},
                        PRICE double
                    )"""
        cursor.execute(sql)

        db.close()

        logging.info(f"finished initializing the Training set, database name: '{schema_name}.{training_set_name}'")

    def append_labelled_instance(self, x, y):
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"""INSERT INTO {training_set_name} (
                                {input_reference}, PRICE
                             ) VALUES (
                                {input_placeholders}, %s
                             )"""
        val = x_to_str_tuple(x) + (str(y),)

        cursor.execute(sql, val)
        db.commit()

        db.close()

    def retrieve_labelled_instance(self):
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT MIN(id), {input_reference}, PRICE FROM {training_set_name}")
        result = cursor.fetchall()

        if (len(result) == 0) or (result[0][0] is None):
            db.close()
            raise NoNewElementException(f"{schema_name}.{training_set_name}")

        x = np.array(result[0][1:-1])
        y = result[0][-1]

        return x, y

    def retrieve_all_labelled_instances(self):
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT * FROM {training_set_name}")
        res = cursor.fetchall()

        db.close()

        if (len(res) == 0) or (res[0][0] is None):
            raise NoNewElementException(f"{schema_name}.{training_set_name}")

        xs, ys = np.array([]), np.array([])
        for item in res:
            if len(xs) == 0:
                xs = np.array([np.array(item[1:-1])])
            else:
                xs = np.append(xs, [np.array(item[1:-1])], axis=0)
            ys = np.append(ys, item[-1])

        return xs, ys

    def remove_labelled_instance(self, x):
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DELETE FROM {training_set_name} WHERE {input_equal_check}"
        val = x_to_str_tuple(x)

        cursor.execute(sql, val)
        db.commit()
        db.close()

    def clear(self):
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

        # TODO: ensure unique identification by input
        # TODO: how should uncertainty look?????
        sql = f"""CREATE TABLE {candidate_set_name} (
                        id int AUTO_INCREMENT PRIMARY KEY,
                        {input_definition},
                        predicted_PRICE double, uncertainty double
                  )"""
        cursor.execute(sql)

        db.close()

        logging.info(f"finished initializing the Candidate set, database name: '{schema_name}.{candidate_set_name}'")

    def initiate_pool(self, x_initial):
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"""INSERT INTO {candidate_set_name} (
                                    {input_reference}
                                 ) VALUES (
                                    {input_placeholders}
                                 )"""
        val = []
        for i in range(len(x_initial)):
            val.append(x_to_str_tuple(x_initial[i]))

        cursor.executemany(sql, val)
        db.commit()

        db.close()

    def is_empty(self) -> bool:
        try:
            self.retrieve_all_instances()
        except NoNewElementException:
            return True
        return False

    def add_instance(self, x, y_prediction, uncertainty):
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"""INSERT INTO {candidate_set_name} (
                                    {input_reference}, predicted_PRICE, uncertainty 
                                 ) VALUES (
                                    {input_placeholders}, %s, %s
                                 )"""
        val = x_to_str_tuple(x) + (str(y_prediction), str(uncertainty))

        cursor.execute(sql, val)
        db.commit()

        db.close()

    def retrieve_all_instances(self):
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT {input_reference}, predicted_PRICE, uncertainty FROM {candidate_set_name}")
        res = cursor.fetchall()

        if (len(res) == 0) or (res[0][0] is None):
            db.close()
            raise NoNewElementException(f"{schema_name}.{candidate_set_name}")

        db.close()

        xs, ys, certainties = np.array([]), np.array([]), np.array([])
        for item in res:
            if len(xs) == 0:
                xs = np.array([np.array(item[0:-2])])
            else:
                xs = np.append(xs, [np.array(item[0:-2])], axis=0)
            ys = np.append(ys, item[-2])
            certainties = np.append(certainties, item[-1])

        return xs, ys, certainties

    def remove_instance(self, x):
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DELETE FROM {candidate_set_name} WHERE {input_equal_check}"
        val = x_to_str_tuple(x)

        cursor.execute(sql, val)
        db.commit()

        db.close()

    def update_instances(self, xs, new_y_predictions, new_certainties):
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"UPDATE {candidate_set_name} SET predicted_PRICE = %s, uncertainty = %s WHERE {input_equal_check}"
        val = []
        for i in range(len(xs)):
            val.append((str(new_y_predictions[i]), str(new_certainties[i])) + x_to_str_tuple(xs[i]))

        cursor.executemany(sql, val)
        db.commit()

        db.close()

    def get_first_instance(self):
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"""SELECT 
                                MIN(id),
                                {input_reference},
                                predicted_PRICE, uncertainty 
                            FROM {candidate_set_name}""")
        res = cursor.fetchall()

        if (len(res) == 0) or (res[0][0] is None):
            db.close()
            raise NoNewElementException(f"{schema_name}.{candidate_set_name}")

        db.close()

        x = np.array(res[0][1:-2])
        predicted = res[0][-2]
        uncertainty = res[0][-1]
        return x, predicted, uncertainty

    def get_instance(self, x):
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"""SELECT 
                        {input_reference},
                        predicted_PRICE, uncertainty 
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
        predicted = res[0][-2]
        uncertainty = res[0][-1]
        return x, predicted, uncertainty


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

    def add_instance(self, x):
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"SELECT id from {query_set_name} WHERE {input_equal_check}"
        val = x_to_str_tuple(x)

        cursor.execute(sql, val)
        res = cursor.fetchall()
        if len(res) == 0:  # TODO: if already exists: how to set last write?
            sql = f"INSERT INTO {query_set_name} ({input_reference}) VALUES ({input_placeholders})"

            cursor.execute(sql, val)
            db.commit()

        db.close()

    def get_instance(self):
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT MIN(id), {input_reference} from {query_set_name}")
        res = cursor.fetchall()
        db.close()

        if (len(res) == 0) or (res[0][0] is None):
            raise NoNewElementException(f"{schema_name}.{query_set_name}")

        x = np.array(res[0][1:])
        return x

    def remove_instance(self, x):
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()
        sql = f"DELETE from {query_set_name} WHERE {input_equal_check}"
        val = x_to_str_tuple(x)

        cursor.execute(sql, val)
        db.commit()

        db.close()
