import logging

import mysql.connector

from additional_component_interfaces import Oracle
from helpers.exceptions import NoSuchElementException


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

ground_truth_set_name = "ground_truth"


class OracleHouses(Oracle):

    def __init__(self, xs, ys):
        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DROP TABLE IF EXISTS {ground_truth_set_name}"
        cursor.execute(sql)

        sql = f"""CREATE TABLE {ground_truth_set_name} (
                            id int AUTO_INCREMENT PRIMARY KEY,
                            {input_definition},
                            PRICE double
                        )"""
        cursor.execute(sql)

        sql = f"""INSERT INTO {ground_truth_set_name} (
                            {input_reference}, PRICE
                         ) VALUES (
                            {input_placeholders}, %s
                         )"""
        val = []
        for i in range(len(xs)):
            val.append(x_to_str_tuple(xs[i]) + (str(ys[i]),))

        cursor.executemany(sql, val)
        db.commit()

        db.close()

    def query(self, x):
        sql = f"SELECT PRICE FROM {ground_truth_set_name} WHERE {input_equal_check}"
        val = x_to_str_tuple(x)

        db = connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(sql, val)
        res = cursor.fetchall()
        db.close()

        if len(res) == 0:
            logging.error("Could not find the queried instance => can't add it to labelled set (SHOULD NOT HAPPEN)")
            raise NoSuchElementException(f"house_pricing_example.{ground_truth_set_name}", x)
        else:
            return res[0][0]
