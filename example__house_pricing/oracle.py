import mysql.connector

from additional_component_interfaces import Oracle
from helpers.exceptions import NoSuchElementException


class OracleHouses(Oracle):

    def __init__(self, xs, ys):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        sql = "DROP TABLE IF EXISTS ground_truth"
        cursor.execute(sql)

        sql = """CREATE TABLE ground_truth (
                            id int AUTO_INCREMENT PRIMARY KEY,
                            ZERO double,
                            ONE double,
                            TWO double,
                            THREE double,
                            FOUR double,
                            FIVE double,
                            SIX double,
                            SEVEN double,
                            EIGHT double,
                            NINE double,
                            TEN double,
                            ELEVEN double,
                            TWELVE double,
                            PRICE double
                        )"""
        cursor.execute(sql)

        sql = """INSERT INTO ground_truth (
                            ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, ELEVEN, TWELVE, PRICE
                         ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                         )"""
        val = []
        for i in range(len(xs)):
            val.append((str(xs[i][0]), str(xs[i][1]), str(xs[i][2]), str(xs[i][3]), str(xs[i][4]), str(xs[i][5]), str(xs[i][6]), str(xs[i][7]),
                        str(xs[i][8]), str(xs[i][9]), str(xs[i][10]), str(xs[i][11]), str(xs[i][12]), str(ys[i])))

        cursor.executemany(sql, val)
        db.commit()

        db.close()

    def query(self, x):
        sql = """SELECT PRICE FROM ground_truth WHERE
                    ZERO = %s AND ONE = %s AND TWO = %s AND THREE = %s AND FOUR = %s AND FIVE = %s AND SIX = %s 
                    AND SEVEN = %s AND EIGHT = %s AND NINE = %s AND TEN = %s AND ELEVEN = %s AND TWELVE = %s"""
        val = (str(x[0]), str(x[1]), str(x[2]), str(x[3]), str(x[4]), str(x[5]), str(x[6]), str(x[7]), str(x[8]), str(x[9]), str(x[10]), str(x[11]),
               str(x[12]))

        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        cursor.execute(sql, val)
        res = cursor.fetchall()
        db.close()

        if len(res) == 0:
            raise NoSuchElementException
        else:
            return res[0][0]
