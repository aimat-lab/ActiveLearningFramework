import mysql.connector
import numpy as np

from al_components.candidate_update.candidate_updater_implementations import Pool
from helpers.exceptions import NoNewElementException
from workflow_management.database_interfaces import TrainingSet, QuerySet


class TrainingSetHouses(TrainingSet):

    def __init__(self):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        sql = "DROP TABLE IF EXISTS labelled_set"
        cursor.execute(sql)

        # TODO: ensure unique identification by input
        sql = """CREATE TABLE labelled_set (
                        id int AUTO_INCREMENT PRIMARY KEY,
                        ZERO double, ONE double, TWO double, THREE double, FOUR double, FIVE double, SIX double, SEVEN double, EIGHT double, NINE double, TEN double, ELEVEN double, TWELVE double,
                        PRICE double
                    )"""
        cursor.execute(sql)

        db.close()

    def append_labelled_instance(self, x, y):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        sql = """INSERT INTO labelled_set (
                                ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, ELEVEN, TWELVE, PRICE
                             ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                             )"""
        val = (str(x[0]), str(x[1]), str(x[2]), str(x[3]), str(x[4]), str(x[5]), str(x[6]), str(x[7]), str(x[8]), str(x[9]), str(x[10]), str(x[11]), str(x[12]), str(y))

        cursor.execute(sql, val)
        db.commit()

        db.close()

    def retrieve_labelled_instance(self):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        cursor.execute("SELECT MIN(id), ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, ELEVEN, TWELVE, PRICE FROM labelled_set")
        result = cursor.fetchall()

        if len(result) == 0:
            db.close()
            raise NoNewElementException("labelled_set")

        x = np.array(result[0][1:-1])
        y = result[0][-1]

        return x, y

    def retrieve_all_labelled_instances(self):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )

        cursor = db.cursor()

        cursor.execute("SELECT * FROM labelled_set")
        result = cursor.fetchall()

        db.close()

        if len(result) == 0:
            raise NoNewElementException("housepricing_example.labelled_set")

        xs, ys = np.array([]), np.array([])
        for item in result:
            if len(xs) == 0:
                xs = np.array([np.array(item[1:-1])])
            else:
                xs = np.append(xs, [np.array(item[1:-1])], axis=0)
        ys = np.append(ys, item[-1])

        return xs, ys

    def remove_labelled_instance(self, x):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        sql = "DELETE FROM labelled_set WHERE ZERO = %s AND ONE = %s AND TWO = %s AND THREE = %s AND FOUR = %s AND FIVE = %s AND SIX = %s AND SEVEN = %s AND EIGHT = %s AND NINE = %s AND TEN = %s AND ELEVEN = %s AND TWELVE = %s"
        val = (str(x[0]), str(x[1]), str(x[2]), str(x[3]), str(x[4]), str(x[5]), str(x[6]), str(x[7]), str(x[8]), str(x[9]), str(x[10]), str(x[11]), str(x[12]))

        cursor.execute(sql, val)
        db.commit()
        db.close()

    def clear(self):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        cursor.execute("DELETE FROM labelled_set")
        db.commit()
        db.close()


class CandidateSetHouses(Pool):

    def __init__(self):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        sql = "DROP TABLE IF EXISTS predicted_set"
        cursor.execute(sql)

        # TODO: ensure unique identification by input
        sql = """CREATE TABLE predicted_set (
                    id int AUTO_INCREMENT PRIMARY KEY,
                    ZERO double, ONE double, TWO double, THREE double, FOUR double, FIVE double, SIX double, SEVEN double, EIGHT double, NINE double, TEN double, ELEVEN double, TWELVE double,
                    predicted_PRICE double,
                    certainty double
                 )"""  # TODO: how should certainty look like?????
        cursor.execute(sql)

        db.close()

    def initiate_pool(self, x_initial):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        sql = """INSERT INTO predicted_set (
                                    ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, ELEVEN, TWELVE
                                 ) VALUES (
                                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                                 )"""
        val = []
        for i in range(len(x_initial)):
            val.append((str(x_initial[i][0]), str(x_initial[i][1]), str(x_initial[i][2]), str(x_initial[i][3]), str(x_initial[i][4]), str(x_initial[i][5]),
                        str(x_initial[i][6]), str(x_initial[i][7]), str(x_initial[i][8]), str(x_initial[i][9]), str(x_initial[i][10]), str(x_initial[i][11]),
                        str(x_initial[i][12])))

        cursor.executemany(sql, val)
        db.commit()

        db.close()

    def add_instance(self, x, y_prediction, certainty):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        sql = """INSERT INTO predicted_set (
                                    ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, ELEVEN, TWELVE, predicted_PRICE, certainty 
                                 ) VALUES (
                                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                                 )"""
        val = (str(x[0]), str(x[1]), str(x[2]), str(x[3]), str(x[4]), str(x[5]), str(x[6]), str(x[7]), str(x[8]), str(x[9]), str(x[10]), str(x[11]),
               str(x[12]), str(y_prediction), str(certainty))

        cursor.execute(sql, val)
        db.commit()

        db.close()

    def retrieve_all_instances(self):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        cursor.execute("""SELECT 
                            ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, ELEVEN, TWELVE,
                            predicted_PRICE, certainty 
                            FROM predicted_set""")
        res = cursor.fetchall()

        if len(res) == 0:
            db.close()
            raise NoNewElementException

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
        pass

    def update_instance(self, x, new_y_prediction, new_certainty):
        pass

    def get_instance(self):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        cursor.execute("""SELECT 
                            MIN(id),
                            ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, ELEVEN, TWELVE,
                            predicted_PRICE, certainty from predicted_set""")
        res = cursor.fetchall()

        if len(res) == 0:
            db.close()
            raise NoNewElementException

        db.close()

        x = np.array(res[0][1:-2])
        predicted = res[0][-2]
        certainty = res[0][-1]
        return x, predicted, certainty


class QuerySetHouses(QuerySet):

    def __init__(self):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        sql = "DROP TABLE IF EXISTS unlabelled_set"
        cursor.execute(sql)

        # TODO: ensure unique identification by input
        sql = """CREATE TABLE unlabelled_set (
                    id int AUTO_INCREMENT PRIMARY KEY,
                    ZERO double, ONE double, TWO double, THREE double, FOUR double, FIVE double, SIX double, SEVEN double, EIGHT double, NINE double, TEN double, ELEVEN double, TWELVE double
                 )"""
        cursor.execute(sql)

        db.close()

        self.last_read_idx = -1
        self.last_write_idx = -1

    def add_instance(self, x):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        sql = "SELECT id from unlabelled_set WHERE ZERO = %s AND ONE = %s AND TWO = %s AND THREE = %s AND FOUR = %s AND FIVE = %s AND SIX = %s AND SEVEN = %s AND EIGHT = %s AND NINE = %s AND TEN = %s AND ELEVEN = %s AND TWELVE = %s"
        val = (str(x[0]), str(x[1]), str(x[2]), str(x[3]), str(x[4]), str(x[5]), str(x[6]), str(x[7]), str(x[8]), str(x[9]), str(x[10]), str(x[11]), str(x[12]))

        cursor.execute(sql, val)
        res = cursor.fetchall()
        if len(res) == 0:  # if already exists: how to set last write?
            sql = """INSERT INTO unlabelled_set (
                            ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, ELEVEN, TWELVE 
                     ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                     )"""

            cursor.execute(sql, val)
            db.commit()

            sql = "SELECT id from unlabelled_set WHERE ZERO = %s AND ONE = %s AND TWO = %s AND THREE = %s AND FOUR = %s AND FIVE = %s AND SIX = %s AND SEVEN = %s AND EIGHT = %s AND NINE = %s AND TEN = %s AND ELEVEN = %s AND TWELVE = %s"

            cursor.execute(sql, val)
            res = cursor.fetchall()
            (id,) = res[0]
            self.last_write_idx = id

        db.close()

    def get_instance(self):
        if self.last_write_idx == self.last_read_idx:
            raise NoNewElementException
        else:
            db = mysql.connector.connect(
                host="localhost",
                user="root",
                password="toor",
                database="housepricing_example"
            )
            cursor = db.cursor()

            cursor.execute("SELECT MIN(id), ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, ELEVEN, TWELVE from unlabelled_set")
            res = cursor.fetchall()[0]
            db.close()

            x = np.array(res[1:])
            self.last_read_idx = res[0]
            return x

    def remove_instance(self, x):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()
        sql = "DELETE from unlabelled_set WHERE ZERO = %s AND ONE = %s AND TWO = %s AND THREE = %s AND FOUR = %s AND FIVE = %s AND SIX = %s AND SEVEN = %s AND EIGHT = %s AND NINE = %s AND TEN = %s AND ELEVEN = %s AND TWELVE = %s"
        val = (str(x[0]), str(x[1]), str(x[2]), str(x[3]), str(x[4]), str(x[5]), str(x[6]), str(x[7]), str(x[8]), str(x[9]), str(x[10]), str(x[11]), str(x[12]))

        cursor.execute(sql, val)
        db.commit()

        db.close()
