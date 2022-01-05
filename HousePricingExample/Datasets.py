import mysql.connector
import numpy as np

from Interfaces import TrainingSet, CandidateSet, QuerySet


class TrainingSetHouses(TrainingSet):

    def __init__(self, x_initial, y_initial):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        sql = "DROP TABLE IF EXISTS labelled_set"
        cursor.execute(sql)

        sql = """CREATE TABLE labelled_set (
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

        sql = """INSERT INTO labelled_set (
                    ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, ELEVEN, TWELVE, PRICE
                 ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                 )"""
        val = []
        for i in range(len(x_initial)):
            val.append((str(x_initial[i][0]), str(x_initial[i][1]), str(x_initial[i][2]), str(x_initial[i][3]), str(x_initial[i][4]),
                        str(x_initial[i][5]), str(x_initial[i][6]), str(x_initial[i][7]), str(x_initial[i][8]), str(x_initial[i][9]),
                        str(x_initial[i][10]), str(x_initial[i][11]), str(x_initial[i][12]), str(y_initial[i])))

        cursor.executemany(sql, val)
        db.commit()

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
        val = (str(x[0]), str(x[1]), str(x[2]), str(x[3]), str(x[4]), str(x[5]), str(x[6]), str(x[7]), str(x[8]), str(x[9]), str(x[10]), str(x[11]),
               str(x[12]), str(y))

        cursor.execute(sql, val)
        db.commit()

        db.close()

    def pop_all_labelled_instances(self):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        cursor.execute("SELECT * FROM labelled_set")
        result = cursor.fetchall()

        cursor.execute("DELETE FROM labelled_set")
        db.commit()
        db.close()

        x, y = np.array([]), np.array([])
        for item in result:
            if len(x) == 0:
                x = np.array([np.array(item[1:-1])])
            else:
                x = np.append(x, [np.array(item[1:-1])], axis=0)
            y = np.append(y, item[-1])
        return x, y


class CandidateSetHouses(CandidateSet):

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

        sql = """CREATE TABLE predicted_set (
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
                            predicted_PRICE double,
                            accuracy double
                        )"""  # how should accuracy look?????
        cursor.execute(sql)

        db.close()

    def add_instance(self, x, y_prediction, accuracy):
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="toor",
            database="housepricing_example"
        )
        cursor = db.cursor()

        sql = """INSERT INTO predicted_set (
                                    ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, ELEVEN, TWELVE, predicted_PRICE, accuracy 
                                 ) VALUES (
                                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                                 )"""
        val = (str(x[0]), str(x[1]), str(x[2]), str(x[3]), str(x[4]), str(x[5]), str(x[6]), str(x[7]), str(x[8]), str(x[9]), str(x[10]), str(x[11]),
               str(x[12]), str(y_prediction), str(accuracy))

        cursor.execute(sql, val)
        db.commit()

        db.close()

    def retrieve_all_instances(self):
        pass

    def remove_instance(self, x):
        pass

    def update_instance(self, x, new_y_prediction, new_certainty):
        pass


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

        sql = """CREATE TABLE unlabelled_set (
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
                            TWELVE double
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

        sql = """INSERT INTO unlabelled_set (
                                            ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, ELEVEN, TWELVE 
                                         ) VALUES (
                                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                                         )"""
        val = (str(x[0]), str(x[1]), str(x[2]), str(x[3]), str(x[4]), str(x[5]), str(x[6]), str(x[7]), str(x[8]), str(x[9]), str(x[10]), str(x[11]),
               str(x[12]))

        cursor.execute(sql, val)
        db.commit()

        db.close()

    def pop_instance(self):
        pass
