import string
from typing import Sequence

from basic_sl_component_interfaces import Oracle
from helpers import X, Y
from helpers.database_helper.database_info_store import DefaultDatabaseHelper
from helpers.exceptions import CantResolveQueryException

oracle_set: string = "oracle_ground_truth_set"


class ButeneOracle(Oracle):

    def __init__(self, host: string, user: string, password: string, database: string, input_definition: string, output_definition: string, xs: Sequence[X], ys: Sequence[Y]):
        self.database_info = DefaultDatabaseHelper(host=host, user=user, password=password, database=database, input_definition=input_definition, output_definition=output_definition, additional_candidate_information_definition="")

        input_reference = self.database_info.create_reference_from_sql_definition(self.database_info.input_definition)
        output_reference = self.database_info.create_reference_from_sql_definition(self.database_info.output_definition)
        input_placeholders = self.database_info.create_placeholders_from_sql_definition(self.database_info.input_definition)
        output_placeholders = self.database_info.create_placeholders_from_sql_definition(self.database_info.output_definition)

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DROP TABLE IF EXISTS {oracle_set}"
        cursor.execute(sql)

        sql = f"CREATE TABLE {oracle_set} (id int AUTO_INCREMENT PRIMARY KEY, {input_definition}, {output_definition})"
        cursor.execute(sql)

        sql = f"""INSERT INTO {oracle_set} (
                                    {input_reference}, {output_reference}
                                 ) VALUES (
                                    {input_placeholders}, {output_placeholders}
                                 )"""
        val = []
        for i in range(len(xs)):
            val.append(self.database_info.x_to_str_tuple(xs[i]) + self.database_info.y_to_str_tuple(ys[i]))
            if len(val) > 100:
                cursor.executemany(sql, val)
                db.commit()
                val = []

        cursor.executemany(sql, val)
        db.commit()

        db.close()

    def query(self, x: X) -> Y:
        input_equal_check = self.database_info.create_equal_check_from_sql_definition(self.database_info.input_definition)
        output_reference = self.database_info.create_reference_from_sql_definition(self.database_info.output_definition)

        sql = f"SELECT {output_reference} FROM {oracle_set} WHERE {input_equal_check}"
        val = self.database_info.x_to_str_tuple(x)

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(sql, val)
        res = cursor.fetchall()
        db.close()

        if len(res) == 0:
            raise CantResolveQueryException(x)
        else:
            return res[0]
