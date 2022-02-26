import string
from typing import Tuple, Sequence

import numpy as np

from al_specific_components.candidate_update.candidate_updater_implementations import Pool
from helpers import X, CandInfo
from helpers.database_helper.database_info_store import DefaultDatabaseHelper
from helpers.exceptions import NoNewElementException, NoSuchElementException


class ButenePool(Pool):

    def __init__(self, host: string, user: string, password: string, database: string, example_x: X, example_cand_info: CandInfo):
        x_sql_definition = ", ".join(["x_" + str(i) + " double" for i in range(len(example_x))])
        cand_info_sql_definition = ", ".join(["cand_info_" + str(i) + " double" for i in range(len(example_cand_info))])
        self.database_info = DefaultDatabaseHelper(host=host, user=user, password=password, database=database,
                                                   input_definition=x_sql_definition, additional_candidate_information_definition=cand_info_sql_definition, output_definition="",
                                                   candidate_set_name="candidate_pool")

        candidate_set_name = self.database_info.candidate_set_name
        input_definition = self.database_info.input_definition
        additional_candidate_information_definition = self.database_info.additional_candidate_information_definition

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"DROP TABLE IF EXISTS {candidate_set_name}"
        cursor.execute(sql)

        sql = f"CREATE TABLE {candidate_set_name} (id int AUTO_INCREMENT PRIMARY KEY, {input_definition}, {additional_candidate_information_definition})"
        cursor.execute(sql)

        db.close()

    def initiate_pool(self, x_initial: Sequence[X]) -> None:
        candidate_set_name = self.database_info.candidate_set_name
        input_reference = self.database_info.create_reference_from_sql_definition(self.database_info.input_definition)
        input_placeholders = self.database_info.create_placeholders_from_sql_definition(self.database_info.input_definition)

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"INSERT INTO {candidate_set_name} ({input_reference}) VALUES ({input_placeholders})"
        val = []
        for i in range(len(x_initial)):
            val.append(self.database_info.x_to_str_tuple(x_initial[i]))

        cursor.executemany(sql, val)
        db.commit()

        db.close()

    def update_instances(self, xs: Sequence[X], new_additional_infos: Sequence[CandInfo]) -> None:
        candidate_set_name = self.database_info.candidate_set_name
        input_equal_check = self.database_info.create_equal_check_from_sql_definition(self.database_info.input_definition)
        additional_candidate_information_set = self.database_info.create_set_reference_from_sql_definition(self.database_info.additional_candidate_information_definition)

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        sql = f"UPDATE {candidate_set_name} SET {additional_candidate_information_set} WHERE {input_equal_check}"
        val = []
        for i in range(len(xs)):
            val.append(self.database_info.additional_candidate_information_to_str_tuple(new_additional_infos[i]) + self.database_info.x_to_str_tuple(xs[i]))

        cursor.executemany(sql, val)
        db.commit()

        db.close()

    def retrieve_all_instances(self) -> Tuple[Sequence[X], Sequence[CandInfo]]:
        candidate_set_name = self.database_info.candidate_set_name
        input_reference = self.database_info.create_set_reference_from_sql_definition(self.database_info.input_definition)
        additional_candidate_information_reference = self.database_info.create_set_reference_from_sql_definition(self.database_info.additional_candidate_information_definition)
        schema_name = self.database_info.database
        x_size = len(self.database_info.input_definition.split(", "))
        cand_info_size = len(self.database_info.additional_candidate_information_definition.split(", "))

        db = self.database_info.connect_to_house_pricing_example_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT {input_reference}, {additional_candidate_information_reference} FROM {candidate_set_name}")
        res = cursor.fetchall()

        if (len(res) == 0) or (res[0][0] is None):
            db.close()
            raise NoNewElementException(f"{schema_name}.{candidate_set_name}")

        db.close()

        xs, add_infos = np.array([]), None
        for item in res:
            x = np.array(item[1:x_size + 1])
            add_info = tuple(item[-cand_info_size:])
            if len(xs) == 0:
                xs = np.array([x])
                add_infos = np.array([add_info])
            else:
                xs = np.append(xs, [x], axis=0)
                add_infos = np.append(add_infos, [add_info], axis=0)

        return xs, add_infos

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
