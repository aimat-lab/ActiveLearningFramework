import string
from dataclasses import dataclass
from typing import Tuple

import mysql.connector

from helpers import X, Y, CandInfo


# noinspection PyMethodMayBeStatic
@dataclass()
class DefaultDatabaseHelper:
    host: string
    user: string
    password: string
    database: string

    # TODO: ensure unique identification by input
    input_definition: string
    additional_candidate_information_definition: string
    output_definition: string
    log_query_decision_information_definition: string = "info_value double, queried double"

    candidate_set_name: string = "candidate_set"
    # TODO: merge training set and stored labelled set
    training_set_name: string = "training_set"
    stored_labelled_set_name: string = "stored_labelled_set"
    query_set_name: string = "query_set"
    log_query_decision_set_name: string = "log_query_decision_database"

    def connect_to_house_pricing_example_db(self):
        return mysql.connector.connect(host=self.host, user=self.user, password=self.password, database=self.database)

    def x_to_str_tuple(self, x: X) -> Tuple[str, ...]:
        return tuple([str(x_part) for x_part in x])

    def y_to_str_tuple(self, y: Y) -> Tuple[str, ...]:
        return str(y),

    def additional_candidate_information_to_str_tuple(self, additional_candidate_information: CandInfo) -> Tuple[str, ...]:
        return tuple([str(info_part) for info_part in additional_candidate_information])

    def log_query_decision_information_to_str_tuple(self, info_value: float, queried: bool) -> Tuple[str, ...]:
        if queried:
            return str(info_value), "1"
        else:
            return str(info_value), "0"

    def create_placeholders_from_sql_definition(self, sql_definition: string) -> string:
        return ", ".join(["%s" for _ in sql_definition.split(", ")])

    def create_reference_from_sql_definition(self, sql_definition: string) -> string:
        return ", ".join([part.strip().split(" ")[0] for part in sql_definition.split(", ")])

    def create_equal_check_from_sql_definition(self, sql_definition: string) -> string:
        return " AND ".join([part.strip().split(" ")[0] + " = %s" for part in sql_definition.split(", ")])

    def create_set_reference_from_sql_definition(self, sql_definition: string) -> string:
        return ", ".join([part.strip().split(" ")[0] + " = %s" for part in sql_definition.split(", ")])
