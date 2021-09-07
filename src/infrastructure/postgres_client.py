from typing import Dict

import psycopg2
from psycopg2 import Error
from psycopg2.extensions import connection, ISOLATION_LEVEL_AUTOCOMMIT


class PostgresClient:

    def __init__(self, host: str = "postgres", port: int = 5432,
                 user: str = "postgres", password: str = "postgres"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password

    def connect_to_database(self, database: str) -> connection:
        try:
            conn = psycopg2.connect(host=self.host, port=self.port,
                                    database=database, user=self.user,
                                    password=self.password)
        except (Exception, Error):
            print(f"Database {database} does not exist. Creating one...")
            conn = psycopg2.connect(host=self.host, port=self.port,
                                    user=self.user, password=self.password)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE {database};")
            print(f"Database {database} has been created.")
            cursor.close()
            conn.close()
            conn = psycopg2.connect(host=self.host, port=self.port,
                                    database=database, user=self.user,
                                    password=self.password)
        finally:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            return conn

    def check_if_table_exists(self, database: str, table: str) -> bool:
        conn = self.connect_to_database(database)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE \
                   table_name='{table}';")
        table_exists = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return table_exists

    def create_table(self, database: str, table: str,
                     entry_name_type_dict: Dict[str, str]):
        conn = self.connect_to_database(database)
        cursor = conn.cursor()
        sql_request = f"CREATE TABLE {table} (" + \
                      ', '.join(entry_name + " " + entry_type
                                for entry_name, entry_type
                                in entry_name_type_dict.items()) + \
                      ");"
        try:
            cursor.execute(sql_request)
        except Exception:
            print(f"Table {table} already exists.")
        cursor.close()
        conn.close()

    def write_in_table(self, database: str, table: str,
                       entry_name_value_dict: Dict[str, str]):
        conn = self.connect_to_database(database)
        cursor = conn.cursor()
        sql_request = f"INSERT INTO {table} (" + \
                      ', '.join(entry_name_value_dict.keys()) + \
                      ") VALUES (" + \
                      ', '.join(entry_name_value_dict.values()) + \
                      ");"
        cursor.execute(sql_request)
        cursor.close()
        conn.close()

    def delete_from_table(self, database: str, table: str,
                          entry_name_value_dict: Dict[str, str]):
        conn = self.connect_to_database(database)
        cursor = conn.cursor()
        sql_request = f"DELETE FROM {table} WHERE " + \
                      ' AND '.join(entry_name + "=" + entry_type
                                   for entry_name, entry_type
                                   in entry_name_value_dict.items()) + \
                      ";"
        cursor.execute(sql_request)
        cursor.close()
        conn.close()

    def check_if_values_in_table(self, database: str, table: str,
                                 entry_name_value_dict: Dict[str, str]) \
            -> bool:
        conn = self.connect_to_database(database)
        cursor = conn.cursor()
        sql_request = f"SELECT * FROM {table} WHERE " + \
                      ' AND '.join(entry_name + "=" + entry_type
                                   for entry_name, entry_type
                                   in entry_name_value_dict.items()) + \
                      ";"
        cursor.execute(sql_request)
        resp = cursor.fetchall()
        cursor.close()
        conn.close()
        return len(resp) > 0
