import pandas as pd
import psycopg2
from psycopg2 import Error
from psycopg2.extensions import connection, ISOLATION_LEVEL_AUTOCOMMIT
import argparse
import sys
from typing import List

## example CLI -> python3 rr_intervals_loader.py --pg-host "localhost" --pg-port 5432 --pg-user "lecoued" --pg-password "DemoAICE" --pg-database "demo_aice" --filepath "rr_00007633_s003_t007.csv" --exam "00007633_s003_t007"
PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWD = "postgres"
PG_DATABASE = "postgres"
PG_TABLE = "physio"

CSV_FILE = "../data/test_data/rr_00007633_s003_t007.csv"
EXAM = "/00007633_s003_t007"

def get_connection_to_db(host: str = PG_HOST, port: int = PG_PORT,
                         database: str = PG_DATABASE, user: str = PG_USER,
                         password: str = PG_PASSWD) -> connection:
    try:
        conn = psycopg2.connect(host=host, port=port,
                                database=database, user=user,
                                password=password)
    except (Exception, Error):
        print(f"Database {database} does not exist. Creating one...")
        conn = psycopg2.connect(host=host, port=port, user=user,
                                password=password)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE {database};")
        print(f"Database {database} has been created.")
        cursor.close()
        conn.close()
        conn = psycopg2.connect(host=host, port=port,
                                database=database, user=user,
                                password=password)
    finally:
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        print("Database is OK")
        return conn


def write_physio_to_db(csv_file: str, pg_host: str, pg_port: int, pg_database: str, pg_user: str, pg_password: str) -> None:
    conn = get_connection_to_db(pg_host, pg_port, pg_database, pg_user, pg_password)
    cursor = conn.cursor()
    # Check if the table already exists
    cursor.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE \
                   table_name='{PG_TABLE}';")
    if not cursor.fetchone()[0]:
        print(f"Table {PG_TABLE} does not exist. Creating one...")
        cursor.execute(f"CREATE TABLE {PG_TABLE} \
            (id SERIAL PRIMARY KEY, \
            time TIMESTAMP, \
            rr_interval REAL, \
            heart_rate INT, \
            patient VARCHAR(100) \
        );")
        print(f"Table {PG_TABLE} has been created")

    df=pd.read_csv(csv_file, sep=',')
    df.rename(columns={ df.columns[0]: "timestamp"}, inplace=True)
    df["heart_rate"] = df["rr_interval"].apply(lambda x: int(60 * 1000 / x))

    for _, row in df.iterrows():
        cursor.execute(f"INSERT INTO {PG_TABLE} VALUES \
            (DEFAULT, \
            '{row['timestamp']}', \
            {row['rr_interval']}, \
            {row['heart_rate']}, \
            '{EXAM}' \
            );")

    print("Physio values added")
    cursor.close()
    conn.close()


def parse_load_physio_args(args_to_parse: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CLI parameter input')
    parser.add_argument('--pg-host',
                        dest='pg_host',
                        required=True)
    parser.add_argument('--pg-port',
                        dest='pg_port',
                        required=True)
    parser.add_argument('--pg-user',
                        dest='pg_user',
                        required=True)
    parser.add_argument('--pg-password',
                        dest='pg_password',
                        required=True)
    parser.add_argument('--pg-database',
                        dest='pg_database',
                        required=True)
    parser.add_argument('--filepath',
                        dest='filepath',
                        required=True)
    parser.add_argument('--exam',
                        dest='exam',
                        required=True)
    args = parser.parse_args(args_to_parse)

    return args


if __name__ == "__main__":
    args = parse_load_physio_args(sys.argv[1:])
    d = vars(args)
    write_physio_to_db(d["filepath"], d["pg_host"], d["pg_port"], d["pg_database"], d["pg_user"], d["pg_password"])
