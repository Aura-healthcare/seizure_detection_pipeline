import pandas as pd
import psycopg2
from psycopg2 import Error
from psycopg2.extensions import connection, ISOLATION_LEVEL_AUTOCOMMIT
import argparse
import sys
from typing import List
from pyedflib import highlevel
from pyedflib import edfreader
import datetime

## example CLI -> python3 physio_data_loader.py --pg-host "localhost" --pg-port 5432 --pg-user "lecoued" --pg-password "DemoAICE" --pg-database "demo_aice" --filepath "rr_00007633_s003_t007.csv" --exam "00007633_s003_t007"
PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWD = "postgres"
PG_DATABASE = "postgres"
PG_TABLE = "annotations"

TSE_BI_FILE = "00007633_s003_t007.tse_bi"
EXAM = "00007633_s003_t007"


def read_tse_bi(filename: str) -> pd.DataFrame:
    if filename.split('/')[-1].split('.')[-1] != 'tse_bi':
        raise ValueError(
            f'Please input a tse_bi file. Input: {filename}')

    df_tse_bi = pd.read_csv(filename,
                            sep=' ',
                            skiprows=1,
                            header=None)

    df_tse_bi.columns = ['start', 'end', 'annotation', 'probablility']
    df_tse_bi.loc[:, ['start', 'end']] = df_tse_bi.loc[:, ['start', 'end']].\
        apply(lambda x: x * 1_000)

    return df_tse_bi

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


def write_annotations_to_db(tsebi_file: str, edf_file:str, pg_host: str, pg_port: int, pg_database: str, pg_user: str, pg_password: str) -> None:
    conn = get_connection_to_db(pg_host, pg_port, pg_database, pg_user, pg_password)
    cursor = conn.cursor()
    # Check if the table already exists
    cursor.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE \
                   table_name='{PG_TABLE}';")
    if not cursor.fetchone()[0]:
        print(f"Table {PG_TABLE} does not exist. Creating one...")
        cursor.execute(f"CREATE TABLE {PG_TABLE} \
            (id SERIAL PRIMARY KEY, \
            start_time TIMESTAMP, \
            end_time TIMESTAMP, \
            tag VARCHAR(50), \
            exam VARCHAR(100) \
        );")
        print(f"Table {PG_TABLE} has been created")

    edf_headers = highlevel.read_edf_header(edf_file)
    start_date = edf_headers['startdate']
    print(start_date)
    df_tse=read_tse_bi(tsebi_file)
    df_tse["start"] = df_tse["start"].apply(lambda x: (start_date + datetime.timedelta(milliseconds=x)))
    df_tse["end"] = df_tse["end"].apply(lambda x: start_date + datetime.timedelta(milliseconds=x))

    for _, row in df_tse.iterrows():
        print(row)
        cursor.execute(f"INSERT INTO {PG_TABLE} VALUES \
            (DEFAULT, \
            '{row['start']}', \
            '{row['end']}', \
            '{row['annotation']}', \
            '{EXAM}' \
            );")

    print("Annotations values added")
    cursor.close()
    conn.close()


def parse_load_annotations_args(args_to_parse: List[str]) -> argparse.Namespace:
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
    parser.add_argument('--annotation-filename',
                        dest='annotation_filename',
                        required=True)
    parser.add_argument('--edf-filename',
                        dest='edf_filename',
                        required=True)
    parser.add_argument('--exam',
                        dest='exam',
                        required=True)
    args = parser.parse_args(args_to_parse)

    return args


if __name__ == "__main__":
    args = parse_load_annotations_args(sys.argv[1:])
    d = vars(args)
    write_annotations_to_db(d["annotation_filename"], d["edf_filename"], d["pg_host"], d["pg_port"], d["pg_database"], d["pg_user"], d["pg_password"])
