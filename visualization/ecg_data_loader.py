import pandas as pd
import psycopg2
import argparse
from psycopg2 import Error
from psycopg2.extensions import connection, ISOLATION_LEVEL_AUTOCOMMIT
from pyedflib import highlevel
import sys
from pyedflib import edfreader
from sqlalchemy import create_engine
from typing import List

PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWD = "postgres"
PG_DATABASE = "postgres"
PG_TABLE = "ecg"

edf_file = "00007633_s003_t007.edf"
EXAM = "00007633_s003_t007"

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


def write_ecg_to_db(edf_file: str,
                    pg_host: str = PG_HOST,
                    pg_port: int = PG_PORT,
                    pg_database: str = PG_DATABASE,
                    pg_user: str = PG_USER,
                    pg_password: str = PG_PASSWD) -> None:

    conn = get_connection_to_db(pg_host, pg_port, pg_database, pg_user, pg_password)
    cursor = conn.cursor()
    # Check if the table already exists
    cursor.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE \
                     table_name='{PG_TABLE}';")
    if not cursor.fetchone()[0]:
        print(f"Table {PG_TABLE} does not exist. Creating one...")
        cursor.execute(f"CREATE TABLE {PG_TABLE} \
            (index INT, \
            ecg_signal REAL, \
            time TIMESTAMP, \
            patient VARCHAR(100), \
            PRIMARY KEY(index, patient) \
        );")
        print(f"Table {PG_TABLE} has been created")

    df = convert_edf_to_dataframe(edf_file=edf_file)

    engine = create_engine(f'postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}')

    try:
        df.to_sql(name=PG_TABLE,
                  con=engine,
                  if_exists='append',
                  chunksize=10_000)
        print("ECG values added")

    except Exception as e:
        # occur most of the time if segment is already loaded
        print(e)

    finally:
        cursor.close()
        conn.close()

def convert_edf_to_dataframe (edf_file: str,
                              channel_name: str = 'EEG FP1-REF')-> pd.DataFrame:

    # From its path, load an edf file for a selected channel and
    # adapt it in DataFrame

    headers = highlevel.read_edf_header(edf_file)
    channels = headers['channels']

    startdate = pd.to_datetime(
        headers['startdate'])

    # InfluxDB API

    fs = headers[
        'SignalHeaders'][
            channels.index(channel_name)]['sample_rate']

    with edfreader.EdfReader(edf_file) as f:
        signals = f.readSignal(channels.index(channel_name))
    # to modify? 
    freq_ns = int(1/fs*1_000_000_000)

    df = pd.DataFrame(signals,
                      columns=['ecg_signal'],
                      index=pd.date_range(startdate,
                                              periods=len(signals),
                                              freq=f'{freq_ns}ns'
                                              ))
    df['time'] = df.index
    df.reset_index(drop=True, inplace=True)
    df['patient'] = edf_file

    return df


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
    args = parser.parse_args(args_to_parse)

    return args



if __name__ == '__main__':
    args = parse_load_physio_args(sys.argv[1:])
    d = vars(args)
    write_ecg_to_db(d["filepath"], d["pg_host"], d["pg_port"], d["pg_database"], d["pg_user"], d["pg_password"])
