import os
import json

import click

import src.domain.ecg_qc_stats as stats
from src.infrastructure.postgres_client import PostgresClient


CHUNKS_FOLDER = 'output/quality'
ENTRY_NAME_TYPE_DICT = {
    "record": "varchar",
    "model_ecg_qc": "varchar",
    "length_record": "integer",
    "min_interval": "integer",
    "max_interval": "integer",
    "med_interval": "real",
    "noisy_percent": "real"
}
POSTGRES_DATABASE = 'postgres'
TABLE_NAME = 'noisy_info'


def ecg_qc_statistical_analysis(chunk_file: str, local_call: bool = False):
    '''
    Computes statistics about results of an ECG QC model, and writes them in a
    table in PostgreSQL.
    '''
    record, model = chunk_file.split('.')[0].split('_#_')
    model_split = model.split('_')
    try:
        length_chunk = int(model_split[-1][:-1])
    except ValueError:
        length_chunk = 9
    with open(os.path.join(CHUNKS_FOLDER, chunk_file), 'r') as f:
        list_noisy_segments = json.load(f)

    length_record = stats.length_record(list_noisy_segments, length_chunk)
    mini, maxi, med = stats.noise_free_intervals_stats(
        list_noisy_segments, length_chunk)
    noisy_pourcent = stats.percentage_noisy_segments(list_noisy_segments)

    if local_call:
        local_port = os.getenv('POSTGRES_PORT')
        postgres_client = PostgresClient(host='localhost', port=local_port)
    else:
        postgres_client = PostgresClient()
    table_exists = postgres_client.check_if_table_exists(POSTGRES_DATABASE,
                                                         TABLE_NAME)
    if not table_exists:
        postgres_client.create_table(POSTGRES_DATABASE, TABLE_NAME,
                                     ENTRY_NAME_TYPE_DICT)

    values_to_insert = [f"'{record}'", f"'{model}'",
                        str(length_record), str(mini),
                        str(maxi), str(med), str(noisy_pourcent)]
    dict_to_insert = dict(zip(
        ENTRY_NAME_TYPE_DICT.keys(), values_to_insert))

    postgres_client.write_in_table(
        POSTGRES_DATABASE, TABLE_NAME, dict_to_insert)


@click.command()
@click.option('--chunk-file', required=True)
def main(chunk_file: str):
    ecg_qc_statistical_analysis(chunk_file, local_call=True)


if __name__ == "__main__":
    main()
