"""
This script is used to make integration testi on both TUH and dataset format.

Copyright (C) 2022 Association Aura
SPDX-License-Identifier: GPL-3.0
"""
import os
import pandas as pd
import shutil
import subprocess
from pathlib import Path


def test_run_pipeline_tuh(path_data_source: str = Path('data/tuh'),
                          path_test: str = Path('tests/temp')):
    """Make integration test on TUH format.

    Copy sample data, process all the bash pipeline up to the creation of the
    ml dataset, then delete temporary files.

    parameters
    ----------
    path_data_source : str
        path of the sample data to use
    path_test : str
        temporary folder to use for integration testing. It will be deleted
        after the testing
    """
    path_data = os.path.join(path_test, 'data')
    path_export = os.path.join(path_test, 'output')

    if os.path.exists(path_test):
        shutil.rmtree(path_test)

    shutil.copytree(path_data_source,
                    path_data,
                    ignore=shutil.ignore_patterns("00004671_s007_t000.*",
                                                  "00007633_s003_t007*"))

    if not os.path.exists(path_export):
        os.mkdir(path_export)

    bash_detect_qrs_wrapper = subprocess.Popen([
        "./scripts/bash_pipeline/1_detect_qrs_wrapper.sh",
        "-i",
        path_data,
        "-o",
        os.path.join(path_export, 'res-v0_6')])
    bash_detect_qrs_wrapper.wait()

    bash_compute_hrv_analysis_features_wrapper = subprocess.Popen([
        "./scripts/bash_pipeline/2_compute_hrvanalysis_features_wrapper.sh",
        "-i",
        os.path.join(path_export, 'res-v0_6'),
        "-o",
        os.path.join(path_export, 'feats-v0_6')])
    bash_compute_hrv_analysis_features_wrapper.wait()

    bash_consolidate_feats_and_annots = subprocess.Popen([
        "./scripts/bash_pipeline/3_consolidate_feats_and_annot_wrapper.sh",
        "-i",
        os.path.join(path_export, 'feats-v0_6'),
        "-a",
        path_data,
        "-o",
        os.path.join(path_export, 'cons-v0_6')])
    bash_consolidate_feats_and_annots.wait()

    create_ml_dataset = subprocess.Popen([
        "python3",
        "src/usecase/create_ml_dataset.py",
        "--input-folder",
        os.path.join(path_export, 'cons-v0_6'),
        "--output-folder",
        os.path.join(path_export, 'ml_dataset')])
    create_ml_dataset.wait()

    df = pd.read_csv(os.path.join(path_export, 'ml_dataset/df_ml.csv'))
    assert(df.shape[0] == 33)
    assert(df.shape[1] >= 31)
    assert('interval_index' in df.columns)
    assert('interval_start_time' in df.columns)
    assert('label' in df.columns)
    assert('timestamp' in df.columns)

    if os.path.exists(path_test):
        shutil.rmtree(path_test)


def test_run_pipeline_dataset(path_data_source: str = Path('data/dataset'),
                              path_test: str = Path('tests/temp')):
    """Make integration test on dataset format.

    Copy sample data from the features phase and process the remaining
    pipeline up to the creation of the  ml dataset, then delete temporary
    files.

    parameters
    ----------
    path_data_source : str
        path of the sample data to use
    path_test : str
        temporary folder to use for integration testing. It will be deleted
        after the testing
    """
    path_data = os.path.join(path_test, 'data')
    path_export = os.path.join(path_test, 'output')

    if os.path.exists(path_test):
        shutil.rmtree(path_test)

    shutil.copytree(path_data_source,
                    path_data,
                    ignore=shutil.ignore_patterns("00004671_s007_t000.*",
                                                  "00007633_s003_t007*"))

    # Create features folder and copy already computed files from the data
    # folder. This will allows next steps of the pipeline to run.

    if not os.path.exists(path_export):
        os.mkdir(path_export)
        os.mkdir(os.path.join(path_export,
                              'feats-v0_6'))
        os.mkdir(os.path.join(path_export,
                              'feats-v0_6/PAT_0'))

    shutil.copyfile(
        "data/test_data/dataset_feats_PAT_0_Annotations_EEG_0_s2.csv",
        os.path.join(path_export,
                     'feats-v0_6/PAT_0/feats_EEG_0_s2.csv'))

    bash_consolidate_feats_and_annots = subprocess.Popen([
        "./scripts/bash_pipeline/3_consolidate_feats_and_annot_wrapper.sh",
        "-i",
        os.path.join(path_export, 'feats-v0_6'),
        "-a",
        path_data,
        "-o",
        os.path.join(path_export, 'cons-v0_6')])
    bash_consolidate_feats_and_annots.wait()

    create_ml_dataset = subprocess.Popen([
        "python3",
        "src/usecase/create_ml_dataset.py",
        "--input-folder",
        os.path.join(path_export, 'cons-v0_6'),
        "--output-folder",
        os.path.join(path_export, 'ml_dataset')])
    create_ml_dataset.wait()

    df = pd.read_csv(os.path.join(path_export, 'ml_dataset/df_ml.csv'))
    assert(df.shape[0] == 599)
    assert(df.shape[1] >= 31)
    assert('interval_index' in df.columns)
    assert('interval_start_time' in df.columns)
    assert('label' in df.columns)
    assert('timestamp' in df.columns)

    if os.path.exists(path_test):
        shutil.rmtree(path_test)
