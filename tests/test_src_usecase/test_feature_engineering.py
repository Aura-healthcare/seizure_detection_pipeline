import pytest
import os

from src.usecase.feature_engineering import Feature_engineering

DATASET_FILE_PATH = '../seizure_detection_pipeline/output/cons-v0_6/PAT_18/cons_PAT_18_Annotations_EEG_26767_s1.csv'
OBJECTIF_VAR = 0.80

def test_init_feature_engineering_object():
    
    feat_eng = Feature_engineering(DATASET_FILE_PATH, OBJECTIF_VAR)

    assert feat_eng


def test_impute_nan_and_infinite_values():

    
    feat_eng = Feature_engineering(DATASET_FILE_PATH, OBJECTIF_VAR)

    not_expected_for_imputation = None

    if feat_eng.dataframe.isna().sum().any() != 0:

        feat_eng.impute_nan_and_infinite_values()

        X_imputed = feat_eng.X_imputed

        assert not_expected_for_imputation != X_imputed.any()


def test_outlier_detection():
    
    feat_eng = Feature_engineering(DATASET_FILE_PATH, OBJECTIF_VAR)

    init_shape = feat_eng.dataframe.shape

    if feat_eng.dataframe.isna().sum().any() != 0:

        feat_eng.impute_nan_and_infinite_values()

    feat_eng.outlier_detection()

    final_shape = feat_eng.Y.shape

    assert init_shape[0] != final_shape[0]