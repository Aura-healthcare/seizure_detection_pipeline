"""
Feature engineering process.

This script clean data before model fitting. The clean actions are defined by exploring data.
It's about imputting nan values by some strategy, detect outliers and remove them and make
some pca analysis to reduce multi-colinéarité.
"""

import string

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.stats import zscore

class FeatureEngineering:
    
    def __init__(
        self,
        patient_dataset_path: string,
        objectif_variance: float
    ) -> None:

        #read file in dataframe from path patient dataset
        self.dataframe: pd.DataFrame = pd.read_csv(patient_dataset_path)

        self.dataframe_copy: pd.DataFrame = self.dataframe.copy()

        #sort row by timestamp
        self.dataframe_copy = self.dataframe_copy.sort_values(by = 'timestamp').reset_index(drop=True)

        #drop some columns that will be unused
        #self.dataframe_copy.drop(['filename', 'interval_index', 'timestamp', 'set'], axis=1, inplace=True)

    def impute_nan_and_infinite_values(self):

        """
        This function will replace infinite values by nan. 
        Because of many nan values can follow one another in an entire window, 
        we choose to impute nan values by global mean in order to avoid extrems values.
        """
        if np.isinf(self.dataframe_copy).sum().any() > 0:
            self.dataframe_copy.replace([np.inf, -np.inf], np.nan, inplace=True)

        X = self.dataframe_copy.drop(['label'], axis=1)
        Y = self.dataframe_copy['label']
        X_imputed = None

        if self.dataframe_copy.isna().sum().any() :
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X)
            X_imputed = imputer.transform(X)

        return X_imputed, Y

    def outlier_detection(self, X):

        """
        This function detect outlier based on local density of points. It will identify
        point with low local density compared with their neighboors and remove them.
        """

        # Detection of outliers
        LOF = LocalOutlierFactor()
        Y_pred = LOF.fit_predict(X)
        X_score = LOF.negative_outlier_factor_

        outlier_score = pd.DataFrame()
        outlier_score['score'] = X_score

        X_init = self.dataframe_copy.drop(['label'], 1)
        Y_init = self.dataframe_copy['label']
        dataframe = pd.DataFrame(X, columns=X_init.columns)

        ## drop outliers values
        filt = outlier_score["score"] < -1.5
        outlier_index = outlier_score[filt].index.tolist()
        new_X = pd.DataFrame(dataframe).drop(outlier_index)
        new_Y = Y_init.drop(outlier_index).values

        return new_X, new_Y

    
    def pca_analysis(self, dataframe):

        """
        This function resolve the problem of multicolinearity in dataset. Some features are too
        correlated. This can be redundant information that we need to remove.
        """
        #normalisation step
        dataframe = zscore(dataframe)


        pca = PCA(n_components=self.objectif_variance)
        pca_out = pca.fit(dataframe)
        principalComponents = pca.transform(dataframe)

        #construction of dataset
        new_size_dataset = pca_out.components_.shape[0]
        list_colmuns = []
        for i in range(1, new_size_dataset):
            list_colmuns.append("PC"+i)
        
        pca_df = pd.DataFrame(principalComponents, columns=list_colmuns)
        pca_df['label'] = self.Y

        return pca_df


        

