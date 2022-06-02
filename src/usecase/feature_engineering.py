"""
Feature engineering process.

Write some texte after
"""

import os
import string

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore

class Feature_engineering:
    
    def __init__(
        self,
        patient_dataset_path: string,
        objectif_variance: float
    ) -> None:

        #read file in dataframe from path patient dataset
        self.dataframe = pd.read_csv(self.patient_dataset_path)

        #sort row by timestamp
        self.dataframe = self.dataframe.sort_values(by = 'timestamp').reset_index(drop=True)

        self.dataframe_copy = self.dataframe.copy()

        #drop some columns that will be unused
        self.dataframe_copy.drop(['filename', 'interval_index'], axis=1, inplace=True)

        self.X = None
        self.Y = None
        self.X_imputed = None


    def impute_nan_and_infinite_values (self):

        """
        This function will replace infinite values by nan. 
        Because of many nan values that follow one another in an entire window, 
        we choose to impute nan values by global mean in order to avoid extrems values.
        """

        self.dataframe_copy.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.X = self.dataframe_copy.drop(['label'], axis=1)
        self.Y = self.dataframe_copy['label']

        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(self.X)
        self.X_imputed = imputer.transform(self.X)


    def outlier_detection (self):

        """
        This function detect outlier based on local density of points. It will identify
        point with low local density compared with their neighboors will be removed.
        """

        # Detection of outliers
        LOF = LocalOutlierFactor()
        Y_pred = LOF.fit_predict(self.X_imputed)
        X_score = LOF.negative_outlier_facto

        outlier_score = pd.DataFrame()
        outlier_score['score'] = X_score

        self.X_imputed = pd.DataFrame(self.X_imputed, columns=self.X.columns)

        ## drop outliers values
        filt = outlier_score["score"] < -1.5
        outlier_index = outlier_score[filt].index.tolist()
        self.X_imputed = pd.DataFrame(self.X_imputed).drop(outlier_index)
        self.Y = self.Y.drop(outlier_index).values

    
    def pca_analysis(self):

        """
        This function resolve the problem of multicolinearity in dataset. Some features are too
        correlated. This can be redundant information that we need to remove.
        """
        #normalisation step
        self.X_imputed = zscore(self.X_imputed)


        pca = PCA(n_components=self.objectif_variance)
        pca_out = pca.fit(self.X_imputed)
        principalComponents = pca.transform(self.X_imputed)

        

