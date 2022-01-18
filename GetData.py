# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 19:57:37 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
warnings.filterwarnings("ignore")

exit_message = 'Download of CRSP data for quotes and returns necessary due to time period adjustments'
exit_message2 = 'Adjustment mode for explanatory variables is active. Select variables to consider. Otherwise set adjustment parameter to False for result generation.'

output_path = r'C:\Users\Matthias Pudel\OneDrive\Studium\Master\Master Thesis\Empirical Evidence\Output Data'
prep_obj_file = 'DataPreparation.pkl'
feat_eng_obj_file = 'FeatureEngineering.pkl'
reg_obj_file = 'Regression.pkl'
pred_obj_file = 'PredictionModel.pkl'
uw_matching_file = 'UnderwriterMatchResults.csv'
public_feat_file = 'PublicFeatures.csv'
ols_result_file = 'RegressionResult_OLS.csv'
ols_keyfig_file = 'KeyFigures_OLS.csv'
ols_aggresult_file = 'AggregatedRegressionResult_OLS.csv'
ols_aggkeyfig_file = 'AggregatedKeyFigures_OLS.csv'
fmb_result_file = 'RegressionResult_FMB.csv'
fmb_keyfig_file = 'KeyFigures_OLS.csv'
fmb_aggresult_file = 'AggregatedRegressionResult_FMB.csv'
fmb_aggkeyfig_file = 'AggregatedKeyFigures_FMB.csv'
prosp_result_file = 'TextAnalysisResults.csv'
master_dict_log_file = 'MasterDictionaryLogFile.txt'
sector_port_result_file = 'SectorPortfolioResults.csv'
pred_model_set_file = 'PredictionModelSet.pkl'
pred_test_sets_file = 'PredictionTestSets.pkl'
trained_neural_network_file = 'TrainedNeuralNetwork.h5'
trained_benchmark_file = 'TrainedBenchmarkModel.pkl'


input_path = r'C:\Users\Matthias Pudel\OneDrive\Studium\Master\Master Thesis\Empirical Evidence\Input Data'
raw_prospectus_path = r'D:\OneDrive_Backup\Master\MasterThesis\ProspectusData'
sdc_raw_file = 'sdc_full_raw.csv'
total_assets_file = 'sdc_total_assets.csv'
quotes_file = 'IPO_Quotes.csv'
uw_file = 'UnderwriterRank.xlsx'
cpi_file = 'CPI_80_21.xlsx'
returns_file = 'IPO_Returns.csv'
index_returns_file = 'CRSP_Market_Returns.csv'
prosp_merge_file = '100_IPO_data_merged_by_DealNumber_without_any_exclusions.csv'
master_dict_file = 'LoughranMcDonald_MasterDictionary_2020.csv'
industry_dummies_file = 'FamaFrenchIndustryDummies.xlsx'


benchmark_param_grids = {
    'RandomForest': [
        {'n_estimators': [50, 100, 150, 200],
         'max_features': [4, 6, 8, 15]},
        {'bootstrap': [False],
         'n_estimators': [100, 200],
         'max_features': [6, 15]},
         ],
    'SVC': [
        {'kernel': ('sigmoid', 'rbf','poly'),
         'C':[1.5, 10]},
        {'kernel': ('sigmoid', 'poly'),
         'C':[1, 4, 8]},
         ],
    }

benchmark_classifier = {
    'RandomForest': RandomForestClassifier(),
    'SVC': SVC()
    }


def save_obj(obj, path, filename):
    with open(path + '\\' + filename, 'wb') as f:
        pickle.dump(obj, f)
        
def get_object(path, filename):
    with open(path + '\\' + filename, 'rb') as f:
        return pickle.load(f)
    
def get_SlopeDummy(base):
    slope_dummy = np.where(base > 0, base, 0)
    slope_dummy = pd.Series(slope_dummy, index = base.index)
        
    nan_ident = pd.isnull(base) == True
    nan_idx = base.loc[nan_ident].index
    slope_dummy.loc[nan_idx] = np.nan
    
    return slope_dummy
    
def get_DummyMax(target, base):
    revision = (target / base) -1
    dummy = np.where(revision > 0, 1, 0)
    dummy = pd.Series(dummy, index = revision.index)
    
    nan_ident = pd.isnull(revision) == True
    nan_idx = revision.loc[nan_ident].index
    dummy.loc[nan_idx] = np.nan

    return dummy
        
def get_DummyMin(target, base):
    revision = (target / base) -1
    dummy = np.where(revision < 0, 1, 0)
    dummy = pd.Series(dummy, index = revision.index)
    
    nan_ident = pd.isnull(revision) == True
    nan_idx = revision.loc[nan_ident].index
    dummy.loc[nan_idx] = np.nan
    
    return dummy

def get_SlopeDummyBounds(base, value):
    slope_dummy = np.where(base == 1, base*value, 0)
    slope_dummy = pd.Series(slope_dummy, index = base.index)
        
    nan_ident = pd.isnull(base) == True
    nan_idx = base.loc[nan_ident].index
    slope_dummy.loc[nan_idx] = np.nan
    
    return slope_dummy