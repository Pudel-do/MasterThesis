# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 19:57:37 2021

@author: Matthias Pudel
"""
import pickle
import pandas as pd
import numpy as np

exit_message = 'Download of CRSP data for quotes and returns necessary due to time period adjustments'
exit_message2 = 'Adjustment mode for explanatory variables is active. Select variables to consider. Otherwise set adjustment parameter to False for result generation.'

output_path = r'C:\Users\Matthias Pudel\OneDrive\Studium\Master\Master Thesis\Empirical Evidence\Code\Output Data'
prep_obj_file = 'DataPreparation.pkl'
feat_eng_obj_file = 'FeatureEngineering.pkl'
reg_obj_file = 'Regression.pkl'
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


input_path = r'C:\Users\Matthias Pudel\OneDrive\Studium\Master\Master Thesis\Empirical Evidence\Code\Input Data'
sdc_raw_file = 'sdc_full_raw.csv'
total_assets_file = 'sdc_total_assets.csv'
quotes_file = 'IPO_Quotes.csv'
uw_file = 'UnderwriterRank.xlsx'
cpi_file = 'CPI_80_21.xlsx'
returns_file = 'IPO_Returns.csv'
index_returns_file = 'CRSP_Market_Returns.csv'


def save_obj(obj, path, filename):
    with open(path + '\\' + filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def get_object(path, filename):
    with open(path + '\\' + filename, 'rb') as f:
        return pickle.load(f)
    
def get_slope_dummy(base):
    slope_dummy = np.where(base > 0, base, 0)
    slope_dummy = pd.Series(slope_dummy, index = base.index)
        
    nan_ident = pd.isnull(base) == True
    nan_idx = base.loc[nan_ident].index
    slope_dummy.loc[nan_idx] = np.nan
    
    return slope_dummy
    
def get_dummy_max(target, base):
    revision = (target / base) -1
    dummy = np.where(revision > 0, 1, 0)
    dummy = pd.Series(dummy, index = revision.index)
    
    nan_ident = pd.isnull(revision) == True
    nan_idx = revision.loc[nan_ident].index
    dummy.loc[nan_idx] = np.nan

    return dummy
        
def get_dummy_min(target, base):
    revision = (target / base) -1
    dummy = np.where(revision < 0, 1, 0)
    dummy = pd.Series(dummy, index = revision.index)
    
    nan_ident = pd.isnull(revision) == True
    nan_idx = revision.loc[nan_ident].index
    dummy.loc[nan_idx] = np.nan

    return dummy