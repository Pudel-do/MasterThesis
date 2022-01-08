# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:37:04 2021

@author: Matthias Pudel
"""

import pandas as pd
from DataPreparation import DataPreparation
from FeatureEngineering import FeatureEngineering
from Regression import Regression
from GetData import *
from Visualization import *
import warnings
# warnings.filterwarnings("ignore")

# ===========================================
# Empirical Evidence - Parameter and Settings
# ===========================================

# =========================
# General Settings
# =========================
start_date = pd.Timestamp('2000-01-01')
end_date = pd.Timestamp('2019-12-31')
start_year = start_date.strftime('%Y')
end_year = end_date.strftime('%Y')
# =========================
# Data Preparation
# =========================
adj_data_prep = False
adj_uw_matching = False
adj_time_range = False
adj_close_price = False
industry_treshold = 0
# =========================
# Feature Engineering
# =========================
adj_feat_eng = True  
adj_public_proxies = True
adj_raw_prospectuses = False
adj_prospectus_analysis = False
plot_outliers = False 
index_weight = 'Equal'
port_days = 15
scale_factor = 100
whisker_factor = 15
# =========================
# Regression
# =========================
clean_file = False
adj_reg_cols = False 
reg_vars = [
    'InitialReturn', 
    'UnderwriterRank', 
    'TotalAssets',
    'TechDummy',
    'VentureDummy',
    # 'AMEX', 
    # 'NASDQ', 
    # 'NYSE',
    'MarketReturn', 
    # 'MarketReturnSlopeDummy',
    'SectorReturn', 
    'SectorReturnSlopeDummy',
    'WordsRevisionDummy',
    'PriceRevision', 
    # 'PriceRevisionSlopeDummy',
    'PriceRevisionMaxDummy',
    # 'PriceRevisionMaxSlopeDummy', 
    # 'PriceRevisionMinDummy',
    # 'PriceRevisionMinSlopeDummy',
    # 'SharesRevision', 
    # 'SharesRevisionSlopeDummy',
    # 'ProceedsRevision', 
    # 'ProceedsRevisionSlopeDummy',
    # 'ProceedsRevisionMaxDummy',
    # 'ProceedsRevisionMaxSlopeDummy', 
    # 'ProceedsRevisionMinDummy',
    # 'ProceedsRevisionMinSlopeDummy',
    ]
# ===========================================
if adj_data_prep == True:
    prep_obj = DataPreparation(start_date, end_date)
    prep_obj.rough_preprocessing(adj_time_range)
    prep_obj.build_aux_vars(industry_treshold)
    prep_obj.extended_preprocessing(adj_uw_matching)
    prep_obj.data_merging(adj_close_price)
    obj_file = prep_obj_file
    obj_file = f'{start_year}_{end_year}_{obj_file}'
    save_obj(prep_obj, output_path, obj_file)
else:
    obj_file = prep_obj_file
    obj_file = f'{start_year}_{end_year}_{obj_file}'
    prep_obj = get_object(output_path, obj_file)
    
if adj_feat_eng == True:
    feat_eng_obj = FeatureEngineering(prep_obj, scale_factor)
    feat_eng_obj.preprocessing(adj_raw_prospectuses)
    feat_eng_obj.firm_features()
    feat_eng_obj.public_features(index_weight, port_days, adj_public_proxies)
    feat_eng_obj.private_features(adj_prospectus_analysis)
    feat_eng_obj.outlier_adjustment(whisker_factor, plot_outliers)
    obj_file = feat_eng_obj_file
    obj_file = f'{start_year}_{end_year}_{obj_file}'
    save_obj(feat_eng_obj, output_path, obj_file)
else:
    obj_file = feat_eng_obj_file
    obj_file = f'{start_year}_{end_year}_{obj_file}'
    feat_eng_obj = get_object(output_path, obj_file)
    
reg_obj = Regression(feat_eng_obj, start_year, end_year)
reg_obj.preprocessing(reg_vars)
reg_obj.ols_regression()
reg_obj.fmb_regression(start_date, end_date)
reg_obj.build_results(adj_reg_cols, clean_file)
obj_file = reg_obj_file
obj_file = f'{start_year}_{end_year}_{obj_file}'
save_obj(reg_obj, output_path, obj_file)
# ===========================================
full_data = feat_eng_obj.full_data
SectorDistribution(prep_obj, True)
DescStat_ProspectusAnalysis(feat_eng_obj)
DescStat_RegressionSample(reg_obj)
RegressionResults(reg_obj)

