# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:37:04 2021

@author: Matthias Pudel
"""

import pandas as pd
import warnings
from GetData import *
warnings.filterwarnings("ignore")

# ===========================================
# Empirical Evidence - Parameter and Settings
# ===========================================
start_date = pd.Timestamp('2000-01-01')
end_date = pd.Timestamp('2019-12-31')
start_year = start_date.strftime('%Y')
end_year = end_date.strftime('%Y')

from DataPreparation import *
adj_preprocess = False
adj_uw_matching = False
adj_time_range = False
adj_close_price = False

from FeatureEngineering import *
adj_feat_eng = False
adj_public_feat = False
index_weight = 'Equal'
port_days = 15
scale_factor = 100

from Regression import *
clean_file = False
adj_reg_cols = True
reg_cols = [
    'InitialReturn', 
    'UnderwriterRank', 'TotalAssets',
    'TechDummy', 'AMEX', 'NASDQ', 'NYSE',
    'MarketReturn', 
    'MarketReturnSlopeDummy',
    'SectorReturn', 
    'SectorReturnSlopeDummy',
    'PriceRevision', 
    # 'PriceRevisionSlopeDummy',
    'PriceRevisionMax', 
    'PriceRevisionMin',
    # 'SharesRevision', 
    # 'SharesRevisionSlopeDummy',
    'ProceedsRevision', 
    # 'ProceedsRevisionSlopeDummy',
    'ProceedsRevisionMax', 
    'ProceedsRevisionMin',
    ]
# ===========================================
# Prediction Model - Parameter and Settings
# ===========================================

if adj_preprocess == True:
    prep_obj = DataPreparation(start_date, end_date)
    prep_obj.rough_preprocessing(adj_time_range)
    prep_obj.build_aux_vars()
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
    feat_eng_obj.preprocessing()
    feat_eng_obj.firm_features()
    feat_eng_obj.public_features(index_weight, port_days, adj_public_feat)
    feat_eng_obj.private_features()
    obj_file = feat_eng_obj_file
    obj_file = f'{start_year}_{end_year}_{obj_file}'
    save_obj(feat_eng_obj, output_path, obj_file)
else:
    obj_file = feat_eng_obj_file
    obj_file = f'{start_year}_{end_year}_{obj_file}'
    feat_eng_obj = get_object(output_path, obj_file)
    full_data = feat_eng_obj.full_data
    model_data = feat_eng_obj.model_data
    
reg_obj = Regression(feat_eng_obj, start_year, end_year)
reg_obj.preprocessing(reg_cols)
reg_obj.ols_regression()
reg_obj.fmb_regression(start_date, end_date)
reg_obj.build_results(adj_reg_cols, clean_file)
obj_file = reg_obj_file
obj_file = f'{start_year}_{end_year}_{obj_file}'
save_obj(reg_obj, output_path, obj_file)

desc_stat = descriptive_statistic(reg_obj)

    