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
# ===========================================
# Prediction Model - Parameter and Settings
# ===========================================F

if adj_preprocess == True:
    prep_obj = DataPreparation(start_date, end_date)
    prep_obj.rough_preprocessing()
    prep_obj.build_aux_vars(adj_time_range)
    prep_obj.extended_preprocessing(adj_uw_matching)
    prep_obj.data_merging(adj_close_price)
    save_obj(prep_obj, output_path, prep_obj_file)
else:
    prep_obj = get_object(output_path, prep_obj_file)
    
if adj_feat_eng == True:
    feat_eng_obj = FeatureEngineering(prep_obj, scale_factor)
    feat_eng_obj.preprocessing()
    feat_eng_obj.firm_features()
    feat_eng_obj.public_features(index_weight, port_days, adj_public_feat)
    feat_eng_obj.private_features()
    save_obj(feat_eng_obj, output_path, feat_eng_obj_file)
    full_data = feat_eng_obj.full_data
    model_data = feat_eng_obj.model_data
else:
    feat_eng_obj = get_object(output_path, feat_eng_obj_file)
    full_data = feat_eng_obj.full_data
    model_data = feat_eng_obj.model_data
    
x = model_data.isna().sum()
    