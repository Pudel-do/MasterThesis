# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:03:47 2022

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from GetData import *

pred_cols = [
    'InitialReturn', 'InitialReturnAdjusted',
    'UnderwriterRank', 'TotalAssets', 'TechDummy', 
    'VentureDummy', 'AMEX', 'NASDQ', 'NYSE',
    'ExpectedProceeds', 'ActualProceeds',
    'SectorVolume', 'Volume',
    'MarketReturn', 'SectorReturn',
    'PriceRevision', 'RegistrationDays',
    'PriceRevisionMaxDummy', 'PriceRevisionMinDummy',
    'PositiveWordsRevision', 'NegativeWordsRevision',
    'SharesRevision', 
    ]

class PredictionModel:
    def __init__(self, reg_obj, start_year, end_year):
        self.reg_obj = reg_obj
        self.model_raw_data = reg_obj.model_data
        self.start_year = start_year
        self.end_year = end_year
        
    def preprocessing(self, target_return, adj_preprocessing, use_feature_selection):
        start_year = self.start_year
        end_year = self.end_year
        model_obj_file = f'{start_year}_{end_year}_{pred_model_set}'
        test_sets_obj_file = f'{start_year}_{end_year}_{pred_test_sets}'
        
        model_data = self.model_raw_data[pred_cols]
        model_data = model_data.dropna()
        
        rets_cols = ['InitialReturn', 'InitialReturnAdjusted']
        returns = model_data[rets_cols]
        target_rets = np.where(returns >= target_return,1,0)
        target_rets = pd.DataFrame(target_rets,
                                   index = returns.index,
                                   columns = rets_cols)

        model_data['Target'] = target_rets['InitialReturnAdjusted']
        model_data = model_data.drop(columns = rets_cols)
        model_cols = pred_cols.copy()
        model_cols.remove('InitialReturn')
        model_cols.remove('InitialReturnAdjusted')
        model_cols.append('Target')
        
        self.target_rets = target_rets
        self.model_data = model_data
        self.model_cols = model_cols
# =========================
        if adj_preprocessing == True:
            scaler = MinMaxScaler()
            drop_chars = ['Dummy', 'NASDQ', 'AMEX', 'NYSE', 'Target']
            scale_cols = []
            for var in model_cols:
                if not any(char in var for char in drop_chars):
                    scale_cols.append(var)

            scaling = scaler.fit(model_data[scale_cols])
            scaled_data = scaling.transform(model_data[scale_cols])
            scaled_data = pd.DataFrame(scaled_data,
                                       index = model_data.index,
                                       columns = scale_cols)
            
            unscaled_data = model_data.drop(columns = scale_cols)
            model_data_adj = scaled_data.join(unscaled_data)
# =========================
            sample_strat = 1
            sampler = RandomOverSampler(sampling_strategy = sample_strat)
            feat_treshold = 0.01
            
            target = model_data_adj['Target']
            regressors = model_data_adj.drop(columns = 'Target')
            model_set = {}
            test_sets = {}
            for i in range(5):
                x_train, x_test, y_train, y_test = train_test_split(regressors, 
                                                                    target, 
                                                                    test_size=0.25)
                
                x_train, y_train = sampler.fit_resample(x_train, y_train)
                y_train_dist = y_train.value_counts(normalize=True)
                self.y_train_dist = y_train_dist
                                
                if i == 0:
                    if use_feature_selection == True:
                        rnd_clf = RandomForestClassifier(n_estimators = 500)
                        rnd_clf.fit(x_train, y_train)
                        feat_weights = rnd_clf.feature_importances_
                        feat_weights = pd.Series(feat_weights)
                        feat_weights.index = x_train.columns
                        features = feat_weights.where(feat_weights>=feat_treshold)
                        features = features.dropna()
                        features = features.index.to_list()
                    else:
                        features = x_train.columns.to_list()
                        
                    model_set['x_test'] = x_test[features]
                    model_set['y_test'] = y_test
                    model_set['x_train'] = x_train[features]
                    model_set['y_train'] = y_train
                else:
                    test_sets[f'x_test_{i}'] = x_test[features]
                    test_sets[f'y_test_{i}'] = y_test
                

            save_obj(model_set, output_path, model_obj_file)
            save_obj(test_sets, output_path, test_sets_obj_file)   
            self.model_set = model_set
            self.test_sets = test_sets
        else:
            model_set = get_object(output_path, model_obj_file)
            test_sets = get_object(output_path, test_sets_obj_file)
            self.model_set = model_set
            self.test_sets = test_sets


                
            
        