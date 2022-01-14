# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:03:47 2022

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

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
    def __init__(self, reg_obj):
        self.reg_obj = reg_obj
        self.model_raw_data = reg_obj.model_data
        
    def preprocessing(self, target_return, adj_preprocessing):
        if adj_preprocessing == True:
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
            target = model_data_adj['Target']
            regressors = model_data_adj.drop(columns = 'Target')
            model_set = {}
            test_sets = {}
            for i in range(5):
                x_train, x_test, y_train, y_test = train_test_split(regressors, 
                                                                    target, 
                                                                    test_size=0.25)
                
                if i == 0:
                    model_set['x_test'] = x_test
                    model_set['y_test'] = y_test
                    model_set['x_train'] = x_train
                    model_set['y_train'] = y_train
                else:
                    test_sets[f'x_test_{i}'] = x_test
                    test_sets[f'y_test_{i}'] = y_test
# =========================
                    
                print('Test')
                
            
        