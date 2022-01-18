# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:03:47 2022

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow import keras 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from GetData import *
from Visualization import *

pred_cols = [
    'InitialReturn', 'InitialReturnAdjusted',
    'UnderwriterRank', 'TotalAssets', 
    'TechDummy', 'VentureDummy', 
    'AMEX', 'NASDQ', 'NYSE',
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
        self.n_sub_testset = 5
        
    def preprocessing(self, target_return, adj_preprocessing, use_feature_selection):
        start_year = self.start_year
        end_year = self.end_year
        model_obj_file = f'{start_year}_{end_year}_{pred_model_set_file}'
        test_sets_obj_file = f'{start_year}_{end_year}_{pred_test_sets_file}'
        
        model_data = self.model_raw_data[pred_cols]
        model_data = model_data.dropna()
        
        rets_cols = ['InitialReturn', 'InitialReturnAdjusted']
        returns = model_data[rets_cols]
        target_rets = np.where(returns >= target_return,1,0)
        target_rets = pd.DataFrame(target_rets,
                                   index = returns.index,
                                   columns = rets_cols)

        target_ret_col = 'InitialReturnAdjusted'
        model_data['Target'] = target_rets[target_ret_col]
        model_data = model_data.drop(columns = rets_cols)
        model_cols = pred_cols.copy()
        model_cols.remove('InitialReturn')
        model_cols.remove('InitialReturnAdjusted')
        model_cols.append('Target')
        
        self.target_ret_col = target_ret_col
        self.target_rets = target_rets
        self.model_data = model_data
        self.model_cols = model_cols
# =========================
        if adj_preprocessing == True:
            scaler = StandardScaler()
            unscale_cols = ['Dummy', 'NASDQ', 'AMEX', 'NYSE', 'Target']
            scale_cols = []
            for var in model_cols:
                if not any(char in var for char in unscale_cols):
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
            feat_treshold = 0.01
            rand_stat = None
            selector = RandomForestClassifier(n_estimators = 500)
            sampler = RandomOverSampler(sampling_strategy = sample_strat, 
                                        random_state = rand_stat)

            target = model_data_adj['Target']
            regressors = model_data_adj.drop(columns = 'Target')
            model_set = {}
            test_sets = {}
            for i in range(self.n_sub_testset+1):
                x_train, x_test, y_train, y_test = train_test_split(regressors, 
                                                                    target, 
                                                                    test_size=0.15,
                                                                    random_state = rand_stat)
                
                x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, 
                                                                      test_size=0.25,
                                                                      random_state = rand_stat)
                
                x_train, y_train = sampler.fit_resample(x_train, y_train)
                x_valid, y_valid = sampler.fit_resample(x_valid, y_valid)
                y_train_dist = y_train.value_counts(normalize=True)
                y_valid_dist = y_valid.value_counts(normalize=True)
                self.y_train_dist = y_train_dist
                self.y_valid_dist = y_valid_dist
                                
                if i == 0:
                    if use_feature_selection == True:
                        selector.fit(x_train, y_train)
                        feat_weights = selector.feature_importances_
                        feat_weights = pd.Series(feat_weights)
                        feat_weights.index = x_train.columns
                        feat_weights = feat_weights.sort_values(ascending = False)
                        features = feat_weights.where(feat_weights>feat_treshold)
                        features = features.dropna()
                        features = features.index.to_list()
                        self.feature_weights = feat_weights
                    else:
                        features = x_train.columns.to_list()
                        
                    model_set['x_test'] = x_test[features]
                    model_set['y_test'] = y_test
                    model_set['x_train'] = x_train[features]
                    model_set['y_train'] = y_train
                    model_set['x_valid'] = x_valid[features]
                    model_set['y_valid'] = y_valid
                    
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

    def model_training(self, adj_model_training, adj_benchmark_training, benchmark):
        self.benchmark = benchmark
        model_set = self.model_set
        x_train = model_set['x_train']
        y_train = model_set['y_train']
        x_valid = model_set['x_valid']
        y_valid = model_set['y_valid']
        
        if adj_model_training == True:
            input_shape = x_train.shape[1]
            model = keras.models.Sequential([
                keras.layers.Flatten(input_shape = [input_shape]),
                keras.layers.Dense(300, activation = 'relu'),
                keras.layers.Dense(100, activation = 'relu'),
                keras.layers.Dense(1, activation = 'sigmoid')
                ])
            
            early_stopping = keras.callbacks.EarlyStopping(patience=30,
                                                           restore_best_weights=True)
            
            model.compile(loss='binary_crossentropy',
                          optimizer='sgd',
                          metrics=['accuracy'])
            
            history = model.fit(x_train, y_train,
                                validation_data=(x_valid, y_valid),
                                callbacks=[early_stopping],
                                epochs=100)
            
            adj_cols = {'loss': 'Loss_Training',
                        'acc': 'Accuracy_Training',
                        'val_loss': 'Loss_Validation',
                        'val_acc': 'Accuracy_Validation'}
            train_performance = pd.DataFrame(history.history)
            train_performance = train_performance.rename(columns = adj_cols)
            labels = train_performance.columns
            plt.figure(figsize=figsize)
            plt.plot(train_performance)
            plt.grid(True)
            plt.legend(labels=labels, fontsize = legend_size)
            plt.xlabel('Epochs', fontdict = xlabel_size)
            plt.ylabel('Value', fontdict = ylabel_size)
            title = 'Training performance of neural network'
            plt.title(title, fontdict = title_size)
            plt.show()
            
            model.save(output_path+'\\'+trained_neural_network_file)
            
        if adj_benchmark_training == True:
            classifier = benchmark_classifier[benchmark]
            param_grid = benchmark_param_grids[benchmark]

            
            benchmark_model = GridSearchCV(classifier,
                                           param_grid,
                                           cv=5,
                                           scoring='accuracy')
            
            benchmark_model.fit(x_train, y_train)
            
            self.benchmark_model = benchmark_model
            benchmark_file = trained_benchmark_file
            benchmark_file = f'{self.benchmark}_{benchmark_file}'
            save_obj(benchmark_model, output_path, benchmark_file)
            
    def model_evaluation(self):
        model_set = self.model_set
        sub_test_sets = self.test_sets
        x_test = model_set['x_test']
        y_test = model_set['y_test']
        model = keras.models.load_model(output_path+'\\'+trained_neural_network_file)
        benchmark_file = trained_benchmark_file
        benchmark_file = f'{self.benchmark}_{benchmark_file}'
        benchmark_model = get_object(output_path, benchmark_file)
        
        y_pred = model.predict_classes(x_test)
        y_pred = y_pred.flatten()
        y_pred = pd.Series(y_pred)
        y_pred.name = 'Prediction'
        y_pred.index = y_test.index
        model_prediction = pd.concat([y_test, y_pred], axis=1)
        
        benchmark_estimator = benchmark_model.best_estimator_
        y_pred_bench = benchmark_estimator.predict(x_test)
        y_pred_bench = y_pred_bench.flatten()
        y_pred_bench = pd.Series(y_pred_bench)
        y_pred_bench.name = 'Prediction'
        y_pred_bench.index = y_test.index
        benchmark_prediction = pd.concat([y_test, y_pred_bench], axis=1)
        
        self.model_prediction = model_prediction
        self.benchmark_model = benchmark_model
        self.benchmark_prediction = benchmark_prediction
        
        sub_results = pd.DataFrame()
        for i in range(1, self.n_sub_testset+1):
            x_test = sub_test_sets[f'x_test_{i}']
            y_test = sub_test_sets[f'y_test_{i}']
            y_pred = model.predict_classes(x_test)
            
            score = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            
            sub_results.loc[i, 'Accuracy'] = score
            sub_results.loc[i, 'Precision'] = precision
            sub_results.loc[i, 'Recall'] = recall
        
        mean = sub_results.mean()
        mean = mean.transpose()
        sub_results.loc['Mean', :] = mean
        self.model_sub_results = sub_results
            
            