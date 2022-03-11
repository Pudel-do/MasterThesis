# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 10:31:16 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import statsmodels.stats.weightstats as smw
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tabulate import tabulate 
import re 
from re import search
from GetData import *

output_path_results = r'C:\Users\Matthias Pudel\OneDrive\Studium\Master\Master Thesis\Empirical Evidence\Results'

figsize = (25,10)
xlabel_size = {'fontsize' : 15}
ylabel_size = {'fontsize' : 15}
suptitle_size = 25
title_size = {'fontsize' : 25}
legend_size = 'xx-large'
hist_bins = 75
cutting_line = 135*'='
cutting_line_thin = 135*'-'
paragraph = '\n\n\n\n'

def WritePlotData(feat_eng_obj, write_plot_data):
    obj = feat_eng_obj
    raw_data = obj.model_data
    
    hist_data_cols = ['MarketReturn', 'SectorReturn']
    hist_data = raw_data[hist_data_cols]
    hist_data = hist_data.round(2)
    if write_plot_data == True:
        file_name = 'ReturnHistograms.xlsx'
        hist_data.to_excel(output_path_plots+'\\'+file_name)
        

def Analysis_Underpricing(feat_eng_obj, plot_initial_return_analysis):
    obj = feat_eng_obj
    raw_data = obj.full_data
    data = raw_data[['InitialReturn', 'IssueDate']]
    data['Year'] = data['IssueDate'].dt.strftime('%Y')
    data['Year'] = data['Year']
    data = data.drop(columns=['IssueDate'])
    data = data.dropna()
    grouped_return = data.groupby(['Year']).mean()
    grouped_ipos = data.groupby(['Year']).size()
    grouped_ipos = pd.DataFrame(grouped_ipos)
    grouped_ipos.columns = ['Count_IPO']
    grouped_data = grouped_return.join(grouped_ipos)
    
    file_name = 'TimeSeries_Underpricing.xlsx'
    grouped_data.to_excel(output_path_plots+'\\'+file_name)
    
    if plot_initial_return_analysis == True:
        fig, ax1 = plt.subplots(figsize = figsize)
        plt.bar(grouped_data.index, grouped_data['InitialReturn'],
                label='Initial Return')
        plt.legend(loc=1)
        plt.xlabel('Date', fontdict = xlabel_size)
        plt.ylabel('Initial Return', fontdict = ylabel_size)
        plt.title('Time series of initial returns and IPO volume', 
                  fontdict = title_size)
        ax2 = ax1.twinx()
        plt.plot(grouped_data['Count_IPO'], 
                 'g', lw=1.5, label='IPO Volume')
        plt.legend(loc=5)
        plt.ylabel('IPO Volume')

def Performance_PredictionModel(obj, plot_prediction_performance):
    raw_data = obj.model_raw_data
    target_col = obj.target_ret_col
    model_pred = obj.model_prediction
    benchmark = obj.benchmark
    benchmark_pred = obj.benchmark_prediction
    
    models = ['NeuralNetwork', benchmark]
    for model in models:
        if model == 'NeuralNetwork':
            pred_data = model_pred
        else:
            pred_data = benchmark_pred
            
        y_true = pred_data['Target']
        y_pred = pred_data['Prediction']
        report = classification_report(y_true, y_pred)
        
        if plot_prediction_performance == True:
            print(cutting_line)
            print(f'Test set performance of {model} model')
            print(cutting_line)
            print(report)
            print(cutting_line, paragraph)

    returns = raw_data[target_col]
    prediction = model_pred['Prediction']
    port_comps = prediction.where(prediction==1)
    port_comps = port_comps.dropna()
    n_port_comps = len(port_comps)
    port_rets = returns.loc[port_comps.index]
    port_ret = port_rets.mean()
    port_std = port_rets.std()
    
    benchmark_rets = pd.Series([])
    benchmark_stds = pd.Series([])
    sample_number = 100
    for i in range(sample_number):
        bench_comps = model_pred.sample(n = n_port_comps)
        bench_rets = returns.loc[bench_comps.index]
        bench_ret = bench_rets.mean()
        bench_std = bench_rets.std()
        benchmark_rets.loc[i] = bench_ret
        benchmark_stds.loc[i] = bench_std
        
    benchmark_ret = benchmark_rets.mean()
    benchmark_std = benchmark_stds.mean()
    
    cols = ['Portfolio', 'Benchmark']
    idx = ['Return', 'StandardDeviation']
    result = pd.DataFrame(columns = cols,
                          index = idx)
    result.loc[idx[0], cols[0]] = port_ret
    result.loc[idx[1], cols[0]] = port_std
    result.loc[idx[0], cols[1]] = benchmark_ret
    result.loc[idx[1], cols[1]] = benchmark_std
    result = result.astype(float)
    
    plot_result = tabulate(result,
                           headers = 'keys',
                           floatfmt = '.2f',
                           tablefmt = 'simple',
                           numalign = 'center',
                           showindex = True)
    
    if plot_prediction_performance == True:
        print(cutting_line)
        print(f'Portfolio performance of predicted and randomly selected IPOs')
        print(cutting_line, '\n')
        print(plot_result)
        print(cutting_line_thin)
        print(f'Number of portfolio components: {n_port_comps}')
        print(f'Number of samples for benchmark construction: {sample_number}')
        print(cutting_line, paragraph)
    

def Analysis_SectorPortfolio(obj, plot_sectorportfolio):
    data = obj.sector_portfolio_results
    sector_id = obj.port_division
    industry = data['Division']
    industry_dist = industry.value_counts(normalize=True)
    industry_dist = industry_dist * 100
    port_comps = data.groupby(['Year'])
    port_comps = port_comps.sum()
    port_comps = port_comps.reset_index()
    time_range = port_comps['Year']
    time_range = time_range.astype(str)
    bar_data = port_comps['PortfolioComponents']
    if plot_sectorportfolio == True:
        plt.figure(figsize = figsize)
        plt.subplot(121)
        labels = industry_dist.index
        plt.pie(industry_dist, 
                labels=labels, 
                autopct='%.2f%%',
                )
        title = 'Division distribution of sector portfolio'
        plt.title(title, fontdict = title_size)
        plt.subplot(122)
        plt.bar(time_range, bar_data)
        plt.xlabel('Year', fontdict = xlabel_size)
        plt.ylabel('Portfolio components', fontdict = ylabel_size)
        title = 'Sum of portfolio components per year'
        plt.title(title, fontdict = title_size)
        plt.grid(False)
        plt.show()

def Statistic_ProspectusAnalysis(obj, plot_prospectus_analysis):
    cols = ['InitialProspectusPositive',
            'InitialProspectusNegative',
            'FinalProspectusPositive',
            'FinalProspectusNegative',
            'PositiveWordsRevision',
            'NegativeWordsRevision',
            'FinalProspectusOfferDays']
        
    prosp_data = obj.full_data
    prosp_data_analysis = prosp_data[cols]
    n_obs = len(prosp_data_analysis)
        
    percs = [0.01, 0.05, 0.5, 0.95, 0.99]
    result = prosp_data_analysis.describe(percentiles = percs)
    result = result.transpose()
    result.index = result.index.rename('Variable')
        
    plot_result = tabulate(result,
                           headers = 'keys',
                           floatfmt = '.2f',
                           tablefmt = 'simple',
                           numalign = 'center',
                           showindex = True)
    
    if plot_prospectus_analysis == True:    
        print('\n')
        print(cutting_line)
        print('Descriptive statistic of prospectus words analysis:')
        print(cutting_line, '\n')
        print(plot_result)
        print(cutting_line_thin)
        print(f'Number of observations: {n_obs}')
        print(cutting_line, paragraph)

def Statistic_RegressionSample(obj):
    drop_chars = ['SlopeDummy', 'Min', 'Max', 'Adjusted']
    reg_vars = obj.reg_vars
    stat_vars = []
    for var in reg_vars:
        if not any(char in var for char in drop_chars):
            stat_vars.append(var)

    stat_vars.append('RegistrationDays')
    stat_vars_ext = stat_vars.copy()
    stat_vars_ext.append('IssueYear')
    
    model_data = obj.model_data
    issue_year = model_data['IssueDate']
    issue_year = issue_year.dt.strftime('%Y')
    model_data['IssueYear'] = issue_year
    model_data = model_data[stat_vars_ext]
    nobs_model_data = len(model_data)
    
    reg_set = model_data.dropna()
    nobs_full_set = len(reg_set)
    reg_idx = reg_set.index
    miss_set = model_data.drop(reg_idx)
    nobs_miss_set = len(miss_set)

    result = pd.DataFrame(index = stat_vars)
    reg_mean = reg_set[stat_vars].mean()
    miss_mean = miss_set[stat_vars].mean()
    miss_values = model_data[stat_vars].isnull().sum()
    period = pd.date_range(obj.start,obj.end,freq='Y')
    period = period.strftime('%Y')
    period = pd.DataFrame(period)
    period.columns = ['Year']
    
    join_stats = [reg_mean, miss_mean, miss_values]
    cols = ['MeanReg', 'MeanNonReg', 'Missing']
    idx = {'index': 'Variable'}
    result = result.join(join_stats)
    result.columns = cols
    result = result.reset_index()
    result = result.rename(columns = idx)
    result = pd.concat([result, period], axis = 1)

    for index, row in result.iterrows():
        if pd.isnull(row['Variable']) == False:
            reg = reg_set[row['Variable']]
            missing = miss_set[row['Variable']]
            missing = missing.dropna()
            t_test = smw.ttest_ind(reg, missing,
                                   usevar = 'unequal',
                                   value = 0)
            
            tstat = t_test[0]
            pvalue = t_test[1]
            cols = ['pValue', 'tValue']
            result.loc[index, cols] = pvalue, tstat
# =========================
            year = row['Year']
            year_filt = model_data['IssueYear'] == year
            sample = model_data[year_filt]
            sample_obs = sample[stat_vars]
            sample_nobs = len(sample_obs)
            reg_sample = sample_obs.dropna()
            reg_sample_nobs = len(reg_sample)
            
            share_reg = reg_sample_nobs /sample_nobs
            mean_ir = sample_obs['InitialReturn'].mean()
            mean_ir_reg = reg_sample['InitialReturn'].mean()
            
            result.loc[index, 'IPOs'] = sample_nobs
            result.loc[index, 'ShareReg'] = share_reg
            result.loc[index, 'MeanReturnFull'] = mean_ir
            result.loc[index, 'MeanReturnReg'] = mean_ir_reg
            
        else:
            year = row['Year']
            year_filt = model_data['IssueYear'] == year
            sample = model_data[year_filt]
            sample_obs = sample[stat_vars]
            sample_nobs = len(sample_obs)
            reg_sample = sample_obs.dropna()
            reg_sample_nobs = len(reg_sample)
            
            share_reg = reg_sample_nobs /sample_nobs
            mean_ir = sample_obs['InitialReturn'].mean()
            mean_ir_reg = reg_sample['InitialReturn'].mean()
            
            result.loc[index, 'IPOs'] = sample_nobs
            result.loc[index, 'ShareReg'] = share_reg
            result.loc[index, 'MeanReturnFull'] = mean_ir
            result.loc[index, 'MeanReturnReg'] = mean_ir_reg
    
    table_cols = [
            'Variable',
            'MeanReg', 'MeanNonReg',
            'pValue', 'tValue',
            'Missing', 'Year', 
            'IPOs', 'ShareReg',
            'MeanReturnFull',
            'MeanReturnReg'
            ]

    result = result.reindex(columns=table_cols)
    plot_result = tabulate(result,
                           headers = 'keys',
                           floatfmt = '.2f',
                           tablefmt = 'simple',
                           numalign = 'center',
                           showindex = False)
    plot_result = plot_result.replace('nan', '---')
    
    print('\n')
    print(cutting_line)
    print('Descriptive statistic for regression model:')
    print(cutting_line, '\n')
    print(plot_result)
    print(cutting_line_thin)
    print(f'Number of observations: {nobs_model_data}')
    print(f'Regression sample size: {nobs_full_set}')
    print(f'Non-regression sample size: {nobs_miss_set}')
    print(cutting_line, paragraph)

    return result

def RegressionResults(obj, plot_yearly_regression):
    ols_full = obj.ols_full_result
    print(cutting_line)
    print(ols_full)
    print(cutting_line, paragraph)
# =========================   
    reg_result = obj.regression_result
    keyfigs = obj.keyfig_result
    reg_result.index = reg_result.index.rename('Variable')
    keyfigs.index = keyfigs.index.rename('Key Figure')
    
    plot_reg_result = tabulate(reg_result,
                               headers = 'keys',
                               floatfmt = '.2f',
                               tablefmt = 'simple',
                               numalign = 'center',
                               showindex = True)

    plot_keyfigs = tabulate(keyfigs,
                            headers = 'keys',
                            floatfmt = '.2f',
                            tablefmt = 'simple',
                            numalign = 'center',
                            showindex = True)
    
    print(cutting_line)
    print('Regression results of OLS and Fama-MacBeth regression')
    print(cutting_line, '\n')
    print(plot_reg_result, '\n\n')
    print(plot_keyfigs)
    print(cutting_line, paragraph)
# =========================
    ols_result = obj.ols_result_agg
    ols_keyfigs = obj.ols_keyfig_agg
    ols_result.index = ols_result.index.rename('Variable')
    ols_keyfigs.index = ols_keyfigs.index.rename('Key Figure')
    
    plot_reg_result = tabulate(ols_result,
                               headers = 'keys',
                               floatfmt = '.2f',
                               tablefmt = 'simple',
                               numalign = 'center',
                               showindex = True)
    plot_reg_result = plot_reg_result.replace('nan', '---')

    plot_keyfigs = tabulate(ols_keyfigs,
                            headers = 'keys',
                            floatfmt = '.2f',
                            tablefmt = 'simple',
                            numalign = 'center',
                            showindex = True)
    
    print(cutting_line)
    print('Aggregated OLS regression results')
    print(cutting_line, '\n')
    print(plot_reg_result, '\n\n')
    print(plot_keyfigs)
    print(cutting_line, paragraph)
# =========================
    fmb_result = obj.fmb_result_agg
    fmb_keyfigs = obj.fmb_keyfig_agg
    fmb_result.index = fmb_result.index.rename('Variable')
    fmb_keyfigs.index = fmb_keyfigs.index.rename('Key Figure')
    
    plot_reg_result = tabulate(fmb_result,
                               headers = 'keys',
                               floatfmt = '.2f',
                               tablefmt = 'simple',
                               numalign = 'center',
                               showindex = True)
    plot_reg_result = plot_reg_result.replace('nan', '---')

    plot_keyfigs = tabulate(fmb_keyfigs,
                            headers = 'keys',
                            floatfmt = '.2f',
                            tablefmt = 'simple',
                            numalign = 'center',
                            showindex = True)
    
    print(cutting_line)
    print('Aggregated Fama-MacBeth regression results')
    print(cutting_line, '\n')
    print(plot_reg_result, '\n\n')
    print(plot_keyfigs)
    print(cutting_line, paragraph)

    if plot_yearly_regression == True:
        data = obj.fmb_coefs
        fig = plt.figure(figsize=(15,25), dpi=150)
        ax1 = fig.add_subplot(projection='3d')
        
        xlabels = np.array(data.columns)
        ylabels = np.array(data.index)
        xpos = np.arange(xlabels.shape[0])
        ypos = np.arange(ylabels.shape[0])
        xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

        zpos = np.array(data)
        zpos = zpos.ravel()

        dx = 0.5
        dy = 0.5
        dz = zpos
        ax1.w_xaxis.set_ticks(xpos + dx/2.)
        ax1.w_xaxis.set_ticklabels(xlabels)
        ax1.w_yaxis.set_ticks(ypos + dy/2.)
        ax1.w_yaxis.set_ticklabels(ylabels)

        values = np.linspace(0.2, 1., xposM.ravel().shape[0])
        colors = cm.rainbow(values)
        ax1.view_init(30,70)
        ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Variables')
        ax1.set_zlabel('Coefficients')
        
        file = 'YeartoYearRegressionResults.xlsx'
        data.to_excel(output_path_results+'\\'+file)
        plt.show()
# =========================
    sub_reg_results = obj.sub_regression_results
    sub_reg_keyfigs = obj.sub_regression_keyfigs
    sub_reg_results_full = obj.sub_regression_full_results

    plot_reg_result = tabulate(sub_reg_results,
                               headers = 'keys',
                               floatfmt = '.2f',
                               tablefmt = 'simple',
                               numalign = 'center',
                               showindex = True)
    plot_reg_result = plot_reg_result.replace('nan', '---')

    plot_keyfigs = tabulate(sub_reg_keyfigs,
                            headers = 'keys',
                            floatfmt = '.2f',
                            tablefmt = 'simple',
                            numalign = 'center',
                            showindex = True)
    
    print(cutting_line)
    print('Regression results of secondary shares analysis')
    print(cutting_line, '\n')
    print('\n\n')
    print(sub_reg_results_full)
    print('\n\n')
    print(plot_reg_result, '\n\n')
    print(plot_keyfigs)
    print(cutting_line, paragraph)

def Analysis_ReturnAdjustments(feat_eng_obj, pred_obj, plot_return_adjustments, target_return):
    initial_rets = feat_eng_obj.model_data
    ret_cols = ['InitialReturn', 
                'InitialReturnAdjusted']
    rets = initial_rets[ret_cols]
    nobs = len(initial_rets)
    percs = [0.25, 0.5, 0.75]
    statistic = rets.describe(percentiles = percs)
    
    target_rets = pred_obj.target_rets
    target_ret = target_rets['InitialReturn']
    target_ret_adj = target_rets['InitialReturnAdjusted']
    target_ret_dist = target_ret.value_counts(normalize=True)
    target_ret_dist = target_ret_dist.sort_index()
    target_ret_adj_dist = target_ret_adj.value_counts(normalize=True)
    target_ret_adj_dist = target_ret_adj_dist.sort_index()
    
    result = tabulate(statistic,
                      headers = 'keys',
                      floatfmt = '.2f',
                      tablefmt = 'simple',
                      numalign = 'center',
                      showindex = True)
    
    if plot_return_adjustments == True:
        print(cutting_line)
        print('Descriptive statistic for initial returns:')
        print(cutting_line, '\n')
        print(result)
        print(cutting_line_thin)
        print(f'Number of observations: {nobs}')
        print(cutting_line, paragraph)
        
        plt.figure(figsize = figsize)
        plt.subplot(121)
        plt.hist(rets['InitialReturn'], bins = hist_bins)
        plt.xlabel('Value', fontdict = xlabel_size)
        plt.ylabel('Frequency', fontdict = ylabel_size)
        plt.title('InitialReturn', fontdict = title_size)
        plt.subplot(122)
        plt.hist(rets['InitialReturnAdjusted'], bins = hist_bins)
        plt.xlabel('Value', fontdict = xlabel_size)
        plt.ylabel('Frequency', fontdict = ylabel_size)
        plt.title('InitialReturnAdjusted', fontdict = title_size)
        
        plt.figure(figsize = figsize)
        smaller_label = f'Return < {target_return}%'
        greater_label = f'Return >= {target_return}%'
        labels = [smaller_label, greater_label]
        plt.subplot(121)
        plt.pie(target_ret_dist, 
                labels=labels, 
                autopct='%.2f%%')
        title = 'Distribution of InitialReturn'
        plt.title(title, fontdict = title_size)
        plt.subplot(122)
        plt.pie(target_ret_adj_dist, 
                labels=labels, 
                autopct='%.2f%%')
        title = 'Distribution of InitialReturnAdjusted'
        plt.title(title, fontdict = title_size)
        

