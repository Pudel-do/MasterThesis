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
from tabulate import tabulate 
import re 
from re import search

output_path_results = r'C:\Users\Matthias Pudel\OneDrive\Studium\Master\Master Thesis\Empirical Evidence\Results'

figsize = (25,10)
xlabel_size = {'fontsize' : 15}
ylabel_size = {'fontsize' : 15}
suptitle_size = 25
title_size = {'fontsize' : 20}
legend_size = {'size': 20}
hist_bins = 100
cutting_line = 135*'='
cutting_line_thin = 135*'-'
paragraph = '\n\n\n\n'

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
            'NegativeWordsRevision']
        
    prosp_data = obj.full_data
    prosp_data = prosp_data[cols]
    n_obs = len(prosp_data)
        
    percs = [0.25, 0.5, 0.75]
    result = prosp_data.describe(percentiles = percs)
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
    cols = ['MeanRegSample', 'MeanNonRegSample', 'Missing']
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
            col = 'pValue'
            result.loc[index, col] = pvalue
# =========================
            year = row['Year']
            year_filt = model_data['IssueYear'] == year
            sample = model_data[year_filt]
            sample_obs = sample[stat_vars]
            sample_nobs = len(sample_obs)
            reg_sample = sample_obs.dropna()
            reg_sample_nobs = len(reg_sample)
            
            share_reg = reg_sample_nobs /sample_nobs
            mean_ir_reg = reg_sample['InitialReturn']
            mean_ir_reg = mean_ir_reg.mean()
            
            result.loc[index, 'IPOs'] = sample_nobs
            result.loc[index, 'ShareRegSample'] = share_reg
            result.loc[index, 'MeanInitialReturnRegSample'] = mean_ir_reg
            
        else:
            year = row['Year']
            year_filt = model_data['IssueYear'] == year
            sample = model_data[year_filt]
            sample_obs = sample[stat_vars]
            sample_nobs = len(sample_obs)
            reg_sample = sample_obs.dropna()
            reg_sample_nobs = len(reg_sample)
            
            share_reg = reg_sample_nobs /sample_nobs
            mean_ir_reg = reg_sample['InitialReturn']
            mean_ir_reg = mean_ir_reg.mean()
            
            result.loc[index, 'IPOs'] = sample_nobs
            result.loc[index, 'ShareRegSample'] = share_reg
            result.loc[index, 'MeanInitialReturnRegSample'] = mean_ir_reg
    
    table_cols = [
            'Variable',
            'MeanRegSample', 'MeanNonRegSample',
            'pValue', 'Missing', 
            'Year', 'IPOs',
            'ShareRegSample', 
            'MeanInitialReturnRegSample'
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
        ax1.view_init(30,70) #(tip, turn counterclockwise(increase) and clockwise(decrease))
        ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Variables')
        ax1.set_zlabel('Coefficients')
        
        file = 'YeartoYearRegressionResults.xlsx'
        data.to_excel(output_path_results+'\\'+file)
        plt.show()

def UnderpricingAnalysis(feat_eng_obj, pred_obj, plot_underpricing_analysis, target_return):
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
    
    if plot_underpricing_analysis == True:
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
        

