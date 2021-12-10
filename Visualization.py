# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 10:31:16 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import statsmodels.stats.weightstats as smw
from tabulate import tabulate 

def descriptive_statistic(obj, equal_std):
    chars = ['Dummy', 'SlopeDummy', 'Min', 'Max']
    reg_vars = obj.reg_vars
    stat_vars = [var for var in reg_vars\
                if chars[0] not in var\
                and chars[1] not in var\
                and chars[2] not in var\
                and chars[3] not in var]

    stat_vars_ext = stat_vars.copy()
    stat_vars_ext.append('IssueYear')
    
    model_data = obj.model_data
    issue_year = model_data['IssueDate']
    issue_year = issue_year.dt.strftime('%Y')
    model_data['IssueYear'] = issue_year
    model_data = model_data[stat_vars_ext]
    
    full_set = model_data.dropna()
    full_idx = full_set.index
    miss_set = model_data.drop(full_idx)

    table = pd.DataFrame(index = stat_vars)
    full_mean = full_set[stat_vars].mean()
    miss_mean = miss_set[stat_vars].mean()
    miss_vals = model_data[stat_vars].isnull().sum()
    full_mean.name = 'MeanFull'
    miss_mean.name = 'MeanMissing'
    miss_vals.name = 'MissingValues'
    period = pd.date_range(obj.start,obj.end,freq='Y')
    period = period.strftime('%Y')
    period = pd.DataFrame(period)
    period.columns = ['Year']
    
    join_stats = [full_mean, miss_mean, miss_vals]
    adj_colname = {'index': 'Variable'}
    table = table.join(join_stats)
    table = table.reset_index()
    table = table.rename(columns = adj_colname)
    table = pd.concat([table, period], axis = 1)
    
    if equal_std == True:
        usevar = 'pooled'
    else:
        usevar = 'unequal'
    for index, row in table.iterrows():
        if pd.isnull(row['Variable']) == False:
            full = full_set[row['Variable']]
            missing = miss_set[row['Variable']]
            missing = missing.dropna()
            t_test = smw.ttest_ind(full, missing,
                                   usevar = usevar,
                                   value = 0)
            
            tstat = t_test[0]
            pvalue = t_test[1]
            col = 'pValue'
            table.loc[index, col] = pvalue
# =========================
            year = row['Year']
            year_filt = model_data['IssueYear'] == year
            sample = model_data[year_filt]
            sample_obs = sample[stat_vars]
            sample_nobs = len(sample_obs)
            
            fullsample_obs = sample_obs.dropna()
            fullsample_nobs = len(fullsample_obs)
            
            share = fullsample_nobs /sample_nobs
            col = 'ShareFullSample'
            table.loc[index, col] = share
        else:
            year = row['Year']
            year_filt = model_data['IssueYear'] == year
            sample = model_data[year_filt]
            sample_obs = sample[stat_vars]
            sample_nobs = len(sample_obs)
            
            fullsample_obs = sample_obs.dropna()
            fullsample_nobs = len(fullsample_obs)
            
            share = fullsample_nobs /sample_nobs
            col = 'ShareFullSample'
            table.loc[index, col] = share
    
    adj_value = 'TotalAssets'
    adj_cols = ['MeanFull', 'MeanMissing']
    loc_idx = table[table['Variable'] == adj_value]
    loc_idx = loc_idx.index
    values = table.loc[loc_idx, adj_cols]
    values_adj = values / 1000000000
    table.loc[loc_idx, adj_cols] = values_adj
    
    cols = ['Variable', 'MeanFull', 'MeanMissing',
            'pValue', 'MissingValues', 'Year',
            'ShareFullSample']
    
    table = table.reindex(columns=cols)
    plot_table = tabulate(table,
                          headers = 'keys',
                          floatfmt = '.2f',
                          tablefmt = 'simple',
                          numalign = 'center',
                          showindex = False)   
    print(plot_table)
             
    return table

