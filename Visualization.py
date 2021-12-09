# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 10:31:16 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import statsmodels.stats.weightstats as smw
from tabulate import tabulate 

def descriptive_statistic(obj, reg_vars):
    chars = ['Dummy', 'SlopeDummy', 'Min', 'Max']
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
    
    full_sample = model_data.dropna()
    full_idx = full_sample.index
    miss_sample = model_data.drop(full_idx)

    table = pd.DataFrame(index = stat_vars)
    full_mean = full_sample[stat_vars].mean()
    miss_mean = miss_sample[stat_vars].mean()
    miss_vals = model_data[stat_vars].isnull().sum()
    full_mean.name = 'MeanFull'
    miss_mean.name = 'MeanMissing'
    miss_vals.name = 'MissingValues'
    year = pd.date_range(obj.start,obj.end,freq='Y')
    year = year.strftime('%Y')
    year = pd.DataFrame(year)
    
    join_stats = [full_mean, miss_mean, miss_vals]
    adj_colname = {'index': 'Variable'}
    table = table.join(join_stats)
    table = table.reset_index()
    table = table.rename(columns = adj_colname)
    table = pd.concat([table, year], axis = 1)
    return 

