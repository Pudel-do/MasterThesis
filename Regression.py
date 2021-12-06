# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:10:59 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.weightstats as smw

class Regression:
    def __init__(self, model_data):
        self.model_data = model_data
        
    def preprocessing(self, reg_cols):
        base = self.model_data
        base = base.dropna()

        issue_year = base['IssueDate']
        issue_year = issue_year.dt.strftime('%Y')
        base['IssueYear'] = issue_year
        self.base = base
        
        endog_var = 'InitialReturn'
        exog_var = reg_cols.copy()
        exog_var.remove('InitialReturn')
        self.endog_var = endog_var
        self.exog_var = exog_var
        
        print('Test')

    def ols_regression(self):
        return
    
    def fmb_regression(self, start_date, end_date):
        self.start = start_date
        self.end = end_date
        
        period = pd.date_range(start = self.start,
                               end = self.end,
                               freq = 'Y')
        
        period = period.strftime('%Y')
        period = pd.Series(period)
        
        for index, value in period.iteritems():
            print('Test')
            
            
        