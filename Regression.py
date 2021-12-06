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
        
        fmb_coefs = pd.DataFrame()
        fmb_pvalues = pd.DataFrame()
        fmb_sub_key = pd.DataFrame()
        fmb_avg_results = {}
        sub_key_cols = ['RSquared']
        for index, value in period.iteritems():
            year_filter = self.base['IssueYear'] == value
            reg_data = self.base[year_filter]
            endog = reg_data[self.endog_var]
            exog = reg_data[self.exog_var]
            exog = sm.add_constant(exog)
            
            fmb_reg = sm.OLS(endog = endog, exog = exog)
            fmb_reg = fmb_reg.fit(cov_type='HC0')
            
            coef = fmb_reg.params
            coef.name = value
            pvalue = fmb_reg.pvalues
            pvalue.name = value
            r_sqr = fmb_reg.rsquared_adj    
            nobs = fmb_reg.nobs

            fmb_coefs = pd.concat([fmb_coefs, coef], axis=1)
            fmb_pvalues = pd.concat([fmb_pvalues, pvalue], axis=1)
            fmb_sub_key.loc['RSquared', value] = r_sqr
            fmb_sub_key.loc['AmountItems', value] = nobs
            
        avg_coefs = fmb_coefs.mean(axis = 1)
        avg_coefs.name = 'Coefficient'
        avg_coefs = pd.DataFrame(avg_coefs)
        avg_pvalues = fmb_pvalues.mean(axis = 1)
        avg_pvalues.name = 'pvalue'
        avg_pvalues = pd.DataFrame(avg_pvalues)
        avg_key = avg_coefs.join(avg_pvalues)
    
        avg_sub_key = fmb_sub_key.mean(axis = 1)
        avg_sub_key = avg_sub_key.loc[sub_key_cols]
        avg_sub_key.name = 'Value'
        
        fmb_avg_results['KeyFigures'] = avg_key
        fmb_avg_results['SubKeyFigures'] = avg_sub_key
        
        self.fmb_coefs = fmb_coefs
        self.fmb_pvalues = fmb_pvalues
        self.fmb_avg_results = fmb_avg_results
        
        
    def build_results(self):
        # x = avg_results.copy()
        # x.loc['SharesFiled', 'Coefficient'] = 12
        # x.loc['SharesFiled', 'pvalue'] = 14
        # y = avg_results.join(x, how = 'right', lsuffix=' ', rsuffix=' ')
        return
        
        
    
        
        

            
            
        