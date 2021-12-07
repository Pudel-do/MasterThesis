# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:10:59 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.weightstats as smw
from GetData import *

class Regression:
    def __init__(self, model_data, start_year, end_year):
        self.model_data = model_data
        self.start_year = start_year
        self.end_year = end_year
        self.coef_col = 'Coeff'
        self.pval_col = 'pvalue'
        
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
        
        key_figures = ['RSquared']
        key_col = 'Value'
        self.key_figures = key_figures
        self.key_col = key_col

    def ols_regression(self):
        reg_data = self.base
        endog = reg_data[self.endog_var]
        exog = reg_data[self.exog_var]
        exog = sm.add_constant(exog)
        
        ols_reg = sm.OLS(endog = endog, exog = exog)
        ols_reg = ols_reg.fit(cov_type='HC0')
        
        coefs = ols_reg.params
        coefs.name = self.coef_col
        coefs = pd.DataFrame(coefs)
        pvalues = ols_reg.pvalues
        pvalues.name = self.pval_col
        pvalues = pd.DataFrame(pvalues)
        r_sqr = ols_reg.rsquared_adj
# =========================
# Dataframe key_figs can be extended 
# by furter key figures at this point
        key_figs = pd.DataFrame()
        key_figs.loc['RSquared', self.key_col] = r_sqr
        result = coefs.join(pvalues)
        full_result = ols_reg.summary()
        
        self.ols_result = result
        self.ols_full_result = full_result
        self.ols_key_figs = key_figs
            
    def fmb_regression(self, start_date, end_date):
        self.start = start_date
        self.end = end_date
        
        period = pd.date_range(start = self.start,
                               end = self.end,
                               freq = 'Y')
        
        period = period.strftime('%Y')
        period = pd.Series(period)
        
        coefs = pd.DataFrame()
        pvalues = pd.DataFrame()
        key_figs = pd.DataFrame()
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

            coefs = pd.concat([coefs, coef], axis=1)
            pvalues = pd.concat([pvalues, pvalue], axis=1)
# =========================
# Dataframe key_figs can be extended 
# by furter key figures at this point            
            key_figs.loc['RSquared', value] = r_sqr
            key_figs.loc['AmountItems', value] = nobs
            
        avg_coefs = coefs.mean(axis = 1)
        avg_coefs.name = self.coef_col
        avg_coefs = pd.DataFrame(avg_coefs)
        avg_pvalues = pvalues.mean(axis = 1)
        avg_pvalues.name = self.pval_col
        avg_pvalues = pd.DataFrame(avg_pvalues)
        avg_result = avg_coefs.join(avg_pvalues)
    
        avg_key_figs = key_figs.mean(axis = 1)
        avg_key_figs = avg_key_figs.loc[self.key_figures]
        avg_key_figs.name = self.key_col
        avg_key_figs = pd.DataFrame(avg_key_figs)
        
        self.fmb_coefs = coefs
        self.fmb_pvalues = pvalues
        self.fmb_result = avg_result
        self.fmb_key_figs = avg_key_figs
        
    def build_results(self, adj_reg_cols):
        start_year = self.start_year
        end_year = self.end_year
        index_col = 'Unnamed: 0'
        
        ols_result = self.ols_result
        file = ols_result_file
        file = f'{start_year}_{end_year}_{file}'
        ols_result.to_csv(output_path+'\\'+file)
        
        ols_key_figs = self.ols_key_figs
        file = ols_key_fig_file
        file = f'{start_year}_{end_year}_{file}'
        ols_key_figs.to_csv(output_path+'\\'+file)
# =========================        
        fmb_result = self.fmb_result
        file = fmb_result_file
        file = f'{start_year}_{end_year}_{file}'
        fmb_result.to_csv(output_path+'\\'+file)
        
        fmb_key_figs = self.fmb_key_figs
        file = fmb_key_fig_file
        file = f'{start_year}_{end_year}_{file}'
        fmb_key_figs.to_csv(output_path+'\\'+file)
# =========================         
        file_ols = ols_agg_result_file
        file_ols = f'{start_year}_{end_year}_{file_ols}'
        file_ols_key = ols_agg_key_file
        file_ols_key = f'{start_year}_{end_year}_{file_ols_key}'
        file_fmb = fmb_agg_result_file
        file_fmb = f'{start_year}_{end_year}_{file_fmb}'
        file_fmb_key = fmb_agg_key_file
        file_fmb_key = f'{start_year}_{end_year}_{file_fmb_key}'
        
        if adj_reg_cols == False:
            ols_agg = ols_result
            ols_key_agg = ols_key_figs
            ols_agg.to_csv(output_path+'\\'+file_ols)
            ols_key_agg.to_csv(output_path+'\\'+file_ols_key)
            fmb_agg = fmb_result
            fmb_key_agg = fmb_key_figs
            fmb_agg.to_csv(output_path+'\\'+file_fmb)
            fmb_key_agg.to_csv(output_path+'\\'+file_fmb_key)
        else:
            ols_agg = pd.read_csv(output_path+'\\'+file_ols,index_col=index_col)
            ols_key_agg = pd.read_csv(output_path+'\\'+file_ols_key,index_col=index_col)
            fmb_agg = pd.read_csv(output_path+'\\'+file_fmb,index_col=index_col)
            fmb_key_agg = pd.read_csv(output_path+'\\'+file_fmb_key,index_col = index_col)

            ols_agg = pd.concat([ols_agg, ols_result], axis=1) 
            ols_key_agg = pd.concat([ols_key_agg, ols_key_figs], axis=1) 
            ols_agg.to_csv(output_path+'\\'+file_ols)
            ols_key_agg.to_csv(output_path+'\\'+file_ols_key)
# ========================= 
            fmb_agg = pd.concat([fmb_agg, fmb_result], axis=1)
            fmb_key_agg = pd.concat([fmb_key_agg, fmb_key_figs], axis=1)
            fmb_agg.to_csv(output_path+'\\'+file_fmb)
            fmb_key_agg.to_csv(output_path+'\\'+file_fmb_key)
            

            print('Test')

        
        
    
        
        

            
            
        