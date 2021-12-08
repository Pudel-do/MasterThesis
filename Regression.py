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
    def __init__(self, feat_eng_obj, start_year, end_year):
        self.feat_eng_obj = feat_eng_obj
        self.model_data = feat_eng_obj.model_data
        self.start_year = start_year
        self.end_year = end_year
        
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
        
        key_figures = ['RSquared', 'Count']
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
        coefs.name = 'Coeff'
        coefs = pd.DataFrame(coefs)
        pvalues = ols_reg.pvalues
        pvalues.name = 'pvalue'
        pvalues = pd.DataFrame(pvalues)
        r_sqr = ols_reg.rsquared_adj
        n_obs = ols_reg.nobs
# =========================
# Dataframe key_figs can be extended 
# by furter key figures at this point
        keyfig = pd.DataFrame()
        keyfig.loc['RSquared', self.key_col] = r_sqr
        keyfig.loc['Count', self.key_col] = n_obs
        result = coefs.join(pvalues)
        full_result = ols_reg.summary()
        
        self.ols_result = result
        self.ols_full_result = full_result
        self.ols_keyfig = keyfig
            
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
        key_fig = pd.DataFrame()
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
            key_fig.loc['RSquared', value] = r_sqr
            key_fig.loc['Count', value] = nobs
            
        avg_coefs = coefs.mean(axis = 1)
        avg_coefs.name = 'Coeff'
        avg_coefs = pd.DataFrame(avg_coefs)
        avg_pvalues = pvalues.mean(axis = 1)
        avg_pvalues.name = 'pvalue'
        avg_pvalues = pd.DataFrame(avg_pvalues)
        avg_result = avg_coefs.join(avg_pvalues)
    
        avg_keyfig = key_fig.mean(axis = 1)
        avg_keyfig = avg_keyfig.loc[self.key_figures]
        avg_keyfig.name = self.key_col
        avg_keyfig = pd.DataFrame(avg_keyfig)
        
        self.fmb_coefs = coefs
        self.fmb_pvalues = pvalues
        self.fmb_result = avg_result
        self.fmb_keyfig = avg_keyfig
        
    def build_results(self, adj_reg_cols):
        start_year = self.start_year
        end_year = self.end_year
        index_col = 'Unnamed: 0'
        
        ols_result = self.ols_result
        file = ols_result_file
        file = f'{start_year}_{end_year}_{file}'
        ols_result.to_csv(output_path+'\\'+file)
        
        ols_keyfig = self.ols_keyfig
        file = ols_keyfig_file
        file = f'{start_year}_{end_year}_{file}'
        ols_keyfig.to_csv(output_path+'\\'+file)
# =========================        
        fmb_result = self.fmb_result
        file = fmb_result_file
        file = f'{start_year}_{end_year}_{file}'
        fmb_result.to_csv(output_path+'\\'+file)
        
        fmb_keyfig = self.fmb_keyfig
        file = fmb_keyfig_file
        file = f'{start_year}_{end_year}_{file}'
        fmb_keyfig.to_csv(output_path+'\\'+file)
# =========================
        cols_ols = {'Coeff': 'Coeff_OLS',
                    'pvalue': 'pvalue_OLS'}
        cols_fmb = {'Coeff': 'Coeff_FMB',
                    'pvalue': 'pvalue_FMB'}
        ols_res_adj = ols_result.rename(columns=cols_ols)
        fmb_res_adj = fmb_result.rename(columns=cols_fmb)
        reg_result = ols_res_adj.join(fmb_res_adj)    
        
        cols_ols = {'Value': 'Value_OLS'}
        cols_fmb = {'Value': 'Value_FMB'}
        ols_keyfig_adj = ols_keyfig.rename(columns=cols_ols)
        fmb_keyfig_adj = fmb_keyfig.rename(columns=cols_fmb)
        keyfig_result = ols_keyfig_adj.join(fmb_keyfig_adj) 
# =========================    
        file_ols = ols_aggresult_file
        file_ols = f'{start_year}_{end_year}_{file_ols}'
        file_ols_keyfig = ols_aggkeyfig_file
        file_ols_keyfig = f'{start_year}_{end_year}_{file_ols_keyfig}'
        file_fmb = fmb_aggresult_file
        file_fmb = f'{start_year}_{end_year}_{file_fmb}'
        file_fmb_keyfig = fmb_aggkeyfig_file
        file_fmb_keyfig = f'{start_year}_{end_year}_{file_fmb_keyfig}'
# =========================        
        if adj_reg_cols == False:
            ols_aggres = ols_result
            ols_aggkey = ols_keyfig
            ols_aggres.to_csv(output_path+'\\'+file_ols)
            ols_aggkey.to_csv(output_path+'\\'+file_ols_keyfig)
            fmb_aggres = fmb_result
            fmb_aggkey = fmb_keyfig
            fmb_aggres.to_csv(output_path+'\\'+file_fmb)
            fmb_aggkey.to_csv(output_path+'\\'+file_fmb_keyfig)
        else:
            ols_aggres = pd.read_csv(output_path+'\\'+file_ols,
                                     index_col = index_col)
            ols_aggkey = pd.read_csv(output_path+'\\'+file_ols_keyfig,
                                     index_col = index_col)
            fmb_aggres = pd.read_csv(output_path+'\\'+file_fmb,
                                     index_col = index_col)
            fmb_aggkey = pd.read_csv(output_path+'\\'+file_fmb_keyfig,
                                     index_col = index_col)

            ols_aggres = pd.concat([ols_aggres, ols_result], axis=1) 
            ols_aggkey = pd.concat([ols_aggkey, ols_keyfig], axis=1) 
            ols_aggres.to_csv(output_path+'\\'+file_ols)
            ols_aggkey.to_csv(output_path+'\\'+file_ols_keyfig)

            fmb_aggres = pd.concat([fmb_aggres, fmb_result], axis=1)
            fmb_aggkey = pd.concat([fmb_aggkey, fmb_keyfig], axis=1)
            fmb_aggres.to_csv(output_path+'\\'+file_fmb)
            fmb_aggkey.to_csv(output_path+'\\'+file_fmb_keyfig)
        
        self.ols_result_agg = ols_aggres
        self.ols_keyfig_agg = ols_aggkey
        self.fmb_result_agg = fmb_aggres
        self.fmb_keyfig_agg = fmb_aggkey
        self.regression_result = reg_result
        self.regression_keyfig = keyfig_result
        