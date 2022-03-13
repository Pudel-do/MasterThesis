# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:10:59 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.weightstats as smw
import sys
from GetData import *
import warnings
warnings.filterwarnings("ignore")

sub_reg_cols = [
    'InitialReturn', 
    'UnderwriterRank', 
    'TotalAssets', 
    'TechDummy', 
    'VentureDummy',
    'ExpectedProceeds', 
    'TechSectorVolume',
    'MarketReturn', 
    'SectorReturn', 
    'WordsRevisionDummy', 
    'PriceRevision', 
    'SecondarySharesRevision', 
    'SharesShiftDummy',
    ]

class Regression:
    def __init__(self, feat_eng_obj, start_year, end_year):
        self.feat_eng_obj = feat_eng_obj
        self.model_data = feat_eng_obj.model_data
        self.start_year = start_year
        self.end_year = end_year
        self.reg_model_cols = feat_eng_obj.model_cols
        
    def preprocessing(self, reg_vars):
        base = self.model_data
        base = base[reg_vars]
        base = base.dropna()
        base = base.join(self.model_data['IssueDate'])

        issue_year = base['IssueDate']
        issue_year = issue_year.dt.strftime('%Y')
        base['IssueYear'] = issue_year
        self.base = base
        self.reg_vars = reg_vars

        endog_var = 'InitialReturn'
        exog_var = reg_vars.copy()
        exog_var.remove('InitialReturn')
        self.endog_var = endog_var
        self.exog_var = exog_var
        
        key_figures = ['AdjRSquared', 'SampleSize']
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
        tvalues = ols_reg.tvalues
        tvalues.name = 'tvalue'
        tvalues = pd.DataFrame(tvalues)
        r_sqr = ols_reg.rsquared_adj
        n_obs = ols_reg.nobs
# =========================
# Dataframe key_figs can be extended 
# by furter key figures at this point
        keyfig = pd.DataFrame()
        keyfig.loc['AdjRSquared', self.key_col] = r_sqr
        keyfig.loc['SampleSize', self.key_col] = n_obs
        result = coefs.join(pvalues)
        result = result.join(tvalues)
        full_result = ols_reg.summary()
        
        self.ols_result = result
        self.ols_full_result = full_result
        self.ols_keyfig = keyfig
        
    def ols_sub_regression(self):
        sub_reg_data = self.model_data[sub_reg_cols]
        sub_reg_data = sub_reg_data.dropna()
        
        sub_reg_endog = sub_reg_data['InitialReturn']
        sub_reg_exog = sub_reg_data.drop(columns=['InitialReturn'])
        sub_reg_exog = sm.add_constant(sub_reg_exog)
        
        ols_sub_reg = sm.OLS(endog = sub_reg_endog, 
                             exog = sub_reg_exog)
        ols_sub_reg = ols_sub_reg.fit(cov_type='HC0')
        self.sub_regression_results = ols_sub_reg.summary()
            
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
        tvalues = pd.DataFrame()
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
            tvalue = fmb_reg.tvalues
            tvalue.name = value
            r_sqr = fmb_reg.rsquared_adj    
            nobs = fmb_reg.nobs

            coefs = pd.concat([coefs, coef], axis=1)
            pvalues = pd.concat([pvalues, pvalue], axis=1)
            tvalues = pd.concat([tvalues, tvalue], axis=1)
# =========================
# Dataframe key_figs can be extended 
# by furter key figures at this point            
            key_fig.loc['AdjRSquared', value] = r_sqr
            key_fig.loc['SampleSize', value] = nobs
            
        avg_coefs = coefs.mean(axis = 1)
        avg_coefs.name = 'Coeff'
        avg_coefs = pd.DataFrame(avg_coefs)
        avg_pvalues = pvalues.drop(columns = ['2008', '2009'])
        avg_pvalues = avg_pvalues.mean(axis = 1)
        avg_pvalues.name = 'pvalue'
        avg_pvalues = pd.DataFrame(avg_pvalues)
        avg_tvalues = tvalues.drop(columns = ['2008', '2009'])
        avg_tvalues = avg_tvalues.mean(axis = 1)
        avg_tvalues.name = 'tvalue'
        avg_tvalues = pd.DataFrame(avg_tvalues)
        avg_result = avg_coefs.join([avg_pvalues, avg_tvalues])
    
        avg_keyfig = key_fig.mean(axis = 1)
        avg_keyfig = avg_keyfig.loc[self.key_figures]
        avg_keyfig.name = self.key_col
        avg_keyfig = pd.DataFrame(avg_keyfig)
        
        self.fmb_coefs = coefs
        self.fmb_pvalues = pvalues
        self.fmb_result = avg_result
        self.fmb_keyfig = avg_keyfig
        
    def build_results(self, adj_regressors, clean_file):
        start_year = self.start_year
        end_year = self.end_year
        index_col = 'Unnamed: 0'
        
        ols_result = self.ols_result
        ols_keyfig = self.ols_keyfig
        fmb_result = self.fmb_result
        fmb_keyfig = self.fmb_keyfig

        cols_ols = {'Coeff': 'Coeff_OLS',
                    'pvalue': 'pvalue_OLS',
                    'tvalue': 'tvalue_OLS'}
        cols_fmb = {'Coeff': 'Coeff_FMB',
                    'pvalue': 'pvalue_FMB',
                    'tvalue': 'tvalue_FMB'}
        ols_res_adj = ols_result.rename(columns=cols_ols)
        fmb_res_adj = fmb_result.rename(columns=cols_fmb)
        reg_result = ols_res_adj.join(fmb_res_adj)   
        
        cols_ols = {'Value': 'Value_OLS'}
        cols_fmb = {'Value': 'Value_FMB'}
        ols_keyfig_adj = ols_keyfig.rename(columns=cols_ols)
        fmb_keyfig_adj = fmb_keyfig.rename(columns=cols_fmb)
        keyfig_result = ols_keyfig_adj.join(fmb_keyfig_adj) 
        
        self.regression_result = reg_result
        self.keyfig_result = keyfig_result
# =========================    
        file_ols = ols_aggresult_file
        file_ols = f'{start_year}_{end_year}_{file_ols}'
        file_ols_keyfig = ols_aggkeyfig_file
        file_ols_keyfig = f'{start_year}_{end_year}_{file_ols_keyfig}'
        file_fmb = fmb_aggresult_file
        file_fmb = f'{start_year}_{end_year}_{file_fmb}'
        file_fmb_keyfig = fmb_aggkeyfig_file
        file_fmb_keyfig = f'{start_year}_{end_year}_{file_fmb_keyfig}'
        
        if adj_regressors == True:
            ols_aggres = pd.read_csv(output_path+'\\'+file_ols, index_col=index_col)
            ols_aggres = pd.concat([ols_aggres, ols_result], axis=1)
            ols_aggres.to_csv(output_path+'\\'+file_ols)
            
            ols_aggkey = pd.read_csv(output_path+'\\'+file_ols_keyfig, index_col=index_col)
            ols_aggkey = pd.concat([ols_aggkey, ols_keyfig], axis=1)
            ols_aggkey.to_csv(output_path+'\\'+file_ols_keyfig)
            
            fmb_aggres = pd.read_csv(output_path+'\\'+file_fmb, index_col=index_col)
            fmb_aggres = pd.concat([fmb_aggres, fmb_result], axis=1)
            fmb_aggres.to_csv(output_path+'\\'+file_fmb)
            
            fmb_aggkey = pd.read_csv(output_path+'\\'+file_fmb_keyfig, index_col=index_col)
            fmb_aggkey = pd.concat([fmb_aggkey, fmb_keyfig], axis=1)
            fmb_aggkey.to_csv(output_path+'\\'+file_fmb_keyfig)
            
            sys.exit(exit_message2)
        else:
            ols_aggres = pd.read_csv(output_path+'\\'+file_ols, index_col=index_col)
            ols_aggres = ols_aggres.iloc[: , 1:]
            ols_aggkey = pd.read_csv(output_path+'\\'+file_ols_keyfig, index_col=index_col)
            ols_aggkey = ols_aggkey.iloc[: , 1:]
            
            fmb_aggres = pd.read_csv(output_path+'\\'+file_fmb, index_col=index_col)
            fmb_aggres = fmb_aggres.iloc[: , 1:]
            fmb_aggkey = pd.read_csv(output_path+'\\'+file_fmb_keyfig, index_col=index_col)
            fmb_aggkey = fmb_aggkey.iloc[: , 1:]
            
            adj_df_cols = [ols_aggres, ols_aggkey, fmb_aggres, fmb_aggkey]       
            for df in adj_df_cols:
                cols = pd.Series(df.columns)
                cols = cols.str.replace('.1', ' ')
                new_cols = cols.to_list()
                df.columns = new_cols
            
            self.ols_result_agg = ols_aggres
            self.ols_keyfig_agg = ols_aggkey
            self.fmb_result_agg = fmb_aggres
            self.fmb_keyfig_agg = fmb_aggkey
            if clean_file == True:
                ols_aggres = pd.DataFrame(columns = [index_col])
                ols_aggres.to_csv(output_path+'\\'+file_ols)
                ols_aggkey = pd.DataFrame(columns = [index_col])
                ols_aggkey.to_csv(output_path+'\\'+file_ols_keyfig)
                fmb_aggres = pd.DataFrame(columns = [index_col])
                fmb_aggres.to_csv(output_path+'\\'+file_fmb)
                fmb_aggkey = pd.DataFrame(columns = [index_col])
                fmb_aggkey.to_csv(output_path+'\\'+file_fmb_keyfig)
                
                sys.exit(exit_message2)
            
            

        