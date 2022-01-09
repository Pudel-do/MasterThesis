# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:40:40 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import re
import sys
from pandas.tseries.offsets import DateOffset
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from GetData import *
import warnings
warnings.filterwarnings("ignore")

class DataPreparation:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.start_year = start_date.strftime('%Y')
        self.end_year = end_date.strftime('%Y')
        
    def rough_preprocessing(self, adj_time_range):
        sdc_raw = pd.read_csv(input_path+'\\'+ sdc_raw_file,
                              parse_dates = ['IssueDate', 'FilingDate'])
        
        dup_ident = ['Issuer']
        sdc = sdc_raw.drop_duplicates(subset = dup_ident)
        ext_port_dt = self.start_date - DateOffset(years=1)
        
        rough_flt = (
            (sdc['IssueDate'] >= ext_port_dt) &\
            (sdc['IssueDate'] <= self.end_date) &\
            (sdc['FilingDate'] <= self.end_date) &\
            (pd.isnull(sdc['IssueDate']) == False) &\
            (pd.isnull(sdc['FilingDate']) == False) &\
            (pd.isnull(sdc['CUSIP9']) == False)
            )   
        sdc = sdc.loc[rough_flt]

        start_year = ext_port_dt.strftime('%Y')
        end_year = self.end_date.strftime('%Y')
        ncusip = sdc['CUSIP9'].str[:8]        
        ncusip_file = f'NCUSIP_{start_year}_{end_year}.txt'
        sdc['NCUSIP'] = ncusip       
        if adj_time_range == True:
            ncusip.to_csv(output_path+'\\'+ncusip_file,
                          header = False,
                          index = False)
            
            sys.exit(exit_message)
# =========================
        extended_flt = (
            (sdc['IPO'] == 'Yes') &\
            (sdc['Units'] == 'No') &\
            (sdc['ADR'] == 'No') &\
            (sdc['CEF'] == 'No') &\
            (pd.isnull(sdc['REIT']) == True) &\
            (sdc['OfferPrice'] >= 5)
                )
            
        exchange_flt = (
            (sdc['ExchangeWhereIssuWillBeLi'] == 'NASDQ') |\
            (sdc['ExchangeWhereIssuWillBeLi'] == 'NYSE') |\
            (sdc['ExchangeWhereIssuWillBeLi'] == 'AMEX')
                )
                              
        port_data = sdc.loc[extended_flt]
        port_data = port_data.loc[exchange_flt]
        base_filter = port_data['IssueDate']>=self.start_date
        base = port_data.loc[base_filter]
        self.port_data = port_data
        self.base = base
        
    def build_aux_vars(self, industry_treshold):   
        onebday_offset = pd.offsets.BusinessDay(1)
        twobday_offset = pd.offsets.BusinessDay(2)
        first_trade_dt = self.base['IssueDate'] + onebday_offset
        second_trade_dt = self.base['IssueDate'] + twobday_offset
        
        last_trade_wk_dt = pd.Series([])
        for index, value in first_trade_dt.items():
            day_rg = pd.date_range(start = value, 
                                   end = value + pd.Timedelta('7 days'),
                                   freq='W-MON')
            last_bday = day_rg[-1] - onebday_offset
            if last_bday == value:
                last_bday = last_bday + pd.Timedelta('7 days')
            last_trade_wk_dt.loc[index] = last_bday
            
        self.base['FirstTradeDate'] = first_trade_dt
        self.base['SecondTradeDate'] = second_trade_dt
        self.base['LastTradeDateWK'] = last_trade_wk_dt
# =========================        
        cpi_merge_dt = self.base['IssueDate'].dt.strftime('%Y-%m')
        self.base['CPI_MergeDate'] = cpi_merge_dt
# =========================        
        registration_days = (self.base['IssueDate'] - self.base['FilingDate'])
        registration_days = registration_days.dt.days
        self.base['RegistrationDays'] = registration_days
# =========================
        ff_industry_data = pd.read_excel(input_path+'\\'+industry_dummies_file, 
                                         engine = 'openpyxl')
         
        fama_french_industries = {}
        for column in ff_industry_data:
            industry = ff_industry_data[column]
            industry = industry.dropna()
            industries = np.array([])
            for index, value in industry.items():
                sample_range = re.findall(r'\b\d+\b', value)
                low_bound = int(sample_range[0])
                up_bound = int(sample_range[1])
                up_bound = up_bound+1
                sample = np.arange(low_bound, up_bound)
                industries = np.append(industries, sample)

            col_adj = re.sub('[^a-zA-Z]+', '', column)
            fama_french_industries[col_adj] = industries
        
        col_name = 'FamaFrenchIndustry'
        for index, row in self.port_data.iterrows():
            sic_code = row['MainSICCode']
            if (pd.isnull(sic_code) == False)&\
            (sic_code.isdecimal() == True):                
                for key, value in fama_french_industries.items():
                    sic_code = int(sic_code)
                    if sic_code in value:
                        self.port_data.loc[index, col_name] = key         
            else:
                self.port_data.loc[index, col_name] = np.nan
        
        treshold = industry_treshold
        indu_dist = self.port_data[col_name]
        indu_dist = indu_dist.value_counts(normalize=True)
        indu_dist_adj = indu_dist.where(indu_dist >= treshold)
        indu_dist_adj = indu_dist_adj.dropna()
        valid_industries = pd.Series(indu_dist_adj.index)
        
        self.industry_dist = indu_dist
        self.industry_dist_adj = indu_dist_adj
        self.valid_industries = valid_industries
        self.base = self.base.join(self.port_data[col_name])
    
    def extended_preprocessing(self, adj_uw_matching):
        mean_prc_rg = np.where(pd.isnull(self.base['OriginalMiddleOfFilingPriceRange']) == True,
                                         self.base['AmendedMiddleOfFilingPrice'],
                                         self.base['OriginalMiddleOfFilingPriceRange'])
                                
        min_prc_rg = np.where(pd.isnull(self.base['OriginalLowFilingPrice']) == True,
                                        self.base['AmendedLowFilingPrice'],
                                        self.base['OriginalLowFilingPrice'])
                                
        max_prc_rg = np.where(pd.isnull(self.base['OriginalHighFilingPrice']) == True,
                                        self.base['AmendedHighFilingPrice'],
                                        self.base['OriginalHighFilingPrice'])
        
        self.base['MeanPriceRange'] = mean_prc_rg
        self.base['MinPriceRange'] = min_prc_rg
        self.base['MaxPriceRange'] = max_prc_rg
# =========================         
        shares_filed = np.where(pd.isnull(self.base['SharesFiledSumOfAllMkts']) == True,
                                          self.base['AmendedShsFiledSumOfAllMkts'],
                                          self.base['SharesFiledSumOfAllMkts'])
        
        shares_filed_idx = self.base['SharesFiledSumOfAllMkts'].index
        shares_filed = pd.Series(shares_filed, index = shares_filed_idx)
        shares_filed = shares_filed.str.replace(',', '')
        shares_filed = shares_filed.astype(float)
        self.base['SharesFiled'] = shares_filed
# =========================       
        ritter_data = pd.read_excel(input_path+'\\'+uw_file,
                                            engine = 'openpyxl',
                                            sheet_name = 'UnderwriterRank')
        
        name = ritter_data['Underwriter Name']
        name = name[:-2]
        name = name.map(lambda x: re.sub(r'\W+', '', x))
        name = name.str.upper()
        name = name.str.strip()
        name = pd.DataFrame(name)
        name.columns = ['Name']
        
        rank_1985_2000 = ritter_data.loc[:, 'Rank8591':'Rank9200']
        rank_2001_2020 = ritter_data.loc[:, 'Rank0104':'Rank1820']
        
        if self.start_date == pd.Timestamp('2000-01-01'):
            rank_data = name.join(rank_2001_2020)
        else:
            rank_data = name.join(rank_1985_2000)
          
        rank_data = rank_data.replace(-9,np.nan)
        rank_data['MeanRank'] = rank_data.mean(axis = 1)
        rank_data['MinRank'] = rank_data.min(axis = 1)
        rank_data['MaxRank'] = rank_data.max(axis = 1)
        rank_data_cols = ['Name', 'MeanRank',
                          'MinRank', 'MaxRank']
        
        self.rank_data = rank_data[rank_data_cols]

        cols = ['DealNumber', 'LeadManagersLongName']
        sdc_uwriter = self.base[cols]
        match_results = pd.DataFrame()
        if adj_uw_matching == True:
            iter_count = 0
            progress = 100
            print('Underwriter matching started')
            for index, row in sdc_uwriter.iterrows():
                iter_count += 1
                if iter_count % progress == 0:
                    print(f'Progress: {iter_count} items')
                uwriters = row['LeadManagersLongName']
                uwriters = re.findall(r'[^|]+(?=|[^|]*$)', uwriters)
                match_lvl_treshold = 0
                for uwriter in uwriters:
                    uwriter_adj = re.sub(r'\W+', '', uwriter)
                    uwriter_adj = uwriter_adj.upper()
                    uwriter_adj = uwriter_adj.strip()
                    
                    matching = process.extractOne(uwriter_adj, rank_data['Name'])
                    match_lvl = matching[1]
                    match_name = matching[0]
                    if match_lvl > match_lvl_treshold:
                        match_lvl_treshold = match_lvl
                        match_results.loc[index, 'DealNumber'] = row['DealNumber']
                        match_results.loc[index, 'SDC_Name'] = uwriter_adj
                        match_results.loc[index, 'Ritter_Name'] = match_name
                        match_results.loc[index, 'Matching_Level'] = match_lvl
            
            file_name = uw_matching_file
            file_name = f'{self.start_year}_{self.end_year}_{file_name}'
            match_results.to_csv(output_path+'\\'+file_name, index = False)
            print('Underwriter matching finished', '\n')
        else:
            file_name = uw_matching_file
            file_name = f'{self.start_year}_{self.end_year}_{file_name}'
            match_results = pd.read_csv(output_path+'\\'+file_name)
            
        self.base = pd.merge(self.base,
                             match_results, 
                             how = 'left',
                             on = 'DealNumber')      

        self.full_data = self.base.copy()
        
    def data_merging(self, adj_close_price):
        input_cols = [
            'DealNumber',
            'EDGAR_Initial_Prospectus_FormType_CIK_merged', 
            'EDGAR_Initial_Prospectus_DateFiled_CIK_merged', 
            'EDGAR_Initial_Prospectus_FileName_CIK_merged',
            'EDGAR_Final_Prospectus_FormType_CIK_merged', 
            'EDGAR_Final_Prospectus_DateFiled_CIK_merged', 
            'EDGAR_Final_Prospectus_FileName_CIK_merged',
            ]
        output_cols = [
            'DealNumber',
            'InitialProspectusType', 
            'InitialProspectusDateFiled', 
            'InitialProspectusFileName',
            'FinalProspectusType', 
            'FinalProspectusDateFiled', 
            'FinalProspectusFileName',
            ]
        
        prospectus_data = pd.read_csv(input_path+'\\'+prosp_merge_file,
                                 usecols = input_cols)
        prospectus_data = prospectus_data.dropna()
        prospectus_data.columns = output_cols
        
        file_cols = ['InitialProspectusFileName', 'FinalProspectusFileName']
        remove_char = 'edgar/data/'
        replace_char = '/'
        
        adj_cols = prospectus_data[file_cols]
        adj_cols = adj_cols.replace(remove_char,'', regex = True)
        adj_cols = adj_cols.replace(replace_char,'_', regex = True)
        for col in file_cols:
            adj_cols[col] = adj_cols[col].str.strip()
            
        prospectus_data[file_cols] = adj_cols
        self.full_data = pd.merge(self.full_data,
                                  prospectus_data,
                                  how = 'left',
                                  on = 'DealNumber')                
# =========================         
        total_assets = pd.read_csv(input_path+'\\'+total_assets_file)
        self.full_data = pd.merge(self.full_data,
                                  total_assets,
                                  how = 'left',
                                  on = 'DealNumber')
# =========================         
        cpi_data = pd.read_excel(input_path+'\\'+cpi_file,
                                 engine = 'openpyxl',
                                 names = ['Date', 'CPI'])
        
        start = self.start_date
        base_year = start - DateOffset(years=1)
        base_year = base_year.strftime('%Y')
        
        cpi_dt_adj = cpi_data['Date'].dt.strftime('%Y')
        cpi_base_year = cpi_data[cpi_dt_adj == base_year]
        cpi_base_year = cpi_base_year['CPI'].mean()
        cpi_data['Date'] = cpi_data['Date'].dt.strftime('%Y-%m')
        
        self.cpi_base_year = cpi_base_year
        self.full_data = pd.merge(self.full_data,
                                  cpi_data,
                                  how = 'left',
                                  left_on = 'CPI_MergeDate',
                                  right_on = 'Date')
        self.full_data = self.full_data.drop(columns = 'Date')
# =========================     
        self.full_data = pd.merge(self.full_data,
                                  self.rank_data,
                                  how = 'left',
                                  left_on = 'Ritter_Name',
                                  right_on = 'Name')
# ========================= 
        quotes = pd.read_csv(input_path+'\\'+quotes_file,
                             parse_dates = ['date'])
        
        quotes['PRC'] = quotes['PRC'].abs()
        quotes_cols = ['date', 'NCUSIP', 'PRC', 'OPENPRC']
        
        if adj_close_price == False:
            self.full_data = pd.merge(self.full_data, 
                                      quotes[quotes_cols], 
                                      how = 'left',
                                      left_on = ['NCUSIP', 'FirstTradeDate'],
                                      right_on = ['NCUSIP', 'date'])
            new_col_names = {'PRC': 'ClosePrice',
                             'OPENPRC': 'OpenPrice'}
            self.full_data = self.full_data.rename(columns = new_col_names)
            self.full_data = self.full_data.drop(columns = ['date'])
        else:
            self.full_data = pd.merge(self.full_data, 
                                      quotes[quotes_cols], 
                                      how = 'left',
                                      left_on = ['NCUSIP', 'FirstTradeDate'],
                                      right_on = ['NCUSIP', 'date'])
            quotes_cols.remove('OPENPRC')
            new_col_names = {'PRC': 'FirstClosePrice',
                             'OPENPRC': 'OpenPrice'}
            self.full_data = self.full_data.rename(columns = new_col_names)
            self.full_data = self.full_data.drop(columns = ['date'])
# =========================              
            self.full_data = pd.merge(self.full_data, 
                                      quotes[quotes_cols], 
                                      how = 'left',
                                      left_on = ['NCUSIP', 'SecondTradeDate'],
                                      right_on = ['NCUSIP', 'date'])
            new_col_names = {'PRC': 'SecondClosePrice'}
            self.full_data = self.full_data.rename(columns = new_col_names)
            self.full_data = self.full_data.drop(columns = ['date'])
# =========================         
            self.full_data = pd.merge(self.full_data, 
                                      quotes[quotes_cols], 
                                      how = 'left',
                                      left_on = ['NCUSIP', 'LastTradeDateWK'],
                                      right_on = ['NCUSIP', 'date'])
            new_col_names = {'PRC': 'LastClosePriceWK'}
            self.full_data = self.full_data.rename(columns = new_col_names)
            self.full_data = self.full_data.drop(columns = ['date'])
# =========================             
            close_prc = np.where((pd.isnull(self.full_data['FirstClosePrice']) == True) &\
                                 (pd.isnull(self.full_data['SecondClosePrice']) == False),
                                 self.full_data['SecondClosePrice'],
                                 self.full_data['FirstClosePrice'])
            self.full_data['ClosePrice'] = close_prc
            
            close_prc_ext = np.where((pd.isnull(self.full_data['ClosePrice']) == True) &\
                                     (pd.isnull(self.full_data['LastClosePriceWK']) == False),
                                     self.full_data['LastClosePriceWK'],
                                     self.full_data['ClosePrice'])
            self.full_data['ClosePrice'] = close_prc_ext
           
            

            


        
    