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

class DataPreparation:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        
    def rough_preprocessing(self):
        sdc_raw = pd.read_csv(input_path+'\\'+ sdc_raw_file,
                              parse_dates = ['IssueDate', 'FilingDate'])
        
        dup_ident = ['Issuer']
        mask = (sdc_raw['IssueDate'] >= self.start_date) &\
                (sdc_raw['IssueDate'] <= self.end_date) &\
                (sdc_raw['FilingDate'] <= self.end_date) &\
                (pd.isnull(sdc_raw['IssueDate']) == False) &\
                (pd.isnull(sdc_raw['FilingDate']) == False) &\
                (sdc_raw['IPO'] == 'Yes') &\
                (sdc_raw['ADR'] == 'No') &\
                (sdc_raw['CEF'] == 'No') &\
                (sdc_raw['Units'] == 'No') &\
                (pd.isnull(sdc_raw['REIT']) == True) &\
                (pd.isnull(sdc_raw['CUSIP9']) == False) &\
                (sdc_raw['OfferPrice'] >= 5)
        
        sdc = sdc_raw.loc[mask]
        sdc = sdc.drop_duplicates(subset = dup_ident)
        self.sdc = sdc
# =========================  
        ipo_port_start = self.start_date - DateOffset(years=1)
        mask_ipo_port = (sdc_raw['IssueDate'] >= ipo_port_start) &\
                        (sdc_raw['IssueDate'] <= self.end_date) &\
                        (sdc_raw['FilingDate'] <= self.end_date) &\
                        (pd.isnull(sdc_raw['IssueDate']) == False) &\
                        (pd.isnull(sdc_raw['FilingDate']) == False) &\
                        (sdc_raw['IPO'] == 'Yes') &\
                        (sdc_raw['ADR'] == 'No') &\
                        (sdc_raw['CEF'] == 'No') &\
                        (sdc_raw['Units'] == 'No') &\
                        (pd.isna(sdc_raw['REIT']) == True) &\
                        (pd.isna(sdc_raw['CUSIP9']) == False) &\
                        (sdc_raw['OfferPrice'] >= 5)
                        
        ipo_port_data = sdc_raw.loc[mask_ipo_port]
        ipo_port_data = ipo_port_data.drop_duplicates(subset = dup_ident)
        self.ipo_port_data = ipo_port_data
        
    def build_aux_vars(self, adj_time_range):
        start_year = self.start_date.strftime('%Y')
        end_year = self.end_date.strftime('%Y')
        ncusip_quotes = self.sdc['CUSIP9'].str[:8]
        ncusip_port = self.ipo_port_data['CUSIP9'].str[:8]
        ncusip_quote_file = f'NCUSIP_{start_year}_{end_year}_Quotes.txt'
        ncusip_port_file = f'NCUSIP_{start_year}_{end_year}_PortfolioReturns.txt'
        self.sdc['NCUSIP'] = ncusip_quotes
        self.ipo_port_data['NCUSIP'] = ncusip_port
        self.start_year = start_year
        self.end_year = end_year
  
        if adj_time_range == True:
            ncusip_quotes.to_csv(output_path+'\\'+ncusip_quote_file,
                                 header = False,
                                 index = False)
            
            ncusip_port.to_csv(output_path+'\\'+ncusip_port_file,
                               header = False,
                               index = False)
            
            sys.exit(exit_message)
# =========================        
        onebday_offset = pd.offsets.BusinessDay(1)
        twobday_offset = pd.offsets.BusinessDay(2)
        first_trade_dt = self.sdc['IssueDate'] + onebday_offset
        second_trade_dt = self.sdc['IssueDate'] + twobday_offset
        
        last_trade_wk_dt = pd.Series([])
        for index, value in first_trade_dt.items():
            day_rg = pd.date_range(start = value, 
                                   end = value + pd.Timedelta('7 days'),
                                   freq='W-MON')
            last_bday = day_rg[-1] - onebday_offset
            if last_bday == value:
                last_bday = last_bday + pd.Timedelta('7 days')
            last_trade_wk_dt.loc[index] = last_bday
            
        self.sdc['FirstTradeDate'] = first_trade_dt
        self.sdc['SecondTradeDate'] = second_trade_dt
        self.sdc['LastTradeDateWK'] = last_trade_wk_dt
# =========================        
        cpi_merge_dt = self.sdc['IssueDate'].dt.strftime('%Y-%m')
        self.sdc['CPI_MergeDate'] = cpi_merge_dt
# =========================        
        registration_days = (self.sdc['IssueDate'] - self.sdc['FilingDate'])
        registration_days = registration_days.dt.days
        self.sdc['RegistrationDays'] = registration_days
        
    def extended_preprocessing(self, adj_uw_matching):
        exchange = self.sdc['ExchangeWhereIssuWillBeLi']
        exchange_dummies = pd.get_dummies(exchange)
        exchange_dummies = exchange_dummies.astype(float)
        exchange_cols = ['AMEX', 'NASDQ', 'NYSE']
        
        self.sdc = pd.concat([self.sdc, 
                              exchange_dummies[exchange_cols]], 
                             axis = 1)
# =========================        
        mean_prc_rg = np.where(pd.isnull(self.sdc['OriginalMiddleOfFilingPriceRange']) == True,
                                          self.sdc['AmendedMiddleOfFilingPrice'],
                                          self.sdc['OriginalMiddleOfFilingPriceRange'])
                                
        min_prc_rg = np.where(pd.isnull(self.sdc['OriginalLowFilingPrice']) == True,
                                          self.sdc['AmendedLowFilingPrice'],
                                          self.sdc['OriginalLowFilingPrice'])
                                
        max_prc_rg = np.where(pd.isnull(self.sdc['OriginalHighFilingPrice']) == True,
                                          self.sdc['AmendedHighFilingPrice'],
                                          self.sdc['OriginalHighFilingPrice'])
        
        self.sdc['MeanPriceRange'] = mean_prc_rg
        self.sdc['MinPriceRange'] = min_prc_rg
        self.sdc['MaxPriceRange'] = max_prc_rg
# =========================         
        shares_filed = np.where(pd.isnull(self.sdc['SharesFiledSumOfAllMkts']) == True,
                                          self.sdc['AmendedShsFiledSumOfAllMkts'],
                                          self.sdc['SharesFiledSumOfAllMkts'])
        
        shares_filed_idx = self.sdc['SharesFiledSumOfAllMkts'].index
        shares_filed = pd.Series(shares_filed, index = shares_filed_idx)
        shares_filed = shares_filed.str.replace(',', '')
        shares_filed = shares_filed.astype(float)
        self.sdc['SharesFiled'] = shares_filed
# =========================         
        ritter_data = pd.read_excel(input_path+'\\'+uw_file,
                                            engine = 'openpyxl',
                                            sheet_name = 'UnderwriterRank')
        
        name = ritter_data['Underwriter Name']
        name = name[:-2]
        name = name.map(lambda x: re.sub(r'\W+', '', x))
        name = name.str.casefold() 
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

        sdc_uwriter = self.sdc['LeadManagersLongName']
        match_results = pd.DataFrame()
        if adj_uw_matching == True:
            for index, value in sdc_uwriter.items():
                print(index)
                char_pos = value.find('|')
                if char_pos != -1:
                    sdc_name = value[:char_pos]
                    sdc_name = re.sub(r'\W+', '', sdc_name)
                    sdc_name = sdc_name.casefold()
                else:
                    sdc_name = value
                    sdc_name = re.sub(r'\W+', '', sdc_name)
                    sdc_name = sdc_name.casefold()
                
                matching = process.extractOne(sdc_name, 
                                              rank_data['Name'])
            
                match_results.loc[index, 'SDC_Name'] = sdc_name
                match_results.loc[index, 'Ritter_Name'] = matching[0]
                match_results.loc[index, 'Matching_Level'] = matching[1]
                
            match_results.to_csv(output_path+'\\'+uw_matching_file)
        else:
            index_col = ['Unnamed: 0']
            match_results = pd.read_csv(output_path+'\\'+uw_matching_file,
                                        index_col = index_col)
            
        self.sdc = self.sdc.join(match_results, how = 'left')
        
    def data_merging(self, adj_close_price):
        total_assets = pd.read_csv(input_path+'\\'+total_assets_file)
        self.full_data = pd.merge(self.sdc,
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

            


        
    