# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:40:40 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pickle
import re
import sys

exit_message = 'Download of CRSP data for quotes and returns necessary due to time period adjustments'


input_path = r'C:\Users\Matthias Pudel\OneDrive\Studium\Master\Master Thesis\Empirical Evidence\Code\Input Data'
sdc_raw_file = 'sdc_full_raw.csv'
sdc_total_assets_file = 'sdc_total_assets.csv'
quotes_file = 'Quotes.csv'
uwriter_file = 'UnderwriterRank.xlsx'
cpi_file = 'CPI_85_20.xlsx'

output_path = r'C:\Users\Matthias Pudel\OneDrive\Studium\Master\Master Thesis\Empirical Evidence\Code\Output Data'
process_obj_file = 'DataPreparation.pkl'

cols = []


def save_obj(obj, path, filename):
    with open(path + '\\' + filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def get_object(path, filename):
    with open(path + '\\' + filename, 'rb') as f:
        return pickle.load(f)


class DataPreparation:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        
    def rough_preprocessing(self):
        sdc_raw = pd.read_csv(input_path+'\\'+ sdc_raw_file,
                              parse_dates = ['IssueDate', 'FilingDate'])
        
        sdc_cols = sdc_raw.columns.to_list()
        sdc_cols.sort()
        print(sdc_cols)
        
        mask =  (sdc_raw['IssueDate'] >= self.start_date) &\
                (sdc_raw['IssueDate'] <= self.end_date) &\
                (sdc_raw['IPO'] == 'Yes') &\
                (sdc_raw['ADR'] == 'No') &\
                (sdc_raw['CEF'] == 'No') &\
                (sdc_raw['Units'] == 'No') &\
                (pd.isna(sdc_raw['REIT']) == True) &\
                (pd.isna(sdc_raw['CUSIP9']) == False) &\
                (sdc_raw['OfferPrice'] >= 5)
               
        sdc = sdc_raw.loc[mask]
        self.sdc = sdc
        
    def build_aux_vars(self, update_time):
        self.sdc['NCUSIP'] = self.sdc['CUSIP9'].str[:8]
        start_year = self.start_date.strftime('%Y')
        end_year = self.end_date.strftime('%Y')
        ncusip_file = f'NCUSIP_{start_year}_{end_year}.txt'
        self.start_year = start_year
        self.end_year = end_year
        
        if update_time == True:
            self.sdc['NCUSIP'].to_csv(output_path+'\\'+ncusip_file,
                                      header = False,
                                      index = False)
            sys.exit(exit_message)
        
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
        self.sdc['LastTradeDateWK'] = last_trade_wk_dt
        
        cpi_merge_dt = self.sdc['IssueDate'].dt.strftime('%Y-%m')
        self.sdc['CPI_MergeDate'] = cpi_merge_dt
        
        registration_days = (self.sdc['IssueDate'] - self.sdc['FilingDate'])
        registration_days = registration_days.dt.days
        self.sdc['RegistrationDays'] = registration_days
        
    def extended_preprocessing(self):
        
        mean_price_rg = np.where(pd.isnull(self.sdc['OriginalMiddleOfFilingPriceRange']) == True,
                                          self.sdc['AmendedMiddleOfFilingPrice'],
                                          self.sdc['OriginalMiddleOfFilingPriceRange'])
                                
        min_price_rg = np.where(pd.isnull(self.sdc['OriginalLowFilingPrice']) == True,
                                          self.sdc['AmendedLowFilingPrice'],
                                          self.sdc['OriginalLowFilingPrice'])
                                
        max_price_rg = np.where(pd.isnull(self.sdc['OriginalHighFilingPrice']) == True,
                                          self.sdc['AmendedHighFilingPrice'],
                                          self.sdc['OriginalHighFilingPrice'])
        
        shares_filed = np.where(pd.isnull(self.sdc['SharesFiledSumOfAllMkts']) == True,
                                          self.sdc['AmendedShsFiledSumOfAllMkts'],
                                          self.sdc['SharesFiledSumOfAllMkts'])
        
        shares_filed = pd.Series(shares_filed,
                                 index = self.sdc['SharesFiledSumOfAllMkts'].index)
        shares_filed = shares_filed.str.replace(',', '')
        shares_filed = shares_filed.astype(float)
        
        self.sdc['MeanPriceRange'] = mean_price_rg
        self.sdc['MinPriceRange'] = min_price_rg
        self.sdc['MaxPriceRange'] = max_price_rg
        self.sdc['SharesFiled'] = shares_filed
        
        ritter_data = pd.read_excel(input_path+'\\'+uwriter_file,
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
        
        if self.end_year == '2019':
            rank_data = name.join(rank_2001_2020)
        else:
            rank_data = name.join(rank_1985_2000)
          
        rank_data = rank_data.replace(-9,np.nan)
        rank_data['MeanRank'] = rank_data.mean(axis = 1)
        rank_data['MinRank'] = rank_data.min(axis = 1)
        rank_data['MaxRank'] = rank_data.max(axis = 1)

        sdc_uwriter = self.sdc['LeadManagersLongName']
        match_results = pd.DataFrame()
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
            
            match_results.loc[index, 'SDC_Uwriter'] = sdc_name
            match_results.loc[index, 'Ritter_Uwritter'] = matching[0]
            match_results.loc[index, 'Matching_Level'] = matching[1]
            
        self.sdc = self.sdc.join(match_results, how = 'left')
        
    def data_merging(self):
        quotes = pd.read_csv(input_path+'\\'+quotes_file)
        # self.base_data = pd.merge(self.sdc, quotes, 
        #                           how = 'left',
        #                           left_on = [])

        
        
            


        
    