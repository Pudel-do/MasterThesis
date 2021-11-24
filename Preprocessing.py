# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:40:40 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import pickle
import sys

exit_message = 'Download of CRSP data for quotes and returns necessary due to time period adjustments'


input_path = r'C:\Users\Matthias Pudel\OneDrive\Studium\Master\Master Thesis\Empirical Evidence\Code\Input Data'
sdc_raw_file = 'sdc_full_raw.csv'
sdc_total_assets_file = 'sdc_total_assets.csv'
quotes_file = 'Quotes.csv'
uwriter_rank_file = 'UnderwriterRank.xlsx'
cpi_file = 'CPI_85_20.xlsx'

output_path = r'C:\Users\Matthias Pudel\OneDrive\Studium\Master\Master Thesis\Empirical Evidence\Code\Output Data'
process_obj_file = 'Preprocessing.pkl'

cols = []


def save_obj(obj, path, filename):
    with open(path + '\\' + filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def get_object(path, filename):
    with open(path + '\\' + filename, 'rb') as f:
        return pickle.load(f)


class Preprocessing:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        
    def rough_preprocessing(self):
        sdc_raw = pd.read_csv(input_path+'\\'+ sdc_raw_file,
                              parse_dates = ['IssueDate', 'FilingDate'])
        
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
        
    def build_aux_vars(self, adj_time):
        self.sdc['NCUSIP'] = self.sdc['CUSIP9'].str[:8]
        start_year = self.start_date.strftime('%Y')
        end_year = self.end_date.strftime('%Y')
        ncusip_file = f'NCUSIP_{start_year}_{end_year}.txt'
        self.start_year = start_year
        self.end_year = end_year
        
        if adj_time == True:
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
        
    def ext_preprocessing(self):

    def data_merging(self):
        cpi_data = pd.read_excel(input_path+'\\'+cpi_file,
                                 engine = 'openpyxl',
                                 names = ['CPI_Date', 'CPI'])
        cpi_data['CPI_Date'] = cpi_data['CPI_Date'].dt.strftime('%Y-%m')
        
        self.data_full = pd.merge(self.sdc,
                                  cpi_data,
                                  how = 'left',
                                  left_on = 'CPI_MergeDate',
                                  right_on = 'CPI_Date')
        
        
        
        
