# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 22:23:49 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
from DataPreparation import *

feat_eng_obj_file = 'Base_FeatureEngineering.pkl'

model_cols = ['InitialReturn', 'UnderwriterRank', 'TechDummy',
              'TotalAssets', 'AMEX', 'NASDQ', 'NYSE']


class FeatureEngineering:
    def __init__(self, prep_obj):
        self.prep_obj = prep_obj
        self.full_data = prep_obj.full_data
        cols = self.full_data.columns.to_list()
        cols.sort()
        print(cols)
        
    def preprocessing(self):
        total_assets = self.full_data['TotalAssetsBeforeTheOfferingMil']
        total_assets = total_assets.str.replace(',', '')
        total_assets = total_assets.astype(float)
        total_assets = total_assets * 1000000
        self.full_data['TotalAssets'] = total_assets
# =========================
        cpi_base_year = self.prep_obj.cpi_base_year
        cpi_disc_factor = cpi_base_year / self.full_data['CPI']
        self.full_data['CPI_DiscountFactor'] = cpi_disc_factor
# =========================
        sector_sdc = self.prep_obj.ipo_port_data['HighTechIndustryGroup']
        sector = pd.Series([])
        for index, value in sector_sdc.items():
            if pd.isnull(value) == False:
                char_pos = value.find('|')
                if char_pos != -1:
                    sector_name = value[:char_pos]
                else:
                    sector_name = value
            else:
                sector_name = np.nan
            sector.loc[index] = sector_name
            
        self.prep_obj.ipo_port_data['Sector'] = sector
        
        merge_cols = ['DealNumber', 'Sector']
        self.full_data = pd.merge(self.full_data,
                                  self.prep_obj.ipo_port_data[merge_cols],
                                  how = 'left',
                                  on = 'DealNumber')
                
    def firm_features(self):
        close_prc = self.full_data['ClosePrice']
        offer_prc = self.full_data['OfferPrice']
        init_ret = (close_prc / offer_prc) -1
        init_ret = init_ret * 100
        
        self.full_data['InitialReturn'] = init_ret
# =========================        
        treshold = 90
        rank = np.where(self.full_data['Matching_Level'] >= treshold,
                        self.full_data['MeanRank'],
                        np.nan)
        
        self.full_data['UnderwriterRank'] = rank
# =========================
        tech = 1
        non_tech = 0
        sector = self.full_data['Sector']
        tech_dummy = np.where(sector == 'Non-Hitech',
                              non_tech,
                              tech)
        nan_ident = pd.isnull(self.full_data['Sector']) == True
        tech_nan_idx = self.full_data.loc[nan_ident].index
        self.full_data['TechDummy'] = tech_dummy
        self.full_data.loc[tech_nan_idx, 'TechDummy'] = np.nan
# =========================
        disc_fact = self.full_data['CPI_DiscountFactor']
        total_assets = self.full_data['TotalAssets']
        total_assets = total_assets * disc_fact
        
        self.full_data['TotalAssets'] = total_assets
# =========================
        shares_filed = self.full_data['SharesFiled']
        mean_prc_rg = self.full_data['MeanPriceRange']
        exp_pro = shares_filed * mean_prc_rg
        exp_pro = exp_pro * disc_fact
        
        shares_offered = self.full_data['SharesOfferedSumOfAllMkts']
        act_pro = shares_offered * offer_prc
        act_pro = act_pro * disc_fact
# =========================
        nan_ident = pd.isnull(self.full_data['ExchangeWhereIssuWillBeLi']) == True
        exchange_nan_idx = self.full_data.loc[nan_ident].index
        exchange_cols = ['AMEX', 'NASDQ', 'NYSE']
        self.full_data.loc[exchange_nan_idx, exchange_cols] = np.nan
        
        print('kdjafö')

        
