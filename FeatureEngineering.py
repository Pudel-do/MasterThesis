# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 22:23:49 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
from GetData import *

model_cols = ['InitialReturn', 'UnderwriterRank', 'TotalAssets',
              'TechDummy', 'AMEX', 'NASDQ', 'NYSE',
              'MarketReturn', 'SectorReturn',
              'PriceRevision', 'PriceRevisionSlopeDummy',
              'PriceRevisionMax', 'PriceRevisionMin',
              'SharesRevision', 'SharesRevisionSlopeDummy',
              'ProceedsRevision', 'ProceedsRevisionSlopeDummy',
              'ProceedsRevisionMax', 'ProceedsRevisionMin',
              'RegistrationDays']

class FeatureEngineering:
    def __init__(self, prep_obj, scale_factor):
        self.prep_obj = prep_obj
        self.full_data = prep_obj.full_data
        self.scale = scale_factor
        
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
        self.full_data['InitialReturn'] = init_ret * self.scale
# =========================        
        treshold = 90
        rank = np.where(self.full_data['Matching_Level'] >= treshold,
                        self.full_data['MeanRank'],
                        np.nan)
        
        self.full_data['UnderwriterRank'] = rank
# =========================
        tech_id = 1
        non_tech_id = 0
        sector = self.full_data['Sector']
        tech_dummy = np.where(sector == 'Non-Hitech',
                              non_tech_id,
                              tech_id)
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
        mean_prc_rg = self.full_data['MeanPriceRange']
        min_prc_rg = self.full_data['MinPriceRange']
        max_prc_rg = self.full_data['MaxPriceRange']
        shares_filed = self.full_data['SharesFiled']
        
        exp_pro = shares_filed * mean_prc_rg
        exp_pro = exp_pro * disc_fact
        exp_pro_min = shares_filed * min_prc_rg
        exp_pro_min = exp_pro_min * disc_fact
        exp_pro_max = shares_filed * max_prc_rg
        exp_pro_max = exp_pro_max * disc_fact
        
        shares_offered = self.full_data['SharesOfferedSumOfAllMkts']
        act_pro = shares_offered * offer_prc
        act_pro = act_pro * disc_fact
        
        self.full_data['SharesOffered'] = shares_offered
        self.full_data['ExpectedProceeds'] = exp_pro
        self.full_data['ExpectedProceedsMin'] = exp_pro_min
        self.full_data['ExpectedProceedsMax'] = exp_pro_max
        self.full_data['ActualProceeds'] = act_pro
# =========================
        nan_ident = pd.isnull(self.full_data['ExchangeWhereIssuWillBeLi']) == True
        exchange_nan_idx = self.full_data.loc[nan_ident].index
        exchange_cols = ['AMEX', 'NASDQ', 'NYSE']
        self.full_data.loc[exchange_nan_idx, exchange_cols] = np.nan
        
    def public_features(self, index_weight, port_days, adj_public):
        if adj_public == True:
            ipo_rets_data = pd.read_csv(input_path+'\\'+returns_file,
                                        usecols = ['date', 'NCUSIP', 'RETX'],
                                        parse_dates = ['date'],
                                        index_col = ['date'])
        
            adj_col = ipo_rets_data['RETX']
            ipo_rets = ipo_rets_data.loc[adj_col.str.isalpha() == False]
            ipo_rets['RETX'] = ipo_rets['RETX'].astype(float)
        
            index_rets_data = pd.read_csv(input_path+'\\'+index_returns_file,
                                          parse_dates = ['DATE'],
                                          index_col = 'DATE')
        
            if index_weight == 'Equal':
                index_rets = index_rets_data['ewretx']
            else:
                index_rets = index_rets_data['vwretx']
            
            port_data = self.prep_obj.ipo_port_data
            public_features = pd.DataFrame()
            for index, row in self.full_data.iterrows():
                print(index)
                if port_days != None:
                    dt_offset = pd.offsets.BusinessDay(port_days)
                    start_date = row['IssueDate'] - dt_offset
                    end_date = row['IssueDate']
                    port_period = pd.date_range(start = start_date,
                                                end = end_date,
                                                freq = 'B')
            
                    port_period = pd.DataFrame(port_period)
                    port_period.columns = ['Date']
                    port_period = port_period.set_index('Date')  
                else:
                    start_date = row['FilingDate']
                    end_date = row['IssueDate']
                    port_period = pd.date_range(start = start_date,
                                                end = end_date,
                                                freq = 'B')
                
                    port_period = pd.DataFrame(port_period)
                    port_period.columns = ['Date']
                    port_period = port_period.set_index('Date')
# =========================
                index_ret = port_period.join(index_rets, how = 'left')
                index_ret = index_ret.sum()
                index_ret = float(index_ret)
                index_ret = index_ret * self.scale
            
                col_name = ['MarketReturn']
                public_features.loc[index, col_name] = index_ret
# =========================
                last_year = row['IssueDate'] - DateOffset(years=1)
                last_month = row['IssueDate'] - DateOffset(months=1)
                sector = row['Sector']
                col_name = ['SectorReturn']
                
                if pd.isnull(sector) == False:
                    mask = (port_data['IssueDate'] >= last_year) &\
                            (port_data['IssueDate'] <= last_month) &\
                            (port_data['Sector'] == sector)
                   
                    port_comp = port_data.loc[mask]
                    port_comp = port_comp['NCUSIP']
                    port_comp = port_comp.drop_duplicates()    
                    
                    if port_comp.empty == False:
                        match_dt = port_period.join(ipo_rets, how = 'inner')
                        comp_ident = match_dt['NCUSIP'].isin(port_comp)
                        comp_rets = match_dt.loc[comp_ident]
                        
                        noa = len(port_comp)
                        sector_ret = comp_rets['RETX'].sum()
                        sector_ret = 1/noa * sector_ret
                        sector_ret = sector_ret * self.scale
                        public_features.loc[index, col_name] = sector_ret
                    else:
                        public_features.loc[index, col_name] = np.nan
                else:
                    public_features.loc[index, col_name] = np.nan
            
            public_features.to_csv(output_path+'\\'+public_feat_file)
            self.full_data = self.full_data.join(public_features)
        else:
            public_features = pd.read_csv(output_path+'\\'+public_feat_file,
                                    index_col = 'Unnamed: 0')
            
            self.full_data = self.full_data.join(public_features)
            
    def private_features(self):
        offer_prc = self.full_data['OfferPrice']
        mid_prc_rg = self.full_data['MeanPriceRange']
        prc_rev = (offer_prc / mid_prc_rg) -1
        prc_rev = prc_rev * self.scale
        self.full_data['PriceRevision'] = prc_rev
        
        prc_rev_slope = get_slope_dummy(prc_rev)
        col_name = 'PriceRevisionSlopeDummy'
        self.full_data[col_name] = prc_rev_slope
        
        max_prc_rg = self.full_data['MaxPriceRange']
        prc_rev_max = get_dummy_max(offer_prc, max_prc_rg)
        col_name = 'PriceRevisionMax'
        self.full_data[col_name] = prc_rev_max
        
        min_prc_rg = self.full_data['MinPriceRange']
        prc_rev_min = get_dummy_min(offer_prc, min_prc_rg)
        col_name = 'PriceRevisionMin'
        self.full_data[col_name] = prc_rev_min
# =========================
        shares_off = self.full_data['SharesOffered']
        shares_fil = self.full_data['SharesFiled']
        shares_rev = (shares_off / shares_fil) -1
        shares_rev = shares_rev * self.scale
        self.full_data['SharesRevision'] = shares_rev
        
        shares_rev_slope = get_slope_dummy(shares_rev)
        col_name = 'SharesRevisionSlopeDummy'
        self.full_data[col_name] = shares_rev_slope        
# =========================
        act_pro = self.full_data['ActualProceeds']
        exp_pro = self.full_data['ExpectedProceeds']
        pro_rev = (act_pro / exp_pro) -1
        pro_rev = pro_rev * self.scale
        self.full_data['ProceedsRevision'] = pro_rev
        
        pro_rev_slope = get_slope_dummy(pro_rev)
        col_name = 'ProceedsRevisionSlopeDummy'
        self.full_data[col_name] = pro_rev_slope
        
        exp_pro_max = self.full_data['ExpectedProceedsMax']
        pro_rev_max = get_dummy_max(act_pro, exp_pro_max)
        col_name = ['ProceedsRevisionMax']
        self.full_data[col_name] = pro_rev_max
        
        exp_pro_min = self.full_data['ExpectedProceedsMin']
        pro_rev_min = get_dummy_min(act_pro, exp_pro_min)
        col_name = ['ProceedsRevisionMin']
        self.full_data[col_name] = pro_rev_min
# =========================        
        self.model_data = self.full_data[model_cols]
        
        
        
        
        
        
        
        
        
        
        
                
                

        

        
        
        
        
        
