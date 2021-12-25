# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 22:23:49 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset
from GetData import *
import re 
from re import search
from tabulate import tabulate 
import warnings
import os
warnings.filterwarnings("ignore")

model_cols = ['InitialReturn', 
              'UnderwriterRank', 'TotalAssets',
              'TechSector', 'AMEX', 'NASDQ', 'NYSE',
              'MarketReturn', 'MarketReturnSlopeDummy',
              'SectorReturn', 'SectorReturnSlopeDummy',
              'PriceRevision', 'PriceRevisionSlopeDummy',
              'PriceRevisionMaxDummy', 'PriceRevisionMinDummy',
              'PriceRevisionMaxSlopeDummy', 'PriceRevisionMinSlopeDummy',
              'SharesRevision', 'SharesRevisionSlopeDummy',
              'ProceedsRevision', 'ProceedsRevisionSlopeDummy',
              'ProceedsRevisionMaxDummy', 'ProceedsRevisionMinDummy',
              'ProceedsRevisionMaxSlopeDummy', 'ProceedsRevisionMinSlopeDummy',
              'RegistrationDays', 'IssueDate',
              ]


class FeatureEngineering:
    def __init__(self, prep_obj, scale_factor):
        self.prep_obj = prep_obj
        self.full_data = prep_obj.full_data
        self.scale = scale_factor
        
    def preprocessing(self):
        total_assets = self.full_data['TotalAssetsBeforeTheOfferingMil']
        total_assets = total_assets.str.replace(',', '')
        total_assets = total_assets.astype(float)
        self.full_data['TotalAssets'] = total_assets
# =========================
        cpi_base_year = self.prep_obj.cpi_base_year
        cpi_disc_factor = cpi_base_year / self.full_data['CPI']
        self.full_data['CPI_DiscountFactor'] = cpi_disc_factor
# =========================
        sector_sdc = self.prep_obj.port_data['HighTechIndustryGroup']
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
            
        self.prep_obj.port_data['Sector'] = sector
        merge_cols = ['DealNumber', 'Sector']
        self.full_data = pd.merge(self.full_data,
                                  self.prep_obj.port_data[merge_cols],
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
        self.full_data['TechSector'] = tech_dummy
        self.full_data.loc[tech_nan_idx, 'TechSector'] = np.nan
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
        
    def public_features(self, index_weight, port_days, adj_portfolio_rets):
        if adj_portfolio_rets == True:
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
                
            iter_count = 0
            progress = 100
            print('Public features construction started')
            port_data = self.prep_obj.port_data
            public_features = pd.DataFrame()
            for index, row in self.full_data.iterrows():
                deal_num = row['DealNumber']
                col_name = ['DealNumber']
                public_features.loc[index, col_name] = deal_num
                iter_count += 1
                if iter_count % progress == 0:
                    print(f'Progress: {iter_count} items')
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
                    
            market_ret = public_features['MarketReturn']
            market_ret_slope = get_slope_dummy(market_ret)
            col_name = 'MarketReturnSlopeDummy'
            public_features[col_name] = market_ret_slope
            
            sector_ret = public_features['SectorReturn']
            sector_ret_slope = get_slope_dummy(sector_ret)
            col_name = 'SectorReturnSlopeDummy'
            public_features[col_name] = sector_ret_slope
            
            start_year = self.prep_obj.start_year
            end_year = self.prep_obj.end_year
            file_name = public_feat_file
            file_name = f'{start_year}_{end_year}_{file_name}'
            public_features.to_csv(output_path+'\\'+file_name, index = False)
            print('Public features construction finished')

        else:
            start_year = self.prep_obj.start_year
            end_year = self.prep_obj.end_year
            file_name = public_feat_file
            file_name = f'{start_year}_{end_year}_{file_name}'
            public_features = pd.read_csv(output_path+'\\'+file_name)
        
        self.full_data = pd.merge(self.full_data,
                                  public_features,
                                  how = 'left',
                                  on = 'DealNumber')
            
    def private_features(self, adj_words_revison):
        if adj_words_revison == True:
            init_prosp_path = prospectus_path+'\\'+'Initial_Prospects'
            final_prosp_path = prospectus_path+'\\'+'Final_Prospects'
            
            for index, row in self.full_data.iterrows():
                if pd.isnull(row['IP_FileName']) == False:
                    init_file_name = row['IP_FileName']
                    init_path = init_prosp_path+'\\'+init_file_name
                    try:
                        file = open(os.path.join(init_path, init_file_name),'r')
                        self.full_data.loc[index, 'IP_FileMatch'] = True
                    except:
                        self.full_data.loc[index, 'IP_FileMatch'] = np.nan
                else:
                    self.full_data.loc[index, 'IP_FileMatch'] = np.nan

                if pd.isnull(row['FP_FileName']) == False:
                    final_file_name = row['FP_FileName']
                    final_path = final_prosp_path+'\\'+final_file_name
                    try:
                        file = open(os.path.join(final_path, final_file_name),'r')
                        self.full_data.loc[index, 'FP_FileMatch'] = True
                    except:
                        self.full_data.loc[index, 'FP_FileMatch'] = np.nan
                else:
                    self.full_data.loc[index, 'FP_FileMatch'] = np.nan
# =========================
        offer_prc = self.full_data['OfferPrice']
        mid_prc_rg = self.full_data['MeanPriceRange']
        prc_rev = (offer_prc / mid_prc_rg) -1
        prc_rev = prc_rev * self.scale
        self.full_data['PriceRevision'] = prc_rev
        
        prc_rev_slp = get_SlopeDummy(prc_rev)
        col_name = 'PriceRevisionSlopeDummy'
        self.full_data[col_name] = prc_rev_slp
        
        max_prc_rg = self.full_data['MaxPriceRange']
        prc_rev_max = get_DummyMax(offer_prc, max_prc_rg)
        col_name = 'PriceRevisionMaxDummy'
        self.full_data[col_name] = prc_rev_max
        
        prc_rev_maxslp = get_SlopeDummyBounds(prc_rev_max, prc_rev)
        col_name = ['PriceRevisionMaxSlopeDummy']
        self.full_data[col_name] = prc_rev_maxslp
        
        min_prc_rg = self.full_data['MinPriceRange']
        prc_rev_min = get_DummyMin(offer_prc, min_prc_rg)
        col_name = 'PriceRevisionMinDummy'
        self.full_data[col_name] = prc_rev_min
        
        prc_rev_minslp = get_SlopeDummyBounds(prc_rev_min, prc_rev)
        col_name = ['PriceRevisionMinSlopeDummy']
        self.full_data[col_name] = prc_rev_minslp
# =========================
        shares_off = self.full_data['SharesOffered']
        shares_fil = self.full_data['SharesFiled']
        shares_rev = (shares_off / shares_fil) -1
        shares_rev = shares_rev * self.scale
        self.full_data['SharesRevision'] = shares_rev
        
        shares_rev_slp = get_SlopeDummy(shares_rev)
        col_name = 'SharesRevisionSlopeDummy'
        self.full_data[col_name] = shares_rev_slp        
# =========================
        act_pro = self.full_data['ActualProceeds']
        exp_pro = self.full_data['ExpectedProceeds']
        pro_rev = (act_pro / exp_pro) -1
        pro_rev = pro_rev * self.scale
        self.full_data['ProceedsRevision'] = pro_rev
        
        pro_rev_slp = get_SlopeDummy(pro_rev)
        col_name = 'ProceedsRevisionSlopeDummy'
        self.full_data[col_name] = pro_rev_slp
        
        exp_pro_max = self.full_data['ExpectedProceedsMax']
        pro_rev_max = get_DummyMax(act_pro, exp_pro_max)
        col_name = ['ProceedsRevisionMaxDummy']
        self.full_data[col_name] = pro_rev_max
        
        pro_rev_maxslp = get_SlopeDummyBounds(pro_rev_max, pro_rev)
        col_name = ['ProceedsRevisionMaxSlopeDummy']
        self.full_data[col_name] = pro_rev_maxslp
        
        exp_pro_min = self.full_data['ExpectedProceedsMin']
        pro_rev_min = get_DummyMin(act_pro, exp_pro_min)
        col_name = ['ProceedsRevisionMinDummy']
        self.full_data[col_name] = pro_rev_min
        
        pro_rev_minslp = get_SlopeDummyBounds(pro_rev_min, pro_rev)
        col_name = ['ProceedsRevisionMinSlopeDummy']
        self.full_data[col_name] = pro_rev_minslp
# =========================
    def outlier_adjustment(self, whisker_factor, adj_outliers):
        if adj_outliers == True:
            drop_cols = ['Dummy', 'SlopeDummy', 'Min', 'Max',
                         'UnderwriterRank', 'TechSector', 
                         'AMEX', 'NASDQ', 'NYSE', 'IssueDate'
                      ]
            outlier_cols = []
            for col in model_cols:
                if not any(char in col for char in drop_cols):
                    outlier_cols.append(col)
        
            idx = outlier_cols
            measure_cols = [
                    'Mean', 'Median',
                    'Min', 'Max',
                    'StandDev',
                    'LowWhisker', 'UpWhisker'
                    ]
        
            stat_data = self.full_data[outlier_cols]
            mean = stat_data.mean()
            median = stat_data.median()
            minimum = stat_data.min()
            maximum = stat_data.max()
            std = stat_data.std()
            nobs = len(stat_data)
            measures = [mean, median, minimum, maximum, std]
        
            desc_stat = pd.DataFrame(index = idx)
            desc_stat = desc_stat.join(measures)

            for col in outlier_cols:
                data = self.full_data[col]
                data = data.dropna()
                plt.figure(figsize = (20,10))
                plt.subplot(121)
                plt.hist(data, bins = 50)
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.title(f'{col}')
                plt.grid(True)
                plt.subplot(122)
                plt.boxplot(data, whis = whisker_factor)
                plt.xlabel('Value')
                plt.title(f'{col}')
            
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                whisker_low = q1 - (whisker_factor * iqr)
                whisker_up = q3 + (whisker_factor * iqr)
            
                desc_stat.loc[col, 'LowWhisk'] = whisker_low
                desc_stat.loc[col, 'UpWhisk'] = whisker_up
            
            desc_stat.index = desc_stat.index.rename('Variable')
            desc_stat.columns = measure_cols
            plot_desc_stat = tabulate(desc_stat,
                                      headers = 'keys',
                                      floatfmt = '.2f',
                                      tablefmt = 'simple',
                                      numalign = 'center',
                                      showindex = True)
            
            print(100*'=')
            print('Descriptive statistic for outlier identification')
            print(100*'=', '\n')
            print(plot_desc_stat)
            print(95*'-')
            print(f'Number of observations: {nobs}')
            print(100*'=', '\n\n\n\n')
# =========================
# The entries in the array adj_whiskers display the the lower and 
# upper tresholds to identify outliers. The order must refer to 
# the same as in the list adj_outlier_cols

            adj_outlier_cols = ['TotalAssets',
                                'PriceRevision',
                                'SharesRevision',
                                'ProceedsRevision',
                                'RegistrationDays']
        
            adj_whiskers = np.array([
                [-5942.31, 50000],
                [-230.22, 229.67],
                [-230.22, 229.67],
                [-313.33, 317.00],
                [0, 1338] 
                ])
        
            adj_whisk_cols = ['AdjustedLowWhisker', 'AdjustedUpWhisker']
            adj_whiskers = pd.DataFrame(adj_whiskers)
            adj_whiskers.columns = adj_whisk_cols
            adj_whiskers.index = adj_outlier_cols
        
            n_outliers = 0
            for index, row in adj_whiskers.iterrows():
                data = self.full_data[index]
                outlier_filter = (data <= row[adj_whisk_cols[0]])|\
                                 (data >= row[adj_whisk_cols[1]])
                outliers = data.loc[outlier_filter]
                outliers = outliers.index
                n_outliers += len(outliers)
                self.full_data = self.full_data.drop(outliers)
            
            stat_data_adj = self.full_data[adj_outlier_cols]
            mean_adj = stat_data_adj.mean()
            median_adj = stat_data_adj.median()
            minimum_adj = stat_data_adj.min()
            maximum_adj = stat_data_adj.max()
            std_adj = stat_data_adj.std()
            nobs_adj = len(stat_data_adj)
            measures_adj = [mean_adj, median_adj,
                            minimum_adj, maximum_adj,
                            std_adj, adj_whiskers]
        
            idx = outlier_cols
            adj_cols = measure_cols[:-2]
            adj_cols.append(adj_whisk_cols[0])
            adj_cols.append(adj_whisk_cols[1])
            desc_stat_adj = pd.DataFrame(index = idx)
            desc_stat_adj = desc_stat_adj.join(measures_adj)
            desc_stat_adj.columns = adj_cols
        
            for col in adj_outlier_cols:
                data = self.full_data[col]
                plt.figure(figsize = (20,10))
                plt.hist(data, bins = 50)
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.title(f'{col} adjusted for outliers')
                plt.grid(True)
        
            desc_stat_adj.index = desc_stat_adj.index.rename('Variable')
            plot_desc_stat_adj = tabulate(desc_stat_adj,
                                          headers = 'keys',
                                          floatfmt = '.2f',
                                          tablefmt = 'simple',
                                          numalign = 'center',
                                          showindex = True)
            plot_desc_stat_adj = plot_desc_stat_adj.replace('nan', '---')
            
            print(100*'=')
            print('Descriptive statistic after outlier adjustments')
            print(100*'=', '\n')
            print(plot_desc_stat_adj)
            print(95*'-')
            print(f'Number of observations after adjustments: {nobs_adj}')
            print(f'Number of removed outliers: {n_outliers}')
            print(100*'=', '\n\n')
       
            self.model_data = self.full_data[model_cols]
        else:
            self.model_data = self.full_data[model_cols]
            
            
            
            
            
        
        
        

        
        
        
        
        
        
        
        
        
        
        
                
                

        

        
        
        
        
        
