# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 22:23:49 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset
from bs4 import BeautifulSoup
import re 
from re import search
from tabulate import tabulate 
import warnings
import os
from GetData import *
from Visualization import *
from Generic_Parser import *
from MOD_Load_MasterDictionary_v2020 import *
import MOD_Load_MasterDictionary_v2020 as LM
warnings.filterwarnings("ignore")

model_cols = [
    'InitialReturn', 'InitialReturnAdjusted',
    'UnderwriterRank', 'TotalAssets', 'TechDummy', 
    'VentureDummy', 'AMEX', 'NASDQ', 'NYSE',
    'MarketReturn', 'MarketReturnSlopeDummy',
    'SectorReturn', 'SectorReturnSlopeDummy',
    'PriceRevision', 'PriceRevisionSlopeDummy',
    'PriceRevisionMaxDummy', 'PriceRevisionMinDummy',
    'PriceRevisionMaxSlopeDummy', 'PriceRevisionMinSlopeDummy',
    'WordsRevisionDummy', 'PositiveWordsRevision', 'NegativeWordsRevision',
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
        
    def preprocessing(self, adj_raw_prospectuses):
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
# =========================
        if adj_raw_prospectuses == True:
            prospectus_forms = ['InitialProspectus', 'FinalProspectus']
            unclassified_prospectuses = 0
            classified_prospectuses = 0
            for form in prospectus_forms:
                iter_count = 0
                progress = 100
                print('Preprocessing of raw prospectuses started', '\n')
                print(f'Prospectus type: {form}')
                for index, row in self.full_data.iterrows():
                    iter_count += 1
                    if iter_count % progress == 0:
                        print(f'Progress: {iter_count} items')
                    if pd.isnull(row[f'{form}FileName']) == False:
                        file = row[f'{form}FileName']
                        path = raw_prospectus_path+'\\'+form+'\\'+file
                        try:
                            raw_prosp = open(path+'\\'+file, 'r')
                            raw_prosp = raw_prosp.read()
                            
                            prosp = raw_prosp.rstrip()
                            prosp = re.sub(r'\n', '', prosp)
                            prosp = re.sub(r'\t', '', prosp)
                            prosp = re.sub(r'\v', '', prosp)

                            chars = ['-----END PRIVACY-ENHANCED MESSAGE-----', 
                                      '<DIV>', '<TR>', '<TD>', '<FONT>', '_'
                                      ] 
                            for char in chars:
                                prosp = re.sub(char, '', prosp)

                            prosp = re.sub(r'^.*?</SEC-HEADER>', '', prosp) 
                            prosp = re.sub(r'<TABLE>.+?</TABLE>', '', prosp) 
                            prosp = re.sub(r'<XBRL>.+?</XBRL>', '', prosp) 
                            prosp = re.sub(r'\s*<TYPE>(?:GRAPHIC|ZIP|EXCEL|PDF).+?end', '', prosp) 
                            prosp = re.sub(r'(-|\.|=)\s*', '', prosp) 
                            prosp = re.sub(r'-\s*$\s+', '', prosp) 
                            prosp = re.sub(r'-(?!\w)|(?<!\w)-', '', prosp) 
                            prosp = re.sub(r' {3,}', ' ', prosp) 
                            prosp = re.sub(r'and/or', 'and or', prosp) 
                            prosp = prosp.encode("ascii", "ignore")
                            prosp = prosp.decode()
                            html_soup = BeautifulSoup(prosp, 'lxml')
                            prosp = html_soup.get_text()
                            prosp = re.sub(r'<.+?>', '', prosp) 

                            file_name = row['DealNumber']
                            file_name = f'{file_name}.txt'
                            path_name = input_path+'\\'+form
                            output_file = open(path_name+'\\'+file_name,
                                               'w', encoding='utf-8')
                            output_file.write(prosp)
                            output_file.close()
                            classified_prospectuses += 1
                        except:
                            unclassified_prospectuses += 1
                            
            self.classified_prospectuses = classified_prospectuses
            self.unclassified_prospectuses = unclassified_prospectuses
            print('Preprocessing of raw prospectuses finished')
                            
    def firm_features(self):
        open_prc = self.full_data['OpenPrice']
        close_prc = self.full_data['ClosePrice']
        offer_prc = self.full_data['OfferPrice']
        init_ret = (close_prc / offer_prc) -1
        init_ret_adj = (close_prc / open_prc) -1
        self.full_data['InitialReturn'] = init_ret * self.scale
        self.full_data['InitialReturnAdjusted'] = init_ret_adj * self.scale
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
        exchange = self.full_data['ExchangeWhereIssuWillBeLi']
        exchange_dummies = pd.get_dummies(exchange)
        exchange_dummies = exchange_dummies.astype(float)
        features = ['AMEX', 'NASDQ', 'NYSE']
        
        self.full_data = pd.concat([self.full_data, 
                                    exchange_dummies[features]], 
                                   axis = 1)
        
        nan_ident = pd.isnull(self.full_data['ExchangeWhereIssuWillBeLi']) == True
        exchange_nan_idx = self.full_data.loc[nan_ident].index
        self.full_data.loc[exchange_nan_idx, features] = np.nan
# =========================
        vent = self.full_data['VentureBacked']
        vent_dummy = pd.get_dummies(vent)
        vent_dummy = vent_dummy['Yes']
        vent_dummy = vent_dummy.astype(float)
        feature = 'VentureDummy'
        vent_dummy.name = feature
        
        self.full_data = pd.concat([self.full_data, vent_dummy], axis = 1)                                
        nan_ident = pd.isnull(self.full_data['VentureBacked']) == True
        vent_nan_idx = self.full_data.loc[nan_ident].index
        self.full_data.loc[vent_nan_idx, feature] = np.nan
        
        
    def public_features(self, index_weight, port_days, adj_public_proxies):
        if adj_public_proxies == True:
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
            count_port_comps = []
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

                feature = ['MarketReturn']
                public_features.loc[index, feature] = index_ret
# =========================
                last_year = row['IssueDate'] - DateOffset(years=1)
                last_month = row['IssueDate'] - DateOffset(months=1)
                sector = row['FamaFrenchIndustry']
                feature = ['SectorReturn']
                valid_indus = self.prep_obj.valid_industries
                
                if (pd.isnull(sector) == False)&\
                (valid_indus[valid_indus.isin([sector])].empty == False):
                    mask = (port_data['IssueDate'] >= last_year) &\
                            (port_data['IssueDate'] <= last_month) &\
                            (port_data['FamaFrenchIndustry'] == sector)
                   
                    port_comp = port_data.loc[mask]
                    port_comp = port_comp['NCUSIP']
                    port_comp = port_comp.drop_duplicates()
                    count_port_comps.append(len(port_comp))
                    
                    if port_comp.empty == False:
                        match_dt = port_period.join(ipo_rets, how = 'inner')
                        comp_ident = match_dt['NCUSIP'].isin(port_comp)
                        comp_rets = match_dt.loc[comp_ident]
                        
                        noa = len(port_comp)
                        sector_ret = comp_rets['RETX'].sum()
                        sector_ret = 1/noa * sector_ret
                        sector_ret = sector_ret * self.scale
                        public_features.loc[index, feature] = sector_ret
                    else:
                        public_features.loc[index, feature] = np.nan
                else:
                    public_features.loc[index, feature] = np.nan
                
            count_port_comps = pd.Series(count_port_comps)
            count_port_comps.name = 'CountPortfolioComps'
            count_port_comps.to_csv(output_path+'\\'+count_portfolio_cons, 
                                    index = False)
                    
            market_ret = public_features['MarketReturn']
            market_ret_slope = get_SlopeDummy(market_ret)
            feature = 'MarketReturnSlopeDummy'
            public_features[feature] = market_ret_slope
            
            sector_ret = public_features['SectorReturn']
            sector_ret_slope = get_SlopeDummy(sector_ret)
            feature = 'SectorReturnSlopeDummy'
            public_features[feature] = sector_ret_slope
            
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
            count_port_comps = pd.read_csv(output_path+'\\'+count_portfolio_cons)
        
        self.full_data = pd.merge(self.full_data,
                                  public_features,
                                  how = 'left',
                                  on = 'DealNumber')
        
        self.count_portfolio_constituents = count_port_comps
            
    def private_features(self, adj_prospectus_analysis):
        if adj_prospectus_analysis == True:    
            col_names = ['DealNumber', 'FileSize', 'WordsCount', 
                          'Negative', 'Positive', 'Uncertain', 
                          'Litigious', 'StrongModal', 'WeakModal',
                          'Constraining', 'AlphabeticCount', 'DigitsCount',
                          'NumbersCount', 'AvgSyllablesPerWord', 
                          'AvgWordLength', 'Vocabulary']
            
            prospectus_forms = ['InitialProspectus', 'FinalProspectus']
            for form in prospectus_forms:          
                col_names_adj = []
                for col in col_names:
                    if col == 'DealNumber':
                        col_names_adj.append(col)
                    else:
                        col_adj = f'{form}{col}'
                        col_names_adj.append(col_adj)
                        
                iter_count = 0
                progress = 100
                print(f'Text analysis of {form} started')                 
                output_file = f'{form}_{prosp_result_file}'
                f_out = open(output_path+'\\'+output_file, 'w')
                wr = csv.writer(f_out, lineterminator='\n')
                wr.writerow(col_names_adj)

                file_path = input_path+'\\'+form
                file_list = glob.glob(file_path+'\\*.*')
                for file in file_list:
                    iter_count += 1
                    if iter_count % progress == 0:
                        print(f'Progress: {iter_count} items')
                        
                    dealnumber = re.findall('[0-9]+', file)
                    dealnumber = int(dealnumber[0])
                    with open(file, 'r', encoding='UTF-8', 
                              errors='ignore') as f_in:
                        doc = f_in.read()
                    doc = re.sub('(May|MAY)', ' ', doc)
                    doc = doc.upper()

                    output_data = get_data(doc)
                    output_data[0] = dealnumber
                    output_data[1] = len(doc)
                    wr.writerow(output_data)
                f_out.close()
                print(f'Text analysis of {form} finished', 
                      '\n')
                
        init_prosp_file = f'InitialProspectus_{prosp_result_file}'
        init_prosp = pd.read_csv(output_path+'\\'+init_prosp_file)
        
        final_prosp_file = f'FinalProspectus_{prosp_result_file}'
        final_prosp = pd.read_csv(output_path+'\\'+final_prosp_file)
        
        self.full_data = pd.merge(self.full_data,
                                  init_prosp,
                                  how = 'left',
                                  on = 'DealNumber')
        
        self.full_data = pd.merge(self.full_data,
                                  final_prosp,
                                  how = 'left',
                                  on = 'DealNumber')
        
        init_pos = self.full_data['InitialProspectusPositive']
        init_neg = self.full_data['InitialProspectusNegative']
        final_pos = self.full_data['FinalProspectusPositive']
        final_neg = self.full_data['FinalProspectusNegative']
        
        change_pos = final_pos - init_pos
        change_neg = final_neg - init_neg
        self.full_data['PositiveWordsRevision'] = change_pos
        self.full_data['NegativeWordsRevision'] = change_neg
        
        feature = 'WordsRevisionDummy'
        mean_pos_change = change_pos.mean()
        std_change_pos = change_pos.std()
        treshold = mean_pos_change
        dummy_cond = (change_pos > treshold)&(change_neg <= 0)
        pos_words_dummy = np.where(dummy_cond, 1, 0)
        nan_ident = (pd.isnull(change_pos) == True)|\
                    (pd.isnull(change_neg) == True)
        nan_idx = self.full_data.loc[nan_ident].index
        self.full_data[feature] = pos_words_dummy
        self.full_data.loc[nan_idx, feature] = np.nan
# =========================
        offer_prc = self.full_data['OfferPrice']
        mid_prc_rg = self.full_data['MeanPriceRange']
        prc_rev = (offer_prc / mid_prc_rg) -1
        prc_rev = prc_rev * self.scale
        max_prc_rg = self.full_data['MaxPriceRange']
        min_prc_rg = self.full_data['MinPriceRange']
        self.full_data['PriceRevision'] = prc_rev
        
        shares_off = self.full_data['SharesOffered']
        shares_fil = self.full_data['SharesFiled']
        shares_rev = (shares_off / shares_fil) -1
        shares_rev = shares_rev * self.scale
        self.full_data['SharesRevision'] = shares_rev
        
        act_pro = self.full_data['ActualProceeds']
        exp_pro = self.full_data['ExpectedProceeds']
        pro_rev = (act_pro / exp_pro) -1
        pro_rev = pro_rev * self.scale
        exp_pro_max = self.full_data['ExpectedProceedsMax']
        exp_pro_min = self.full_data['ExpectedProceedsMin']
        self.full_data['ProceedsRevision'] = pro_rev

        slope_dummy_cols = [
            'PriceRevision',
            'SharesRevision',
            'ProceedsRevision'
            ]
        slope_dummy_vars = [
            prc_rev,
            shares_rev,
            pro_rev
            ]
        for col, var in zip(slope_dummy_cols, slope_dummy_vars):
            slope_dummy = get_SlopeDummy(var)
            feature = f'{col}SlopeDummy'
            self.full_data[feature] = slope_dummy

        bound_dummy_cols = [
            'PriceRevision',
            'ProceedsRevision'
            ]
        bound_dummy_id = ['Max', 'Min']
        for col in bound_dummy_cols:
            for dummy_id in bound_dummy_id:
                if col == 'PriceRevision':
                    if dummy_id == 'Max':
                        bnd_dummy = get_DummyMax(offer_prc, max_prc_rg)
                        bnd_slp_dummy = get_SlopeDummyBounds(bnd_dummy, prc_rev)
                        feature = f'{col}MaxDummy'
                        col_name_slope = f'{col}MaxSlopeDummy'
                    else:
                        bnd_dummy = get_DummyMin(offer_prc, min_prc_rg)
                        bnd_slp_dummy = get_SlopeDummyBounds(bnd_dummy, prc_rev)
                        feature = f'{col}MinDummy'
                        col_name_slope = f'{col}MinSlopeDummy'
                else:
                    if dummy_id == 'Max':
                        bnd_dummy = get_DummyMax(act_pro, exp_pro_max)
                        bnd_slp_dummy = get_SlopeDummyBounds(bnd_dummy, pro_rev)
                        feature = f'{col}MaxDummy'
                        col_name_slope = f'{col}MaxSlopeDummy'
                    else:
                        bnd_dummy = get_DummyMin(act_pro, exp_pro_min)
                        bnd_slp_dummy = get_SlopeDummyBounds(bnd_dummy, pro_rev)
                        feature = f'{col}MinDummy'
                        col_name_slope = f'{col}MinSlopeDummy'
                    
                self.full_data[feature] = bnd_dummy
                self.full_data[col_name_slope] = bnd_slp_dummy
                
        self.model_data = self.full_data[model_cols]
           
    def outlier_adjustment(self, whisker_factor, plot_outliers):
        drop_cols = [
            'Dummy', 'SlopeDummy', 'Min', 'Max',
            'UnderwriterRank', 'AMEX', 
            'NASDQ', 'NYSE', 'IssueDate',
            ]
        outlier_cols = []
        for col in model_cols:
            if not any(char in col for char in drop_cols):
                outlier_cols.append(col)
        
        idx = outlier_cols
        measure_cols = [
            'Mean', 'Median',
            'Minimum', 'Maximum',
            'StandardDeviation',
            'LowerWhisker', 'UpperWhisker'
            ]
        
        stat_data = self.model_data[outlier_cols]
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
            data = self.model_data[col]
            data = data.dropna()
            if plot_outliers == True:
                plt.figure(figsize = (20,10))
                plt.subplot(121)
                plt.hist(data, bins = 50)
                plt.xlabel('Value', fontdict = xlabel_size)
                plt.ylabel('Frequency', fontdict = ylabel_size)
                plt.title(f'{col}', fontdict = title_size)
                plt.grid(True)
                plt.subplot(122)
                plt.boxplot(data, whis = whisker_factor)
                plt.xlabel('Value', fontdict = xlabel_size)
                plt.title(f'{col}', fontdict = title_size)
            
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            whisker_low = q1 - (whisker_factor * iqr)
            whisker_up = q3 + (whisker_factor * iqr)
            
            desc_stat.loc[col, 'LowerWhisker'] = whisker_low
            desc_stat.loc[col, 'UpperWhisker'] = whisker_up
            
        desc_stat.index = desc_stat.index.rename('Variable')
        desc_stat.columns = measure_cols
        plot_desc_stat = tabulate(desc_stat,
                                  headers = 'keys',
                                  floatfmt = '.2f',
                                  tablefmt = 'simple',
                                  numalign = 'center',
                                  showindex = True)
            
        print(cutting_line)
        print('Descriptive statistic for outlier identification')
        print(cutting_line, '\n')
        print(plot_desc_stat)
        print(cutting_line_thin)
        print(f'Number of observations: {nobs}')
        print(cutting_line, '\n\n\n\n')
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
            [0, 1321.00] 
            ])
        
        adj_whisk_cols = ['AdjustedLowerWhisker', 
                          'AdjustedUpperWhisker']
        adj_whiskers = pd.DataFrame(adj_whiskers)
        adj_whiskers.columns = adj_whisk_cols
        adj_whiskers.index = adj_outlier_cols
        
        n_outliers = 0
        for index, row in adj_whiskers.iterrows():
            data = self.model_data[index]
            outlier_filter = (data <= row[adj_whisk_cols[0]])|\
                             (data >= row[adj_whisk_cols[1]])
            outliers = data.loc[outlier_filter]
            outliers = outliers.index
            n_outliers += len(outliers)
            self.model_data = self.model_data.drop(outliers)
            
        stat_data_adj = self.model_data[adj_outlier_cols]
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
            if plot_outliers == True:
                data = self.model_data[col]
                plt.figure(figsize = (20,10))
                plt.hist(data, bins = 50)
                plt.xlabel('Value', fontdict = xlabel_size)
                plt.ylabel('Frequency', fontdict = ylabel_size)
                plt.title(f'{col} adjusted for outliers', 
                          fontdict = title_size)
                plt.grid(True)
        
        desc_stat_adj.index = desc_stat_adj.index.rename('Variable')
        plot_desc_stat_adj = tabulate(desc_stat_adj,
                                      headers = 'keys',
                                      floatfmt = '.2f',
                                      tablefmt = 'simple',
                                      numalign = 'center',
                                      showindex = True)
        plot_desc_stat_adj = plot_desc_stat_adj.replace('nan', '---')
            
        print(cutting_line)
        print('Descriptive statistic after outlier adjustments')
        print(cutting_line, '\n')
        print(plot_desc_stat_adj)
        print(cutting_line_thin)
        print(f'Number of observations after adjustments: {nobs_adj}')
        print(f'Number of removed outliers: {n_outliers}')
        print(cutting_line, '\n\n')
       
            
            
            
            
            
        
        
        

        
        
        
        
        
        
        
        
        
        
        
                
                

        

        
        
        
        
        
