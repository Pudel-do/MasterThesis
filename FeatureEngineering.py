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
    'ExpectedProceeds', 'ActualProceeds',
    'SectorVolume', 'Volume', 'MarketReturn', 
    'MarketReturnSlopeDummy', 'SectorReturn', 
    'SectorReturnSlopeDummy', 'PriceRevision', 
    'PriceRevisionSlopeDummy', 'PriceRevisionMaxDummy', 
    'PriceRevisionMinDummy', 'PriceRevisionMaxSlopeDummy', 
    'PriceRevisionMinSlopeDummy', 'WordsRevisionDummy', 
    'PositiveWordsRevision', 'NegativeWordsRevision',
    'SecondarySharesDummy', 'SecondarySharesRevisionDummy', 
    'SecondarySharesRevisionRatio', 'TotalSharesRevisionDummy',
    'PrimarySharesRevisionDummy', 'RegistrationDays', 'IssueDate',
    ]

class FeatureEngineering:
    def __init__(self, prep_obj, scale_factor):
        self.prep_obj = prep_obj
        self.full_data = prep_obj.full_data
        self.scale = scale_factor
        self.model_cols = model_cols
        
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
            
        self.prep_obj.port_data['TechIndustry'] = sector
        merge_cols = ['DealNumber', 'TechIndustry']
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
        sector = self.prep_obj.port_data['TechIndustry']
        tech_dummy = np.where(sector == 'Non-Hitech',
                              non_tech_id,
                              tech_id)
        nan_ident = pd.isnull(self.prep_obj.port_data['TechIndustry']) == True
        tech_nan_idx = self.prep_obj.port_data.loc[nan_ident].index
        self.prep_obj.port_data['TechDummy'] = tech_dummy
        self.prep_obj.port_data.loc[tech_nan_idx, 'TechDummy'] = np.nan
        
        merge_cols = ['DealNumber', 'TechDummy']
        self.full_data = pd.merge(self.full_data,
                                  self.prep_obj.port_data[merge_cols],
                                  how = 'left',
                                  on = 'DealNumber')
# =========================
        disc_fact = self.full_data['CPI_DiscountFactor']
        total_assets = self.full_data['TotalAssets']
        total_assets = total_assets * disc_fact
        total_assets = np.log(total_assets)
        self.full_data['TotalAssets'] = total_assets
# =========================
        mean_prc_rg = self.full_data['MeanPriceRange']
        min_prc_rg = self.full_data['MinPriceRange']
        max_prc_rg = self.full_data['MaxPriceRange']
        tot_shares_filed = self.full_data['TotalSharesFiled']
        tot_shares_offered = self.full_data['SharesOfferedSumOfAllMkts']

        exp_pro = tot_shares_filed * mean_prc_rg
        exp_pro = exp_pro * disc_fact
        exp_pro = np.log(exp_pro)
        exp_pro_min = tot_shares_filed * min_prc_rg
        exp_pro_min = exp_pro_min * disc_fact
        exp_pro_max = tot_shares_filed * max_prc_rg
        exp_pro_max = exp_pro_max * disc_fact

        act_pro = tot_shares_offered * offer_prc
        act_pro = act_pro * disc_fact
        act_pro = np.log(act_pro)
        
        self.full_data['TotalSharesOffered'] = tot_shares_offered
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
        
    def public_features(self, portfolio_division, portfolio_period, adj_public_proxies):
        port_division = portfolio_division
        self.port_division = port_division
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
            index_rets = index_rets_data['ewretx']

            iter_count = 0
            progress = 100
            print('Public features construction started')
            port_data = self.prep_obj.port_data
            public_features = pd.DataFrame()
            sector_port_results = pd.DataFrame()
            for index, row in self.full_data.iterrows():
                deal_num = row['DealNumber']
                col_name = ['DealNumber']
                public_features.loc[index, col_name] = deal_num
                iter_count += 1
                if iter_count % progress == 0:
                    print(f'Progress: {iter_count} items')
                if portfolio_period != None:
                    dt_offset = pd.offsets.BusinessDay(portfolio_period)
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
                prior_weeks = row['IssueDate'] - DateOffset(weeks=6)
                follow_weeks = row['IssueDate'] + DateOffset(weeks=2)
                division = row['TechDummy']
                
                sector_ipo_mask = (port_data['IssueDate'] >= prior_weeks) &\
                                  (port_data['IssueDate'] <= follow_weeks) &\
                                  (port_data['TechDummy'] == division)     
                sector_ipo_volume = port_data.loc[sector_ipo_mask]
                sector_ipo_volume = sector_ipo_volume['NCUSIP']
                sector_ipo_volume = sector_ipo_volume.drop_duplicates()
                sector_ipo_volume = len(sector_ipo_volume)
                feature = ['SectorVolume']
                public_features.loc[index, feature] = sector_ipo_volume
                
                ipo_mask = (port_data['IssueDate'] >= prior_weeks) &\
                           (port_data['IssueDate'] <= follow_weeks)
                ipo_volume = port_data.loc[ipo_mask]
                ipo_volume = ipo_volume['NCUSIP']
                ipo_volume = ipo_volume.drop_duplicates()
                ipo_volume = len(ipo_volume)
                feature = ['Volume']
                public_features.loc[index, feature] = ipo_volume
# =========================
                last_year = row['IssueDate'] - DateOffset(years=1)
                last_month = row['IssueDate'] - DateOffset(months=1)
                feature = ['SectorReturn']
                division = row[port_division]
                valid_ind = self.prep_obj.valid_industries
                
                if pd.isnull(division) == False:
                    if port_division == 'Industry':
                        if valid_ind[valid_ind.isin([division])].empty == False:
                            port_mask = (port_data['IssueDate'] >= last_year) &\
                                        (port_data['IssueDate'] <= last_month) &\
                                        (port_data[port_division] == division)
                                    
                            port_comp = port_data.loc[port_mask]
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
                                public_features.loc[index, feature] = sector_ret
                                
                                year = row['IssueDate'].strftime('%Y')
                                sector_port_results.loc[index, 'PortfolioComponents'] = noa
                                sector_port_results.loc[index, 'Division'] = division
                                sector_port_results.loc[index, 'Year'] = year
                        else:
                            public_features.loc[index, feature] = np.nan
                    else:
                        port_mask = (port_data['IssueDate'] >= last_year) &\
                                    (port_data['IssueDate'] <= last_month) &\
                                    (port_data[port_division] == division)
                                
                        port_comp = port_data.loc[port_mask]
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
                            public_features.loc[index, feature] = sector_ret
                        
                            year = row['IssueDate'].strftime('%Y')
                            sector_port_results.loc[index, 'PortfolioComponents'] = noa
                            sector_port_results.loc[index, 'Division'] = division
                            sector_port_results.loc[index, 'Year'] = year
                        else:
                            public_features.loc[index, feature] = np.nan
                else:
                    public_features.loc[index, feature] = np.nan
            
            sector_port_file = sector_port_result_file
            sector_port_file = f'{port_division}_{sector_port_file}'
            sector_port_results.to_csv(output_path+'\\'+sector_port_file, 
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
            pub_measure_file = public_feat_file
            pub_measure_file = f'{start_year}_{end_year}_{port_division}_{pub_measure_file}'
            public_features.to_csv(output_path+'\\'+pub_measure_file, 
                                   index = False)
            print('Public features construction finished')

        else:
            start_year = self.prep_obj.start_year
            end_year = self.prep_obj.end_year
            pub_measure_file = public_feat_file
            pub_measure_file = f'{start_year}_{end_year}_{port_division}_{pub_measure_file}'
            public_features = pd.read_csv(output_path+'\\'+pub_measure_file)
            sector_port_file = sector_port_result_file
            sector_port_file = f'{port_division}_{sector_port_file}'
            sector_port_results = pd.read_csv(output_path+'\\'+sector_port_file)
        
        self.full_data = pd.merge(self.full_data,
                                  public_features,
                                  how = 'left',
                                  on = 'DealNumber')
        
        self.sector_portfolio_results = sector_port_results
            
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
        max_prc_rg = self.full_data['MaxPriceRange']
        min_prc_rg = self.full_data['MinPriceRange']
        prc_rev = (offer_prc / mid_prc_rg) -1
        prc_rev = prc_rev * self.scale
        self.full_data['PriceRevision'] = prc_rev

        tot_shares_offered = self.full_data['TotalSharesOffered']
        prim_shares_offered = self.full_data['PrimaryShsOfrdSumOfAllMkts']
        sec_shares_offered = self.full_data['SecondaryShsOfrdSumOfAllMkts']
        
        tot_shares_filed = self.full_data['TotalSharesFiled']
        prim_shares_filed = self.full_data['PrimarySharesFiled']
        sec_shares_filed = self.full_data['SecondarySharesFiled']
        
        primary_shares_mask = (self.full_data['SecondarySharesFiled'] == 0 )|\
                              (self.full_data['SecondaryShsOfrdSumOfAllMkts'] == 0)
                             
        secondary_shares_mask = (self.full_data['SecondarySharesFiled'] != 0 )&\
                                (self.full_data['SecondaryShsOfrdSumOfAllMkts'] != 0)                      
                          
        primary_shares_idx = self.full_data[primary_shares_mask].index
        secondary_shares_idx = self.full_data[secondary_shares_mask].index
        self.full_data.loc[primary_shares_idx, 'SecondarySharesDummy'] = 0
        self.full_data.loc[secondary_shares_idx, 'SecondarySharesDummy'] = 1
            
        sec_shares_diff = sec_shares_offered - sec_shares_filed
        sec_shares_rev = (sec_shares_offered / sec_shares_filed) -1
        sec_shares_rev_dummy = np.where(sec_shares_rev > 0, 1 ,0)
        nan_ident = pd.isnull(sec_shares_rev) == True
        nan_idx = sec_shares_rev.loc[nan_ident].index
        self.full_data['SecondarySharesRevision'] = sec_shares_rev
        self.full_data['SecondarySharesRevisionDummy'] = sec_shares_rev_dummy
        self.full_data['SecondarySharesRevisionDummy'].loc[nan_idx] = np.nan

        sec_shares_rev_dummy = self.full_data['SecondarySharesRevisionDummy']
        sec_shares_rev_ratio = sec_shares_diff / sec_shares_filed
        sec_shares_rev_ratio = np.where(sec_shares_diff > 0, sec_shares_rev_ratio, 0)
        self.full_data['SecondarySharesRevisionRatio'] = sec_shares_rev_ratio
        nan_ident = pd.isnull(sec_shares_diff) == True
        nan_idx = sec_shares_diff.loc[nan_ident].index
        self.full_data['SecondarySharesRevisionRatio'].loc[nan_idx] = np.nan
        
        primary_shares_rev = (prim_shares_offered / prim_shares_filed) -1
        primary_shares_rev_dummy = np.where(primary_shares_rev > 0, 1 ,0)
        nan_ident = pd.isnull(primary_shares_rev) == True
        nan_idx = primary_shares_rev.loc[nan_ident].index
        self.full_data['PrimarySharesRevisionDummy'] = primary_shares_rev_dummy
        
        tot_shares_rev = (tot_shares_offered / tot_shares_filed) -1
        tot_shares_rev_dummy = np.where(tot_shares_rev > 0, 1 ,0)
        nan_ident = pd.isnull(tot_shares_rev) == True
        nan_idx = tot_shares_rev.loc[nan_ident].index
        self.full_data['TotalSharesRevisionDummy'] = tot_shares_rev_dummy
        
        
        act_pro = self.full_data['ActualProceeds']
        exp_pro = self.full_data['ExpectedProceeds']
        pro_rev = (act_pro / exp_pro) -1
        pro_rev = pro_rev * self.scale
        exp_pro_max = self.full_data['ExpectedProceedsMax']
        exp_pro_min = self.full_data['ExpectedProceedsMin']
        self.full_data['ProceedsRevision'] = pro_rev

        slope_dummy_cols = [
            'PriceRevision',
            'ProceedsRevision'
            ]
        slope_dummy_vars = [
            prc_rev,
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
        stat_data = self.model_data[outlier_cols]
        nobs = len(stat_data)
        desc_stat = stat_data.describe()
        desc_stat = desc_stat.transpose()

        for col in outlier_cols:
            data = self.model_data[col]
            data = data.dropna()
            if plot_outliers == True:
                plt.figure(figsize = figsize)
                plt.subplot(121)
                plt.hist(data, bins = hist_bins)
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
            
            desc_stat.loc[col, 'WhiskerLow'] = whisker_low
            desc_stat.loc[col, 'WhiskerUp'] = whisker_up
            
        desc_stat.index = desc_stat.index.rename('Variable')
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
        print(cutting_line, paragraph)
# =========================
# The entries in the array adj_whiskers display the the lower and 
# upper tresholds to identify outliers. The order must refer to 
# the same as in the list adj_outlier_cols

        adj_outlier_cols = [
            'PriceRevision',
            'RegistrationDays'
            ]
        
        adj_whiskers = np.array([
            [-230.22, 229.67],
            [30, 365.00] 
            ])
        
        adj_whisk_cols = ['AdjustedWhiskerLow', 
                          'AdjustedWhiskerUp']
        adj_whiskers = pd.DataFrame(adj_whiskers)
        adj_whiskers.columns = adj_whisk_cols
        adj_whiskers.index = adj_outlier_cols
        
        n_outliers = 0
        for index, row in adj_whiskers.iterrows():
            data = self.model_data[index]
            outlier_filter = (data < row[adj_whisk_cols[0]])|\
                             (data > row[adj_whisk_cols[1]])
            outliers = data.loc[outlier_filter]
            outliers = outliers.index
            n_outliers += len(outliers)
            self.model_data = self.model_data.drop(outliers)
            
        stat_data_adj = self.model_data[adj_outlier_cols]
        nobs_adj = len(stat_data_adj)
        
        idx = outlier_cols
        adj_cols = desc_stat.columns[:-2]
        adj_cols = list(adj_cols)
        adj_cols.append(adj_whisk_cols[0])
        adj_cols.append(adj_whisk_cols[1])
        desc_stat_adj = stat_data_adj.describe()
        desc_stat_adj = desc_stat_adj.transpose()
        desc_stat_adj = desc_stat_adj.join(adj_whiskers)
        
        for col in adj_outlier_cols:
            if plot_outliers == True:
                data = self.model_data[col]
                plt.figure(figsize = figsize)
                plt.hist(data, bins = hist_bins)
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
            
        print(cutting_line)
        print('Descriptive statistic after outlier adjustments')
        print(cutting_line, '\n')
        print(plot_desc_stat_adj)
        print(cutting_line_thin)
        print(f'Number of observations after adjustments: {nobs_adj}')
        print(f'Number of removed outliers: {n_outliers}')
        print(cutting_line, paragraph)
       
            
            
            
            
            
        
        
        

        
        
        
        
        
        
        
        
        
        
        
                
                

        

        
        
        
        
        
