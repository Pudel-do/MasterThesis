import pandas as pd
from DataPreparation import DataPreparation
from FeatureEngineering import FeatureEngineering
from Regression import Regression
from PredictionModel import PredictionModel
from GetData import *
from Visualization import *
import warnings
warnings.filterwarnings("ignore")
# =====================================
# General Settings
# =====================================
start_date = pd.Timestamp('2000-01-01')
end_date = pd.Timestamp('2019-12-31')
start_year = start_date.strftime('%Y')
end_year = end_date.strftime('%Y')
# ==============================
# Data Preparation
# ==============================
use_data_preparation = False
adj_underwriter_matching = False
adj_time_range = False
adj_close_price = False
industry_treshold = 0.01
# ==============================
# Feature Engineering
# ==============================
use_feature_engineering = False
adj_public_proxies = False
adj_raw_prospectuses = False
adj_prospectus_analysis = False
plot_outliers = False   
portfolio_division = 'Industry'
portfolio_period = 15
scale_factor = 100
whisker_factor = 15
# ==============================
# Prediction Model
# ==============================
use_prediction_model = False
adj_preprocessing = False
use_feature_selection = False
adj_model_training = False
adj_benchmark_training = False
benchmark = 'SVC'
target_return = 10
# ==============================
# Visualization
# ==============================
plot_sectorportfolio = False
plot_prospectus_analysis = False
plot_yearly_regression = False
plot_underpricing_analysis = False
plot_prediction_performance = True
# ==============================
# Regression
# ==============================
'''Variable adj_regressors must be set to False for file cleaning'''
clean_file = False 
adj_regressors = False 
reg_vars = [
    'InitialReturn', 
    'UnderwriterRank', 
    'TotalAssets',
    'TechDummy',
    'VentureDummy',
    'ExpectedProceeds',
    'SectorVolume',
    'MarketReturn', 
    'MarketReturnSlopeDummy',
    'SectorReturn', 
    'SectorReturnSlopeDummy',
    'WordsRevisionDummy',
    'PriceRevision', 
    # 'PriceRevisionSlopeDummy',
    'PriceRevisionMaxDummy',
    # 'PriceRevisionMaxSlopeDummy', 
    # 'PriceRevisionMinDummy',
    # 'PriceRevisionMinSlopeDummy',
    'SharesRevision', 
    'SharesRevisionSlopeDummy',
    ]
# =========================
if use_data_preparation == True:
    prep_obj = DataPreparation(start_date, end_date)
    prep_obj.rough_preprocessing(adj_time_range)
    prep_obj.build_aux_vars(industry_treshold)
    prep_obj.extended_preprocessing(adj_underwriter_matching)
    prep_obj.data_merging(adj_close_price)
    obj_file = prep_obj_file
    obj_file = f'{start_year}_{end_year}_{obj_file}'
    save_obj(prep_obj, output_path, obj_file)
else:
    obj_file = prep_obj_file
    obj_file = f'{start_year}_{end_year}_{obj_file}'
    prep_obj = get_object(output_path, obj_file)
    
if use_feature_engineering == True:
    feat_eng_obj = FeatureEngineering(prep_obj, scale_factor)
    feat_eng_obj.preprocessing(adj_raw_prospectuses)
    feat_eng_obj.firm_features()
    feat_eng_obj.public_features(portfolio_division, portfolio_period, adj_public_proxies)
    feat_eng_obj.private_features(adj_prospectus_analysis)
    feat_eng_obj.outlier_adjustment(whisker_factor, plot_outliers)
    obj_file = feat_eng_obj_file
    obj_file = f'{start_year}_{end_year}_{obj_file}'
    save_obj(feat_eng_obj, output_path, obj_file)
else:
    obj_file = feat_eng_obj_file
    obj_file = f'{start_year}_{end_year}_{obj_file}'
    feat_eng_obj = get_object(output_path, obj_file)
    
reg_obj = Regression(feat_eng_obj, start_year, end_year)
reg_obj.preprocessing(reg_vars)
reg_obj.ols_regression()
reg_obj.fmb_regression(start_date, end_date)
reg_obj.build_results(adj_regressors, clean_file)
obj_file = reg_obj_file
obj_file = f'{start_year}_{end_year}_{obj_file}'
save_obj(reg_obj, output_path, obj_file)

if use_prediction_model == True:
    pred_obj = PredictionModel(reg_obj, start_year, end_year)
    pred_obj.preprocessing(target_return, adj_preprocessing, use_feature_selection)
    pred_obj.model_training(adj_model_training, adj_benchmark_training, benchmark)
    pred_obj.model_evaluation()
    obj_file = pred_obj_file
    obj_file = f'{start_year}_{end_year}_{obj_file}'
    save_obj(pred_obj, output_path, obj_file)
else:
    obj_file = pred_obj_file
    obj_file = f'{start_year}_{end_year}_{obj_file}'
    pred_obj = get_object(output_path, obj_file)
# ==============================
Analysis_SectorPortfolio(feat_eng_obj, plot_sectorportfolio)
Statistic_ProspectusAnalysis(feat_eng_obj, plot_prospectus_analysis)
Statistic_RegressionSample(reg_obj)
RegressionResults(reg_obj, plot_yearly_regression)
UnderpricingAnalysis(feat_eng_obj, pred_obj, plot_underpricing_analysis, target_return)
Performance_PredictionModel(pred_obj, plot_prediction_performance)



