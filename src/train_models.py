from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import pickle
import os
import sklearn
from sklearn.metrics import mean_squared_error
import time
import itertools
from typing import Any, Literal
from sklearn.impute import SimpleImputer
import math
import scipy.special
import random
from typing import Literal, NamedTuple
import rtdl_num_embeddings  # https://github.com/yandex-research/rtdl-num-embeddings
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.optim
from tqdm.std import tqdm
import lightgbm as lgb
import catboost as cbt
from catboost import CatBoostRegressor, Pool
import sys
import warnings
warnings.resetwarnings()
warnings.simplefilter('ignore')
#########################################################
base_dir = os.getcwd()
sys.path.append(os.path.join(base_dir,'src'))
from tabM_utils import *
train_model()
from tabpfn import TabPFNRegressor



class RegressionLabelStats(NamedTuple):
    mean: float
    std: float


data_found = False
print('loading raw and preprocessed data')

    
try:
    data_cat  = pd.read_csv(os.path.join(base_dir,'data','processed','data_cat.csv'),index_col=0)
    data_gbm  = pd.read_csv(os.path.join(base_dir,'data','processed','data_gbm.csv'),index_col=0)
    y_train = pd.read_csv(os.path.join(base_dir, 'data','raw','train_labels.csv'))['composite_score']
    # this of course not used during the training
    test = pd.read_csv(os.path.join(base_dir, 'data','raw','test_labels.csv'))
    y_test = test['composite_score']
    ids = test['uid'].values
    
    a = open(os.path.join(base_dir,'data','processed','data_processed_for_tabM.pkl'), 'rb')
    data_numpy = pickle.load(a)
    a.close()
    data_found =True
except:
   print('data not found in the /data/processed folder. Please run the preprocess_data.py script first ')  


if data_found:
 
    
    print('all raw and pre-processed data found, training is starting now')    
    object_cols = data_cat.select_dtypes(include=['object']).columns
    for col in object_cols:
        data_cat[col] = data_cat[col].astype('category')
        data_gbm[col] = data_gbm[col].astype('category')
    
    idx_train = range(len(y_train))
    idx_test = range(len(y_train), data_gbm.shape[0])
    #tab PFN is fitted only if cuda is available, otherwise this is not computationally feasible
    if torch.cuda.is_available():
          print('CUDA found, fitting tabPFN')   
          clf = TabPFNRegressor(random_state=0, n_estimators =8)
          clf.fit(data_cat.iloc[idx_train], y_train)
          prediction = clf.predict(data_cat.iloc[idx_test]).reshape(len(y_test),)   
          print('saving tabPFN predictions and model')
          pd.Series(index = ids, data =prediction).to_csv(os.path.join(base_dir,'predictions','predictions_tabPFN.csv'), header =False)
          try: 
               #~120MB , save this but use a try-except as it may fail
               a = open( os.path.join(base_dir,'models','tabPFN_fitted.pkl'),'wb')
               pickle.dump(clf,a)
               a.close()
          except:
               pass  

        
    else:
        print('CUDA not found, skipping tabPFN fit')   
        
    print(' ')    
    float_cols = data_cat.select_dtypes(include=['float']).columns
    cat_cols = [
        i for i in data_cat.columns if (
            i not in float_cols) & (
                i != 'uid') & (
                    i != 'id')]
    ######################################################################### 
    params = {"objective": "mse",
              "num_leaves": 32,
              "learning_rate": .05,
              'max_depth': 5,
              "verbosity": -1}
    
    n1 = 250    
    print('fitting lgbm')
    lgb_train = lgb.Dataset(data_gbm.iloc[idx_train], y_train)
    gbm = lgb.train(params, lgb_train, num_boost_round=n1)
    pred = gbm.predict(data_gbm.iloc[idx_test])
    ###########################################################################
    print('fitting catboost')
    n2 = 800
    model = CatBoostRegressor(
        learning_rate=.05,
        max_depth=5,
        iterations=n2,
        eval_metric='RMSE')
    
    model.fit(data_cat.iloc[idx_train], y_train, cat_features=list(
        data_cat.dtypes[data_cat.dtypes == 'category'].index), verbose=False)
    pred2 = model.predict(data_cat.iloc[idx_test])
    pd.Series(index = ids, data = pred2.reshape(len(pred2),)).to_csv(os.path.join(base_dir,'predictions','predictions_catboost.csv'),header =False)
    pd.Series(index = ids, data = pred.reshape(len(pred),)).to_csv (os.path.join(base_dir,'predictions','predictions_lightgbm.csv'), header =False)
    a = open( os.path.join(base_dir,'models','lightgbm_fitted.pkl'),'wb')
    pickle.dump(gbm,a)
    a.close()
    a  = open( os.path.join(base_dir,'models','catboost_fitted.pkl'),'wb')
    pickle.dump(model,a)
    a.close()
    print('catboost and lightgbm models trained and saved')
    print(' ')
    
    ############################################################################
    print('fitting tabM for 30 epochs for 5 different iterations corresponding to 5 seeds')
    task_type = 'regression'
    Y_train = data_numpy['train']['y'].copy()
    if task_type == 'regression':
        # For regression tasks, it is highly recommended to standardize the
        # training labels.
        regression_label_stats = RegressionLabelStats(
            Y_train.mean(), Y_train.std()
        )
        Y_train = (Y_train - regression_label_stats.mean) / \
            regression_label_stats.std
    
    for seed in range(5):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        preds = train_model(data_numpy, Y_train, regression_label_stats ,seed, base_dir, 'regression',ids)
        
        
    
    
