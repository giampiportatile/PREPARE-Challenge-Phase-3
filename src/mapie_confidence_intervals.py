import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import shap
import os
from mapie.regression import (
    CrossConformalRegressor,
    JackknifeAfterBootstrapRegressor)


base_dir = os.getcwd()
confidence_level = 0.9
#folds used for the cross-validation
cv = 5
data_found =False
#################################################
try:
    data_cat  = pd.read_csv(os.path.join(base_dir,'data','processed','data_cat.csv'),index_col=0)
    #data_gbm  = pd.read_csv(os.path.join(base_dir,'data','processed','data_gbm.csv'),index_col=0)
    y_train = pd.read_csv(os.path.join(base_dir, 'data','raw','train_labels.csv'))['composite_score']
    # this of course not used during the training
    y_test = pd.read_csv(os.path.join(base_dir, 'data','raw','test_labels.csv'))['composite_score']
    data_found =True
except:
   print('data not found in the /data/processed folder. Please run the preprocess_data.py script first ')  
   

if data_found:
 
    print('All raw and pre-processed data found, training is starting now')    
    object_cols = data_cat.select_dtypes(include=['object']).columns
    for col in object_cols:
        data_cat[col] = data_cat[col].astype('category')
        #data_gbm[col] = data_gbm[col].astype('category')
    
    idx_train = range(int(len(y_train)))
    idx_test = range(len(y_train), data_cat.shape[0])

    float_cols = data_cat.select_dtypes(include=['float']).columns
    cat_cols = [
        i for i in data_cat.columns if (
            i not in float_cols) & (
                i != 'uid') & (
                    i != 'id')]
    ######################################################################### 

    
    STRATEGIES = {
    "cv_plus": {
        "class": CrossConformalRegressor,
        "init_params": dict(method="plus", cv = cv),
    },
    "jackknife": {
        "class": CrossConformalRegressor,
        "init_params": dict(method="base", cv = cv),
    },
    "jackknife_plus": {
        "class": CrossConformalRegressor,
        "init_params": dict(method="plus", cv = cv),
    },
    "jackknife_minmax": {
        "class": CrossConformalRegressor,
        "init_params": dict(method="minmax", cv = cv),
    },
   }

    res ={}
    for i, (strategy_name, strategy_params) in enumerate(STRATEGIES.items()):
        init_params = strategy_params["init_params"]
        class_ = strategy_params["class"]
        n2 = 800
        model = CatBoostRegressor(
             learning_rate = .05,
             max_depth = 5,
             iterations=n2, cat_features=list(
                  data_cat.dtypes[data_cat.dtypes == 'category'].index),
             eval_metric='RMSE', verbose = False)
        
        #initialise the MAPIE model with a catboost regressor
        mapie = CrossConformalRegressor(model
                , confidence_level = confidence_level, random_state=0, n_jobs = 1,**init_params
            )
        print('estimating confidence intervals with ' +strategy_name)
        #fit the catboost model cv times ( 5 in this example), as specified in the STRATEGIES dictionary
        mapie.fit_conformalize(data_cat.iloc[idx_train], y_train)   
        #average prediction and corresponding confidence interval for each subject in the test set
        y_pred, y_pis = mapie.predict_interval(data_cat.iloc[idx_test])
        res[strategy_name+ 'mean'] = y_pred
        res[strategy_name+ 'ci'] = y_pis
        
    for i, (strategy_name, strategy_params) in enumerate(STRATEGIES.items()):
          y_pred =   res[strategy_name+ 'mean'] 
          y_pis = res[strategy_name+ 'ci'] 
         
          idx = np.argsort(y_pred)
       
          plt.figure()
          plt.plot(y_pred[idx], 'or')
          plt.plot(y_pis[idx,1,0], 'b')
          plt.plot(y_pis[idx,0,0], 'b')
          plt.title(strategy_name)
          plt.legend(['mean prediction','upper and lower bounds'])
          plt.ylabel('cognitive score')
