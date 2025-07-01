
import numpy as np
import pandas as pd
import time
import os
import pickle
import catboost as cbt
import torch
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, EShapCalcType, EFeaturesSelectionAlgorithm, Pool

#remember to execute this script from the base directory where the github repo has been downloaded
base_dir = os.getcwd()

#############HYPERPARAMETERS
#fraction of data to use to train the model
fraction_train  = 0.8
#there are 4535 features, I eliminate 200 at each iteration
num_features_to_eliminate_at_each_step  = 200
# at the end, 22*200 =4400 features are elimiated
number_iterations = 22
if torch.cuda.is_available():
   device = 'GPU'
else:
    device ='CPU'

print('reading the features')
data_found =  False
try:
    X = pd.read_csv(os.path.join( base_dir, 'analysis_of_MHAS_data','features.csv'))
    y = pd.read_csv(os.path.join( base_dir, 'analysis_of_MHAS_data','target.csv')) 
    data_found = True
except:
    
    print('features and/or target data not found in the analysis_of_MHAS_data folder')
    print('please execute the load_and_preprocess_data script first')
    
    
if data_found:
    idx = range(int(fraction_train*X.shape[0]))
    idx_test = range(int(fraction_train*X.shape[0]), X.shape[0])
    
    
    cat_f = list(X.dtypes[X.dtypes =='object'].index)
    X[cat_f] = X[cat_f] .fillna('')
    
    ####
    train_X = X.iloc[idx]
    val_X = X.iloc[idx_test]
    train_y = y.iloc[idx]
    val_y = y.iloc[idx_test]
    my_features = X.columns
    del X
         
    eval_dataset = Pool(val_X,
                   val_y,
                   cat_features = cat_f)
    
    
    features ={}
    final_summary ={}
    #save the RMSE on the test set as a function of the iteration after recursively removing the features based on their Shapley values
    errors = np.zeros((number_iterations,1))
    print('start')
        
            
    for my_iter in range(number_iterations):
            print('iteration '+ str(my_iter))
            
            ctb_params = dict(iterations = 3000,
                              learning_rate = .05,
                              depth = 5,
                              loss_function='RMSE',
                              eval_metric = 'RMSE',
                              metric_period = 1,
                              od_type = 'Iter',
                              od_wait = 100,
                              cat_features= cat_f,
                              task_type = device,
                              allow_writing_files=False)
    
            ctb_model = CatBoostRegressor(**ctb_params)
           
            summary = ctb_model.select_features(
                train_X,    train_y,
                eval_set = eval_dataset ,
                features_for_select = my_features,
                num_features_to_select = len(my_features)-num_features_to_eliminate_at_each_step,
                steps = 1,
                verbose= True,
                algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
                shap_calc_type=EShapCalcType.Regular,
                train_final_model = True)
            
            pred = ctb_model.predict( val_X)

            #save error at each iteration     
            errors[my_iter] = mean_squared_error(val_y, pred, squared=False)
            pd.Series(errors[:,0]).to_csv( os.path.join(base_dir,'errors_shapley.csv'))
            
            #save top features at each iteration
            my_features = summary['selected_features_names']
            final_summary['features'+str(my_iter)] =my_features
            a =open (os.path.join(base_dir,'features_shapley.pkl'),'wb')
            pickle.dump(final_summary,a)
            a.close()
            
            #reduce feature set by only looking at features retained in the current iteration
            train_X = train_X[summary['selected_features_names']]
            val_X = val_X[summary['selected_features_names']]
            cat_f = list(val_X.dtypes[val_X.dtypes =='object'].index)
            
            eval_dataset = Pool(val_X,
                           val_y,cat_features= cat_f)
            
          
