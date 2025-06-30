
import numpy as np
import pandas as pd
import time
import os
import pickle
import catboost as cbt
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, EShapCalcType, EFeaturesSelectionAlgorithm, Pool

dir_to_save = r'O:\Gianpaolo\various\AD\phase3\data_MHAS'

fraction_train  = 0.8
#there are 4535 features, I eliminate 200 at each iteration
num_features_to_eliminate_at_each_step  = 200
# at the end, 22*200 =4400 features are elimiated
number_iterations = 22

X = pd.read_csv(r'O:\Gianpaolo\various\AD\phase3\data_MHAS\features.csv')
y = pd.read_csv(r'O:\Gianpaolo\various\AD\phase3\data_MHAS\target.csv') 


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
               cat_features=cat_f)


features ={}
final_summary ={}
#save the RMSE on the test set as a function of the iteration after recursively removing the features based on their Shapley values
errors = np.zeros((number_iterations,1))
print('start')


errors = pd.read_csv( os.path.join(dir_to_save,'errors_shapley.csv'),index_col=0).values
first =np.nonzero(errors==0)[0][0]
a =open (os.path.join(dir_to_save,'features_shapley.pkl'),'rb')
final_summary =  pickle.load(a)
a.close()
l =list(final_summary.keys())
my_features = final_summary[l[-1]]
print(train_X.shape)
train_X = train_X[my_features]
print(train_X.shape)
val_X = val_X[my_features]
cat_f =[ i for i in cat_f if i in my_features]
     
eval_dataset = Pool(val_X,
               val_y,
               cat_features=cat_f)

        
for my_iter in range(first, number_iterations):
        ctb_params = dict(iterations = 3000,
                          learning_rate = .05,
                          depth = 5,
                          loss_function='RMSE',
                          eval_metric = 'RMSE',
                          metric_period = 1,
                          od_type = 'Iter',
                          od_wait = 100,
                          cat_features= cat_f,
                          task_type = 'GPU',
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
        errors[my_iter] = mean_squared_error(val_y, pred, squared=False)
        
        final_summary['best_it'+str(my_iter)] = ctb_model.tree_count_

        my_features = summary['selected_features_names']
        final_summary['features'+str(my_iter)] =my_features
        
        
        pd.Series(errors[:,0]).to_csv( os.path.join(dir_to_save,'errors_shapley.csv'))
        a =open (os.path.join(dir_to_save,'features_shapley.pkl'),'wb')
        pickle.dump(final_summary,a)
        a.close()
        
        
        train_X = train_X[summary['selected_features_names']]
        val_X = val_X[summary['selected_features_names']]
        cat_f = list(val_X.dtypes[val_X.dtypes =='object'].index)
        
        eval_dataset = Pool(val_X,
                       val_y,cat_features= cat_f)
        
      
