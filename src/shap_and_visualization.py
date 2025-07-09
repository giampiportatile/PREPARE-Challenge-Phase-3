
import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import shap
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from mapie.regression import (
    CrossConformalRegressor,
    JackknifeAfterBootstrapRegressor)
import matplotlib
font = {'weight' : 'bold',
            'size'   : 12}

matplotlib.rc('font', **font)

    
#------------------------------------------------------------------------------
#some of the visualization code was taken from this repo
#https://github.com/nicknettleton/PREPARE-Challenge
def visualise_prediction(
    pred, ci_1,
    ci_2,
    true_score):

    sns.set_palette('dark')
    sns.set_style('white')
    
    color = ('#ed7208', '#8e4201') 
    figure, ax = plt.subplots(1, 1)
    # plt.suptitle('Your prediction\ncompared to population', y=1)

    # Population chart
    sns.kdeplot(
        data = true_score,
        x = 'composite_score',
        fill = True,
        ax = ax,
        alpha = 0.2,
        linewidth = 0,
        cut = True
    )

    # Individual chart
    y = ax.get_ylim()
    p = pred
    ax.fill_betweenx(
        y, [ci_1[0]],[ci_1[1]],
        alpha=0.1,
        color=color[0]
    )
    ax.fill_betweenx(
        y, [ci_2[0]], [ci_2[1]],
        alpha=0.1,
        color=color[0]
    )
    
    ax.plot([p,p],y, '-', color=color[0])
    ax.annotate('Model prediction\n',
                (p,0),
                textcoords="offset points",
                xytext=(15,200),
                ha='left',
                color=color[1],
               )
    ax.annotate(str(p),
                (p,0),
                textcoords="offset points",
                xytext=(15,190),
                ha='left',
                color=color[1],
                fontsize=24
               )

    # General plot formatting
    ax.set_xlabel('Cognitive capacity')
    ax.set_ylabel('')
    ax.set_yticklabels('')
    ax.legend([
        'Population',
        'Prediction, with 90% and 50% likelihood'
    ])

    ax.set_ylim(y)
    sns.despine(ax=ax, left=True)



base_dir = os.getcwd()
#base_dir = r'O:\Gianpaolo\various\AD\phase3\repo'
data_found = False
try:
    data_cat  = pd.read_csv(os.path.join(base_dir,'data','processed','data_cat.csv'),index_col=0)
    y_train_all = pd.read_csv(os.path.join(base_dir, 'data','raw','train_labels.csv'))
    y_train =  y_train_all['composite_score']
    models ={}
    a = open(os.path.join(base_dir,'models','catboost_fitted.pkl'),'rb')
    models['cat' ] =pickle.load(a)
    a.close()
    
    data_found =True
except:
   print('data not found in the /data/processed folder. Please run the preprocess_data.py script first ')  
   

if data_found:
    #all these settings can be changed
    confidence_level1 =.5
    confidence_level2 =.9
    #subject whose reusults will be displayed. change at pleasure
    subject = 39
    
    print('All raw and pre-processed data found, training is starting now')    
    object_cols = data_cat.select_dtypes(include=['object']).columns
    for col in object_cols:
        data_cat[col] = data_cat[col].astype('category')

    
    idx_train = range(int(len(y_train)))
    idx_test = range(len(y_train), data_cat.shape[0])

    float_cols = data_cat.select_dtypes(include=['float']).columns
    cat_cols = [
        i for i in data_cat.columns if (
            i not in float_cols) & (
                i != 'uid') & (
                    i != 'id')]
    ######################################################################### 

    sub_explanations = {}
    sub_expected_values = []
    for name, estimator in models.items():#ensemble.named_estimators_.items():
    
        explainer = shap.Explainer(estimator)
        explanation = explainer(data_cat.iloc[idx_test])
 
    ensemble_shap_values     = explanation
    ensemble_expected_value  = explainer.expected_value


    STRATEGIES = {
    "cv_plus": {
        "class": CrossConformalRegressor,
        "init_params": dict(method="plus", cv = 5),
    }
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
                
        mapie = CrossConformalRegressor(model
                , confidence_level = [    confidence_level1,    confidence_level2], random_state=0, n_jobs = 1, **init_params)
        print('estimating confidence intervals with ' +strategy_name)
        #compue conformal scores
        mapie.fit_conformalize(data_cat.iloc[idx_train], y_train)   
        #average prediction and corresponding confidence interval for each subject in the test set
        y_pred, conf_intervals = mapie.predict_interval(data_cat.iloc[idx_test])
        #recenter making the prediction of the original model the mean and keeping the same width
        # that is needed as MAPIE computes te new average pred as EW average of the predictions of the 5 models
        width1_right   =   conf_intervals[:,1,0] - y_pred
        width1_left    =   conf_intervals[:,0,0] - y_pred
        width2_right   =   conf_intervals[:,1,1] - y_pred
        width2_left    =   conf_intervals[:,0,1] - y_pred
        
        y_pred = ensemble_shap_values .sum(1) + ensemble_expected_value
        
        #################################################################################
        ##DISPLAY
        ###############################
        ensemble_explanation = shap.Explanation(values = ensemble_shap_values[subject],
              feature_names = [i.replace('customer','patient')[:16] for i in data_cat.columns],    
              base_values=ensemble_expected_value)

        plt.figure()
        shap.plots.waterfall(  ensemble_explanation,5)
 
        ############################
        y_pred = ensemble_shap_values[subject].values[0].sum() + ensemble_expected_value

        visualise_prediction(
    round(y_pred,1), [y_pred +  width1_left[subject] , y_pred +  width1_right[subject]],
    [y_pred +  width2_left[subject] , y_pred +  width2_right[subject]], y_train_all)
        print('plots for subject ' +str(subject) +' generated. Confidence intervals computed with '+ strategy_name)
     
          
                      
                
                        
                        
                
