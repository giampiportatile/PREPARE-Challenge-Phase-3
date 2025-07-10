import os
import numpy as np
import pandas as pd
import pickle
import itertools
import sklearn
import warnings
import featuretools as ft
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings("ignore")


feat_to_keep =['n_living_child_03', 'visit_med_03', 'tobacco_12', 'rrelgimp_03', 'rinc_pension_12', 'employment_12', 'glob_hlth_03', 'n_iadl_03', 'exer_3xwk_03', 'exer_3xwk_12', 'sad_12', 'decis_famil_03', 'adl_bath_03', 'seg_pop_12', 'insured_03', 'decis_personal_12', 'iadl_meds_03', 'enjoy_03', 'energetic_03', 'rrelgwk_12', 'hypertension_12', 'pem_def_mar_03', 'adl_walk_03', 'out_proc_12', 'iadl_meals_03', 'adl_bed_12', 'attends_club_12', 'happy_12', 'sinc_pension_03', 'rjob_hrswk_12', 'hincome_03', 'ragender', 'decis_personal_03', 'satis_ideal_12', 'volunteer_12', 'arthritis_12', 'tobacco_03', 'iadl_meals_12', 'lonely_03', 'restless_03', 'test_pres_03', 'rameduc_m', 'edu_gru_03', 'bmi_03', 'n_adl_12', 'memory_12', 'alcohol_12', 'hinc_assets_12', 'rearnings_03', 'migration_03', 'test_chol_03', 'cancer_03', 'diabetes_12', 'hinc_rent_03', 'imss_03', 'imss_12', 'care_adult_12', 'adl_bath_12', 'iadl_money_12', 'insur_other_03', 'sgender_03', 'stroke_12', 'happy_03', 'searnings_12', 'hard_12', 'adl_dress_03', 'a21_12', 'tired_12', 'wouldnt_change_12', 'hypertension_03', 'issste_12', 'restless_12', 'adl_toilet_12', 'insur_private_12', 'diabetes_03', 'stroke_03', 'adl_bed_03', 'hard_03', 'n_illnesses_12', 'adl_eat_12', 'test_tuber_03', 'cosas_imp_12', 'resp_ill_03', 'arthritis_03', 'hrt_attack_03', 'migration_12', 'rsocact_m_12', 'n_mar_03', 'adl_eat_03', 'cesd_depressed_03', 'hosp_12', 'adl_walk_12', 'rjlocc_m_03', 'hinc_assets_03', 'hrt_attack_12', 'insured_12', 'test_diab_03', 'n_mar_12', 'adl_toilet_03', 'cancer_12', 'n_adl_03', 'sad_03', 'a22_12', 'out_proc_03', 'visit_med_12', 'a16a_12', 'care_child_12', 'resp_ill_12', 'n_living_child_12', 'depressed_03', 'a33b_12', 'glob_hlth_12', 'reads_12', 'pem_def_mar_12', 'games_12', 'iadl_money_03', 'rjlocc_m_12', 'comms_tel_comp_12', 'rrelgimp_12', 'j11_12', 'year', 'rrfcntx_m_12', 'age_03', 'age_12', 'edu_gru_12', 'bmi_12', 'table_games_12', 'sewing_12', 'uid']
#these set of feature was computed using recursive feature elimination using Shapley values as described in the report
final = ['year of assessment','age_03.MEAN(customer.rafeduc_m)', 'hinc_assets_03', 'age_12.MEAN(customer.volunteer_12)', 'uid.SUM(customer.games_12)', 
                'age_12.MEAN(customer.satis_ideal_12)', 'care_adult_12', 'memory_12', 'uid.MAX(customer.hincome_03)', 'age_12.MEAN(customer.seg_pop_12)', 'age_12.MEAN(customer.visit_dental_12)', 'insur_other_03', 'age_12.MEAN(customer.n_depr_03)', 'hard_12', 'a21_12', 'age_12.MEAN(customer.diabetes_03)', 'employment_12', 'n_living_child_12', 'imss_03', 'age_12.MEAN(customer.bmi_03)', 'games_12', 'insur_private_12', 'n_adl_12', 'age_12.MEAN(customer.bmi_12)', 'age_12.MEAN(customer.tired_03)', 'age_12.MEAN(customer.table_games_12)', 'adl_walk_12', 'age_12.MAX(customer.rearnings_12)', 'age_03.MEAN(customer.hinc_business_12)', 'age_03.MEAN(customer.energetic_12)', 'out_proc_03', 'uid.SUM(customer.reads_12)', 'hrt_attack_12', 'hinc_rent_03', 'adl_dress_03', 'adl_toilet_03', 'seg_pop_12', 'age_12.MEAN(customer.rafeduc_m)', 'iadl_meals_03', 'happy_12', 'uid.SUM(customer.bmi_12)', 'uid.MAX(customer.games_12)', 'restless_03', 'migration_03', 'hard_03', 'age_12.MEAN(customer.reads_12)', 'rrfcntx_m_12', 'care_child_12', 'glob_hlth_12', 'uid.MAX(customer.edu_gru_03)', 'insured_03', 'enjoy_03', 'age_03.MEAN(customer.rjob_end_03)', 'insured_12', 'lonely_03', 'resp_ill_03', 'age_03.MEAN(customer.rearnings_03)', 'uid.MAX(customer.edu_gru_12)', 'age_03.MEAN(customer.depressed_03)', 'volunteer_12', 'out_proc_12', 'age_12.MIN(customer.a16a_12)', 'age_12.MEAN(customer.cancer_03)', 'age_03.MEAN(customer.care_adult_12)', 'cosas_imp_12', 'bmi_03', 'age_12.MEAN(customer.care_adult_12)', 'attends_club_12', 'pem_def_mar_03', 'age_12.MEAN(customer.searnings_03)', 'tobacco_03', 'adl_eat_12', 'age_03.MEAN(customer.games_12)', 'edu_gru_03', 'age_12.MEAN(customer.attends_club_12)', 'stroke_12', 'age_12.MEAN(customer.rjob_end_12)', 'age_12.MEAN(customer.games_12)', 'age_03.MEAN(customer.rearnings_12)', 'age_12.MEAN(customer.energetic_12)', 'age_12.MEAN(customer.cesd_depressed_03)', 'age_03.MIN(customer.rjob_end_03)', 'age_03.MEAN(customer.edu_gru_12)', 'age_12.MEAN(customer.exer_3xwk_12)', 'age_03.MEAN(customer.tobacco_12)', 'age_03.MEAN(customer.visit_dental_03)', 'test_tuber_03', 'age_12.MIN(customer.rjob_end_03)', 'age_03.MEAN(customer.bmi_12)', 'iadl_money_12', 'age_03.MEAN(customer.adl_bed_03)', 'sad_12', 'age_03.MEAN(customer.a16a_12)', 'adl_bath_03', 'table_games_12', 'hrt_attack_03', 'decis_personal_03', 'age_03.MEAN(customer.visit_dental_12)', 'age_12.MEAN(customer.n_mar_03)', 'rafeduc_m', 'rsocact_m_12', 'bmi_12', 'age_12.MEAN(customer.hincome_03)', 'age_03.MAX(customer.rearnings_12)', 'age_03.MEAN(customer.searnings_03)', 'stroke_03', 'hosp_12', 'iadl_meals_12', 'age_12.MEAN(customer.edu_gru_12)', 'age_03.MEAN(customer.exer_3xwk_12)', 'age_03.MAX(customer.hinc_rent_03)', 'age_03.MEAN(customer.rjob_end_12)', 'age_03.MEAN(customer.comms_tel_comp_12)', 'iadl_meds_03', 'age_03.MEAN(customer.hincome_03)', 'tobacco_12', 'age_12.MEAN(customer.rameduc_m)', 'age_12.MEAN(customer.alcohol_12)', 'age_03.MEAN(customer.care_child_12)', 'age_12.MEAN(customer.insur_private_03)', 'age_12.MEAN(customer.attends_class_12)', 'age_12.SUM(customer.searnings_12)', 'age_03.MEAN(customer.rjob_hrswk_12)', 'age_12.SUM(customer.rjob_hrswk_12)', 'age_03.MIN(customer.a16a_12)', 'age_03.MEAN(customer.volunteer_12)', 'adl_eat_03', 'age_03.MEAN(customer.rameduc_m)', 'age_12.MEAN(customer.rearnings_12)', 'j11_12', 'age_12.MEAN(customer.tobacco_12)', 'age_12.MIN(customer.n_mar_03)', 'age_03.MEAN(customer.restless_12)', 'n_living_child_03', 'age_12.MEAN(customer.care_child_12)', 'age_03.MEAN(customer.sewing_12)', 'rjob_hrswk_12', 'age_12.MEAN(customer.act_mant_12)', 'age_03.MEAN(customer.attends_club_12)', 'age_03.MEAN(customer.cancer_03)', 'diabetes_03', 'cancer_03', 'uid.MAX(customer.reads_12)', 'age_03.MEAN(customer.act_mant_12)', 'age_03.MEAN(customer.seg_pop_12)', 'age_12.MEAN(customer.hinc_rent_03)', 'age_12.MEAN(customer.a16a_12)', 'imss_12', 'age_03.MEAN(customer.depressed_12)', 'decis_personal_12', 'age_03.MEAN(customer.hincome_12)', 'age_03.MEAN(customer.bmi_03)', 'age_12', 'age_03.MIN(customer.rjob_end_12)', 'age_03.MEAN(customer.tobacco_03)', 'age_03.MEAN(customer.diabetes_12)', 'cancer_12', 'age_03.MEAN(customer.hinc_cap_12)', 'age_03.MEAN(customer.diabetes_03)', 'diabetes_12', 'age_03.MEAN(customer.hinc_rent_03)', 'age_03.MEAN(customer.attends_class_12)', 'age_03.MEAN(customer.sinc_pension_12)', 'age_12.MAX(customer.n_iadl_12)', 'a16a_12', 'age_12.MEAN(customer.rjob_hrswk_12)', 'reads_12', 'rjlocc_m_12', 'year', 'age_03.MEAN(customer.edu_gru_03)', 'age_03', 'edu_gru_12', 'uid.SUM(customer.edu_gru_12)', 'sewing_12']

www = [i for i in final if i not in feat_to_keep]

#make sure the script is executed from the parent directory where the repo has been cloned
base_dir = os.getcwd()
data_found =False
try:
  y =  pd.read_csv(os.path.join(base_dir, 'data','raw','train_labels.csv'))
  train = pd.read_csv(os.path.join(base_dir, 'data','raw','train_features.csv'))
  test = pd.read_csv(os.path.join(base_dir, 'data','raw','test_features.csv'))
  data_found =True
except:
  print('data not found in the /data/raw folder. Please either a) add the data in the appropriate folder or b) make sure this Python script is executed from the parent directory where the git repository has been downloaded')  
  
if data_found:
    data =pd.concat((train,test))
    
    ids = train['uid'].unique()
    
    ####################################################################
    # decis_personal_* loads as float for 03 and object for 12. Make this consistent
    r = {'1. A lot': '1', '2. A little': '2', '3. None': '3'}
    data['decis_personal_12'] = data['decis_personal_12'].replace(r).astype('float')
    object_cols = data.select_dtypes(include=['object']).columns
    for col in object_cols:
        data[col] = data[col].astype('category')
        data[col] = pd.Categorical(data[col])
    
    data_gbm = data.copy()
    ##########################################################################
    data =pd.concat((train,test))
    data['decis_personal_12'] = data['decis_personal_12'].replace(r).astype('float')
    #for catboost convert data into category after filling the nans with '' as catboost does not support nan in categorical features
    data[object_cols]=  data[object_cols ].fillna(' ')
    # Convert the object columns to category dtype
    for col in object_cols:
        data[col] = data[col].astype('category')
        data[col] = pd.Categorical(data[col])
        
    data_cat = data.copy()
    del train, test, data
    ###########################################################################
    
    ###########################################################################
    ##convert certain 'ordered' categorical features (e.g age) into numerical features. This is expected to 
    #improve the explanatory power of the model, as, after the conversion  we can exploit the 'ordered' nature of these features
    candidate = [
        'age_',
        'n_living_child',
        'glob_hlth',
        'bmi',
        'satis_ideal',
        'memory',
        'rafeduc',
        'rameduc',
        'rsocact_m',
        'rrfcntx_m',
        'exer_3xwk',
        'edu_']
    temp = pd.DataFrame()
    q = []
    for i in object_cols:
        for j in candidate:
            if j in i:
    
                idx = np.nonzero(data_gbm[i].notnull().values)[0]
                temp[i] = pd.Series(index=data_gbm.index).astype(str)
                temp[i].iloc[idx] = data_gbm[i].iloc[idx].apply(
                    lambda x: x.split('.')[0])
                temp[i] = temp[i].astype(float)
                break
    # there are 4 subjects for whom the age in 2003 is higher than the age in 2012
    # This is clearly an error. We replace the age feature for those subjects
    # with a missing value.
    idx = np.nonzero((temp['age_03'] > temp['age_12']).values)[0]
    temp.iloc[idx]['age_03'] = np.nan
    temp.iloc[idx]['age_12'] = np.nan
    converted = temp.columns
    orig = [i for i in data_cat.columns if i not in converted]
    data_cat = pd.concat([data_cat[orig], temp], axis=1)
    data_gbm = pd.concat([data_gbm[orig], temp], axis=1)
    
    ###############################################################################
    y =  pd.read_csv(os.path.join(base_dir, 'data','raw','train_labels.csv'))
    ss = pd.read_csv(os.path.join(base_dir, 'data','raw','submission_format.csv'))
                                  
    y = pd.concat((y,ss))
    data_cat =  pd.merge(data_cat, y, on='uid', how='left')
    data_gbm =  pd.merge(data_gbm, y, on='uid', how='left')
    del data_cat['composite_score']
    ################################################################################
    idx_train =  np.nonzero (data_gbm['uid'].isin(ids).values)[0]
    idx_test =  np.nonzero (~data_gbm ['uid'].isin(ids).values)[0]
    #######################################################################################
    ####add the estimated age at time of assessment feature
    data = data_gbm.copy()
    data['year of assessment'] =np.nan
    both  =np.nonzero( (data['age_03']+data['age_12']).notnull().values)[0]
    only_2012  =np.nonzero( (data['age_03'].isnull() & (data['age_12']).notnull()).values)[0]
    only_2003  =np.nonzero( (data['age_12'].isnull() & (data['age_03']).notnull()).values)[0]
    idx_2016  =np.nonzero( (data['year']==2016).values)[0]
    idx_2021  =np.nonzero( (data['year']==2021).values)[0]
    a =list(set(both).intersection(set(idx_2016)))
    data['year of assessment'] .iloc[a] = (data['age_03']+data['age_12']).iloc[a]/2+.95
    a =list(set(both).intersection(set(idx_2021)))
    data['year of assessment'] .iloc[a] = (data['age_03']+data['age_12']).iloc[a]/2+ 1.35
    a =list(set(only_2012).intersection(set(idx_2016)))
    data['year of assessment'] .iloc[a] = data['age_12'].iloc[a]+.4
    a =list(set(only_2012).intersection(set(idx_2021)))
    data['year of assessment'] .iloc[a] =  data['age_12'].iloc[a]+.85
    a =list(set(only_2003).intersection(set(idx_2016)))
    data['year of assessment'] .iloc[a] = data['age_03'].iloc[a]+1.3
    a =list(set(only_2003).intersection(set(idx_2021)))
    data['year of assessment'] .iloc[a] = data['age_03'].iloc[a]+1.8
    data_gbm =data.copy()
    data_cat['year of assessment'] = data['year of assessment'] 
    del data
    ########################################################################################
    #########################################################################################
    data_cat['id ']=range(len(data_cat))
    es = ft.EntitySet(id='loan')
    es = es.add_dataframe(dataframe_name= 'customer', dataframe =data_cat ,index ='id')
    es = es.normalize_dataframe(base_dataframe_name='customer', new_dataframe_name='uid', index='uid')
    es = es.normalize_dataframe(base_dataframe_name='customer', new_dataframe_name='year', index='year')
        
    default_agg_primitives =  ["sum", "max", "min", "mean", "count"]
    feature_matrix, feature_names = ft.dfs(entityset = es, target_dataframe_name = 'customer',
                                               agg_primitives=default_agg_primitives, 
                                           max_depth = 2, features_only=False, verbose = True)
                                               
    data_cat_orig = data_cat[feat_to_keep[:-1]]
    data_gbm_orig =data_gbm[feat_to_keep[:-1]]
    y =data_gbm['composite_score']
    
    feature_matrix =feature_matrix[[ i for i in feature_matrix.columns if i in www]]
    
    data_cat = pd.concat([data_cat,feature_matrix],axis =1)
    data_gbm = pd.concat([data_gbm,feature_matrix],axis =1)
    del feature_matrix
    
    data_gbm   = data_gbm.loc[:, ~data_gbm.columns.duplicated()].drop(columns =['composite_score','uid'], axis=1)
    data_cat  =  data_cat.loc[:, ~data_cat.columns.duplicated()].drop(columns =['id','uid'], axis=1)
    ####################################################################################################
    # prepare data for TabM
    #note that the y variable for the test set (released after the end of the competition) is only used to assess the performance
    test = pd.read_csv(os.path.join(base_dir, 'data','raw','test_features.csv'))
    y_test = pd.read_csv(os.path.join(base_dir, 'data','raw','test_labels.csv'))
    
    float_cols = data_cat.select_dtypes(include=['float']).columns
    cat_cols = [
        i for i in data_cat.columns if (
            i not in float_cols) & (
                i != 'uid') & (
                    i != 'id')]
    
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean .fit(data_cat.iloc[idx_train][float_cols])
    X_cont = imp_mean .transform(data_cat[float_cols]).astype(np.float32)
    #idx_test = range(len(idx_train), data_cat.shape[0])
    
    data_numpy = {
        'train': {'x_cont': X_cont[idx_train], 'y': y.iloc[idx_train].values},
        'val': {'x_cont': X_cont[idx_test], 'y': y_test['composite_score'].values},
        'test': {'x_cont': X_cont[idx_test], 'y': y_test['composite_score'].values}}
    
    
    # The noise is added to improve the output of QuantileTransformer in some cases.
    X_cont_train_numpy = data_numpy['train']['x_cont']
    noise = (
        np.random.default_rng(0)
        .normal(0.0, 1e-5, X_cont_train_numpy.shape)
        .astype(X_cont_train_numpy.dtype)
    )
    preprocessing = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(len(idx) // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9,
    ).fit(X_cont_train_numpy + noise)
    del X_cont_train_numpy
    
    # Apply the preprocessing to the numerical features
    for part in data_numpy:
        data_numpy[part]['x_cont'] = preprocessing.transform(
            data_numpy[part]['x_cont'])
    
    #one hot encoding all the caterogical features and concatenate with the numerical features
    for i in range(len(cat_cols)):
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(data_cat[[cat_cols[i]]].iloc[idx])
        if i == 0:
            temp = ohe.transform(
                data_cat[[cat_cols[i]]]).toarray().astype(np.float32)
        else:
            temp = np.concatenate((temp, ohe.transform(
                data_cat[[cat_cols[i]]]).toarray().astype(np.float32)), 1)
    data_numpy['train']['x_cont'] = np.concatenate(
        (data_numpy['train']['x_cont'], temp[idx_train]), 1)
    data_numpy['val']['x_cont'] = np.concatenate(
        (data_numpy['val']['x_cont'], temp[idx_test]), 1)
    data_numpy['test']['x_cont'] = np.concatenate(
        (data_numpy['test']['x_cont'], temp[idx_test]), 1)
    ############################################################################################
    print('saving processed data')
    data_cat.to_csv(os.path.join(base_dir, 'data','processed','data_cat.csv'))
    data_gbm.to_csv(os.path.join(base_dir, 'data','processed','data_gbm.csv'))
    a = open(os.path.join(base_dir, 'data','processed','data_processed_for_tabM.pkl'), 'wb')
    pickle.dump(data_numpy, a)
    a.close()
    print('processed data for Catboost, Lightgbm and TabM saved')
    ############################################################################################
