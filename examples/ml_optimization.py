import pandas as pd
from sklearn.preprocessing import Imputer

from mlopt.sklearn_tune import AdaBoostClassifierOpt,KNNOpt,LogisticRegressionOpt
from mlopt.sklearn_tune import RandomForestClassifierOpt,SVCOpt
from mlopt.xgb_tune import XGBoostOpt
from mlopt.lgb_tune import LGBMOpt

# For the hyperparameter optimization examples we will use the kaggle dataset
# from the Porto Seguro competition. In this competition, you’re challenged to 
# build a model that predicts the  probability that a driver will initiate an 
# auto insurance claim in the next year.
# The data can be found through the link below:
# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data

# directory where results will be saved
save_dir = '.../output'

# specify train and test data paths
path_train = '.../input/train.csv'
path_test = '.../input/test.csv'

# load data
train_raw = pd.read_csv(path_train,na_values=-1)
train_labels = train_raw['target'].values
train_ids = train_raw['id'].values
train_data = train_raw.drop(['id', 'target'], axis=1)

# remove uncorrelated features
unwanted = train_data.columns[train_data.columns.str.startswith('ps_calc_')]
train_data = train_data.drop(unwanted, axis=1)

X = train_data
y = train_labels

# replace NaN's by median values per column
imp = Imputer(missing_values='NaN',strategy='median',axis=0)
X = imp.fit_transform(X)

# set cv parameters
params_cv = {'cv_folds':5,
             'early_stopping_rounds':100,
             'scoring':'roc_auc'}

# optimize parameters for logistic regression using gridsearch over a range of parameters
# note: no pre-processing steps will be done for the examples below
lr = LogisticRegressionOpt(X,y,params_cv=params_cv,model_name='lr_porto_seguro',save_dir=save_dir)
lr.tune_params()
print('Best model parameters:')
print(lr.best_model)
lr.save_model()

# we will reduce the size of the dataset for the rest of the examples
# to keep the training time reasonable
X_slice = X[:10000,:]
y_slice = y[:10000]

# adaboost
ada = AdaBoostClassifierOpt(X_slice,y_slice,params_cv=params_cv,model_name='ada_porto_seguro',save_dir=save_dir)
ada.tune_params()
print('Best model parameters:')
print(ada.best_model)
ada.save_model()

# we can set the range of the parameters tuned manually
# more info about the parameters to be tuned for each algorithm
# can be found in the MLOpt class

# kNN
tune_range = {'n_neighbors':[10,20,50,100]}
knn = KNNOpt(X_slice,y_slice,params_cv=params_cv,tune_range=tune_range,model_name='knn_porto_seguro',save_dir=save_dir)
knn.tune_params()
print('Best model parameters:')
print(knn.best_model)
knn.save_model()

# SVC
tune_range = {'C':[0.01,0.1,1,10,100],
              'gamma':[0.01,0.1,1]}
svc = SVCOpt(X_slice,y_slice,params_cv=params_cv,tune_range=tune_range,model_name='svc_porto_seguro',save_dir=save_dir)
svc.tune_params()
print('Best model parameters:')
print(svc.best_model)
svc.save_model()

# we can also set the parameters that are not tuned

# random forest
params = {'criterion':'gini'}
tune_range = {'n_estimators':[50,100,200,500],
              'max_features':[0.5,0.75],
              'min_samples_leaf':[0.001,0.01]}
rf = RandomForestClassifierOpt(X_slice,y_slice,params=params,params_cv=params_cv,tune_range=tune_range,
                               model_name='rf_porto_seguro',save_dir=save_dir)
rf.tune_params()
print('Best model parameters:')
print(rf.best_model)
rf.save_model()

# increase dataset for xgboost and lightgbm
X_slice = X[:50000,:]
y_slice = y[:50000]

# more complex hyperparameter optimization is done for the xgboost and lightgbm algorithms
# the amount of hyperparameters that need to be tuned does not favour a gridsearch
# approach, so as a result the parameters are optimized in different steps:
# 1. fix learning rate and number of estimators for tuning tree-based parameters
# 2. tune max_depth and min_child_weight
# 3. tune gamma
# 4. tune subsample and colsample_bytree
# 5. tune l2 regularization
# 6. reduce learning rate and start over until stopping criterium reached
xgb = XGBoostOpt(X_slice,y_slice,params_cv=params_cv,max_rounds=2,model_name='xgb_porto_seguro',save_dir=save_dir)
xgb.tune_params()
print('Best model score: %f.' %(xgb.best_score))
print('Best model parameters:')
print(xgb.best_model)
xgb.save_model()

# a similar approach is taken for lightgbm:
# 1. fix learning rate and number of estimators for tuning tree-based parameters
# 2. tune num_leaves and min_data_in_leaf
# 3. tune min_gain_to_split
# 4. tune bagging_fraction + bagging_freq and feature_fraction
# 5. tune lambda_l2
# 6. reduce learning rate and start over until stopping criterium reached
lgb = LGBMOpt(X_slice,y_slice,params_cv=params_cv,max_rounds=2,model_name='lgb_porto_seguro',save_dir=save_dir)
lgb.tune_params()
print('Best model score: %f.' %(lgb.best_score))
print('Best model parameters:')
print(lgb.best_model)
lgb.save_model()
