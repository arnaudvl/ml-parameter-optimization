import numpy as np
import os
from sklearn.base import clone
from sklearn.decomposition import FactorAnalysis,FastICA,KernelPCA,PCA,SparsePCA,TruncatedSVD
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import time

class MLOpt():
    """
    Contains optimization functions, default parameters and tuning ranges to optimize 
    a variety of ML algorithms.
    
    Currently supports:
        - xgboost
        - lightgbm
        - random forest
        - adaboost
        - kNN
        - logistic regression
        - SVC
    """
    
    def __init__(self,X,y,model_type,params={},params_cv={},tune_range={},max_runtime=86400,
                 score_type='high',dim_reduction=None,max_rounds=5,running_rounds=2,balance_class=False,
                 save_dir=None,model_name=None):
        
        self.X = X
        self.y = y        
        self.model_type = model_type        
        self.save_dir = save_dir
        self.model_name = model_name
        self.params = params
        self.params_cv = params_cv
        self.tune_range = tune_range
        self.max_runtime = max_runtime
        self.score_type = score_type
        self.dim_reduction = dim_reduction
        self.balance_class = balance_class
        
        # parameters specific to ML algorithms with multiple iterations:
        # like XGBoost and LightGBM
        self.max_rounds = max_rounds
        self.running_rounds = running_rounds
        self._stop_learning = False
        self._params_iround = {}
        self._min_learning_rate = 0.0001
        
        # other parameters
        self.best_score = None
        self.best_model = None
        self._score_mult = None
        self._temp_score = None
        self._start_time = None
        self._greater_is_better = True
        self._step = 0
        self._pipeline = False
        
        self._params = {}
        for key,value in self.params.items():
            self._params[key] = value
        
        # parameters determined by model type
        self._params_default = None
        self._tune_range_default = None
        
        # default parameters for cross validation
        self._params_cv_default = {'cv_folds':5,
                                   'early_stopping_rounds':100,
                                   'stratified':True,
                                   'iid':False,
                                   'metrics':'auc',
                                   'scoring':'roc_auc'}
                                   
        if self.model_type=='lightgbm':
            self._params_cv_default['n_jobs'] = 1
        else:
            self._params_cv_default['n_jobs'] = -1
        
        # default parameters for xgboost
        self._params_default_xgb = {'learning_rate':0.1,
                                    'n_estimators':int(1e4),
                                    'max_depth':5,
                                    'min_child_weight':1,
                                    'gamma':0,
                                    'subsample':0.8,
                                    'colsample_bytree':0.8,
                                    'reg_alpha':0,
                                    'reg_lambda':1}
        
        self._tune_range_default_xgb = {'max_depth':list(range(3,10,2)),
                                        'min_child_weight':[1,5,20,50],
                                        'gamma':[i/10. for i in range(0,4)],
                                        'subsample':[i/10.0 for i in range(6,10)],
                                        'colsample_bytree':[i/10.0 for i in range(6,10)],
                                        'reg_alpha':[1e-6, 1e-2, 0.1, 1, 100],
                                        'reg_lambda':[1e-6, 1e-2, 0.1, 1, 100]}
        
        self._params_tune_xgb = [['n_estimators'],
                                 ['max_depth','min_child_weight'],
                                 ['gamma'],
                                 ['subsample','colsample_bytree'],
                                 ['reg_alpha','reg_lambda']]
        
        # default parameters for lightgbm
        self._params_default_lgb = {'learning_rate':0.1,
                                    'n_estimators':int(1e4),
                                    'num_leaves':31, # num_leaves<=2**max_depth-1
                                    'min_child_samples':20,
                                    'min_split_gain':0,
                                    'subsample':0.8,
                                    'subsample_freq':5,
                                    'colsample_bytree':0.8,
                                    'reg_alpha':0,
                                    'reg_lambda':1,
                                    'n_jobs':1}
        
        self._tune_range_default_lgb = {'num_leaves':[7,31,127,511], # 2**max_depth-1
                                        'min_child_samples':[5,20,50,100],
                                        'min_split_gain':[i/10. for i in range(0,4)],
                                        'subsample':[i/10.0 for i in range(6,10)],
                                        'colsample_bytree':[i/10.0 for i in range(6,10)],
                                        'reg_alpha':[1e-6, 1e-2, 0.1, 1, 100],
                                        'reg_lambda':[1e-6, 1e-2, 0.1, 1, 100]}
        
        self._params_tune_lgb = [['n_estimators'],
                                 ['num_leaves','min_child_samples'],
                                 ['min_split_gain'],
                                 ['subsample','colsample_bytree'],
                                 ['reg_alpha','reg_lambda']]
        
        # default parameters for random forests
        self._params_default_rf = {'n_estimators':100,
                                   'criterion':'gini',
                                   'max_features':0.5,
                                   'min_samples_leaf':1,
                                   'n_jobs':-1,
                                   'random_state':None,
                                   'warm_start':False,
                                   'class_weight':None}
        
        self._tune_range_default_rf = {'n_estimators':[50,100,200,500,1000],
                                       'max_features':[0.25,0.5,0.75],
                                       'min_samples_leaf':[0.0001,0.001,0.01]}
        
        self._params_tune_rf = [['n_estimators','max_features','min_samples_leaf']]
        
        # default parameters for adaboost
        self._params_default_ada = {'n_estimators':50,
                                    'learning_rate':1.}
        
        self._params_default_ada_tree = {'criterion':'gini',
                                         'max_depth':None,
                                         'min_samples_leaf':0.0001,
                                         'max_features':None,
                                         'random_state':None,
                                         'class_weight':None}
        
        self._tune_range_default_ada = {'n_estimators':[50,100,200,500,1000],
                                        'learning_rate':[0.01,0.1,1.,2.]}
        
        self._params_tune_ada = [['n_estimators','learning_rate']]
        
        # default parameters for kNN
        self._params_default_knn = {'n_neighbors':5,
                                    'weights':'uniform',
                                    'p':2,
                                    'n_jobs':-1}
        
        self._tune_range_default_knn = {'n_neighbors':[2,5,10,20,50,100]}
        
        self._params_tune_knn = [['n_neighbors']]
        
        # default parameters for logistic regression
        self._params_default_lr = {'penalty':'l2',
                                   'C':1.,
                                   'class_weight':None,
                                   'random_state':None,
                                   'max_iter':1000,
                                   'warm_start':False,
                                   'n_jobs':-1}
        
        self._tune_range_default_lr = {'C':[0.001,0.01,0.1,1,10,100]}
        
        self._params_tune_lr = [['C']]
        
        # default parameters for SVC
        self._params_default_svc = {'C':1.,
                                    'kernel':'rbf',
                                    'gamma':'auto',
                                    'probability':True,
                                    'class_weight':None,
                                    'random_state':None}
        
        self._tune_range_default_svc = {'C':[0.001,0.01,0.1,1,10,100],
                                        'gamma':[0.001, 0.01, 0.1, 1]}
        
        self._params_tune_svc = [['C','gamma']]
        
        # define default parameters based on model_type
        if self.model_type=='random-forest':
            self._params_default = self._params_default_rf
            self._tune_range_default = self._tune_range_default_rf
            self._params_tune = self._params_tune_rf
        elif self.model_type=='adaboost':
            self._params_default = self._params_default_ada
            self._tune_range_default = self._tune_range_default_ada
            self._params_tune = self._params_tune_ada
        elif self.model_type=='knn':
            self._params_default = self._params_default_knn
            self._tune_range_default = self._tune_range_default_knn
            self._params_tune = self._params_tune_knn
        elif self.model_type=='logistic-regression':
            self._params_default = self._params_default_lr
            self._tune_range_default = self._tune_range_default_lr
            self._params_tune = self._params_tune_lr
        elif self.model_type=='svc':
            self._params_default = self._params_default_svc
            self._tune_range_default = self._tune_range_default_svc
            self._params_tune = self._params_tune_svc
        elif self.model_type=='xgboost':
            self._params_default = self._params_default_xgb
            self._tune_range_default = self._tune_range_default_xgb
            self._params_tune = self._params_tune_xgb
        elif self.model_type=='lightgbm':
            self._params_default = self._params_default_lgb
            self._tune_range_default = self._tune_range_default_lgb
            self._params_tune = self._params_tune_lgb
        else:
            raise ValueError('%s is not a supported ML model. Valid inputs are: "random-forest","adaboost" \
                             ,"knn","logistic-regression","svc","xgboost","lightgbm".' %(self.model_type))
        
    
    def score_init(self):
        """
        set initial value for score and set greater/lower score better
        """
        if self.score_type=='high':
            self.best_score = -1e10
            self._temp_score = -1e10
            self._score_mult = 1
            self._greater_is_better = True
        elif self.score_type=='low':
            self.best_score = 1e10
            self._temp_score = 1e10
            self._score_mult = -1
            self._greater_is_better = False
        return self
    
    
    def set_default(self,_a,_b):
        """
        helper function to get default values if input argument not specified
        """
        for key,value in _b.items():
            try:
                _a[key]
            except KeyError:
                _a[key] = value
        return _a
    
    
    def default_params(self):
        """
        return default initial values for parameters not specified in "params", "params_cv" or "tune_range"
        """        
        self.params = self.set_default(self.params,self._params_default)
        self._params = self.set_default(self._params,self._params_default)
        self.params_cv = self.set_default(self.params_cv,self._params_cv_default)
        self.tune_range = self.set_default(self.tune_range,self._tune_range_default)
        return self
    
    
    def get_params_tune(self):
        """
        return dict of parameters to be tuned with values
        """
        params_tune = self._params_tune[self._step]
        params_tune_dict = {}
        for key,value in self.tune_range.items():
            if key in params_tune:
                params_tune_dict[key] = value
        return params_tune_dict
    
    
    def get_label_weights(self):
        """
        rebalance label weights for each class to remove imbalance
        """
        n_samples = len(self.y)
        unique, counts = np.unique(self.y, return_counts=True)
        label_count = dict(zip(unique, counts))
        label_weights = np.empty(self.y.shape)
        for key,value in label_count.items():
            label_weights[np.where(self.y==key)[0]] = n_samples/value
        return label_weights
    
    
    def print_progress(self,step_time,iround=None,max_rounds=None):
        """
        print update on tuning progress
        """
        if iround is not None and max_rounds is not None:
            print('\nIteration %i/%i.' %(iround+1,max_rounds))
        else:
            print('\nIteration %i/%i.' %(1,1))
        print('...Step %i: tune %s.' %(self._step,self._params_tune[self._step]))
        print('...Time elapsed for this iteration: %f min.' %((time.time() - step_time)/60))
        print('...Total time elapsed: %f min.' %((time.time() - self._start_time)/60))
        for item in self._params_tune[self._step]:
            print('...Optimal value %s: %f.' %(item,self._params[item]))
        print('...Model score: %f.' %(self._temp_score))
    
    
    def dim_reduction_method(self):
        """
        select dimensionality reduction method
        """
        if self.dim_reduction=='pca':
            return PCA()
        elif self.dim_reduction=='factor-analysis':
            return FactorAnalysis()
        elif self.dim_reduction=='fast-ica':
            return FastICA()
        elif self.dim_reduction=='kernel-pca':
            return KernelPCA()
        elif self.dim_reduction=='sparse-pca':
            return SparsePCA()
        elif self.dim_reduction=='truncated-svd':
            return TruncatedSVD()
        elif self.dim_reduction!=None:
            raise ValueError('%s is not a supported dimensionality reduction method. Valid inputs are: \
                             "pca","factor-analysis","fast-ica,"kernel-pca","sparse-pca","truncated-svd".' 
                             %(self.dim_reduction))
    
    
    def update_learning_rate(self):
        """
        update learning rate for next iteration
        used in xgboost_tune and lgb_tune
        """
        self._stop_learning = False
        learning_rate_new = 2/self._params['n_estimators']
        
        if self._params['learning_rate']>learning_rate_new:
            self._params['learning_rate'] = max(learning_rate_new,self._min_learning_rate)
        elif self._params['learning_rate']==learning_rate_new and learning_rate_new==self._min_learning_rate:
            self._stop_learning = True
        elif self._params['learning_rate']<learning_rate_new:
            self._params['learning_rate'] = 0.5 * (self._params['learning_rate'] + self._min_learning_rate)
#            self._params['learning_rate'] = (self._params['learning_rate'] - (self._params['learning_rate'] - 
#                                            self._min_learning_rate) / (self.max_rounds - iround - 1))
        return self
    
    
    def apply_gridsearch(self,model):
        """
        apply grid search on ml algorithm to specified parameters
        returns updated best score and parameters
        """
        # check if custom evalution function is specified
        if callable(self.params_cv['scoring']):
            scoring = make_scorer(self.params_cv['scoring'],greater_is_better=self._greater_is_better)
        else:
            scoring = self.params_cv['scoring']
        
        gsearch = GridSearchCV(estimator=model,param_grid=self.get_params_tune(),scoring=scoring,
                               iid=self.params_cv['iid'],cv=self.params_cv['cv_folds'],n_jobs=self.params_cv['n_jobs'])
        gsearch.fit(self.X,self.y)
        
        # update best model if best_score is improved
        if (gsearch.best_score_ * self._score_mult) > (self.best_score * self._score_mult):
            self.best_model = clone(gsearch.best_estimator_)
            self.best_score = gsearch.best_score_
        
        # update tuned parameters with optimal values
        for key,value in gsearch.best_params_.items():
            self._params[key] = value
        self._temp_score = gsearch.best_score_
        return self
    
    
    def save_model(self):
        """
        save tuned model with model_name in directory save_dir
        """
        if type(self.save_dir)!=str or type(self.model_name)!=str:
            raise ValueError('"save_dir" and "model_name" need to be specified as strings.')
        os.chdir(self.save_dir)
        joblib.dump(self.best_model,self.model_name + '.pkl',compress=1)
