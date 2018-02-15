import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np
import time

from mlopt.ml_tune import MLOpt

class LGBMOpt(MLOpt):
    """
    LightGBM optimizer.
    
    The function "tune_params" will optimize the lgbm classifier as follows:
        1. fix learning rate and number of estimators for tuning tree-based parameters
        2. tune num_leaves and min_data_in_leaf
        3. tune min_gain_to_split
        4. tune bagging_fraction + bagging_freq and feature_fraction
        5. tune lambda_l2
        6. reduce learning rate and start over until stopping criterium reached

    Currently only for classification problems.

    Source: http://testlightgbm.readthedocs.io/en/latest/Parameters-tuning.html
    """
    
    def __init__(self,X,y,params={},params_cv={},tune_range={},max_rounds=5,running_rounds=2,
                 max_runtime=129600,score_type='high',balance_class=False,categorical_feature=None,
                 save_dir=None,model_name=None):
        
        MLOpt.__init__(self,X,y,'lightgbm',params=params,params_cv=params_cv,
                       tune_range=tune_range,score_type=score_type,max_runtime=max_runtime,
                       max_rounds=max_rounds,running_rounds=running_rounds,balance_class=balance_class,
                       save_dir=save_dir,model_name=model_name)
        
        self.categorical_feature = categorical_feature
        
        self._dict_map = {'min_child_samples':'min_data_in_leaf',
                          'min_split_gain':'min_gain_to_split',
                          'subsample':'bagging_fraction',
                          'subsample_freq':'bagging_freq',
                          'colsample_bytree':'feature_fraction',
                          'reg_alpha':'lambda_l1',
                          'reg_lambda':'lambda_l2'}
    
    
    def application_type(self):
        """
        check whether binary or multi-class classification
        """
        unique, counts = np.unique(self.y, return_counts=True)
        if len(unique)==2:
            self._params['application'] = 'binary'
        elif len(unique)>2:
            self._params['application'] = 'multiclass'
            self._params['num_class'] = len(unique)
        else:
            raise ValueError('Labels y are not suitable for a LightGBM classification problem: \
                                  only "binary" and "multiclass" supported for now.')
        return self
    
    
    def get_n_estimators(self):
        """
        returns optimal number of estimators using CV on training set
        """
        lgb_param = {}
        for _params_key,_params_value in self._params.items():
            if _params_key in self._dict_map.keys():
                lgb_param[self._dict_map[_params_key]] = _params_value
            else:
                lgb_param[_params_key] = _params_value
        
        if self.balance_class:
            lgb_train = lgb.Dataset(self.X, label=self.y, weight=self.get_label_weights())
        else:
            lgb_train = lgb.Dataset(self.X, label=self.y)
        
        kwargs_cv = {'num_boost_round':self.params['n_estimators'],
                     'nfold':self.params_cv['cv_folds'],
                     'early_stopping_rounds':self.params_cv['early_stopping_rounds'],
                     'stratified':self.params_cv['stratified']}
        
        try: # check if custom evalution function is specified
            if callable(self.params_cv['feval']):
                kwargs_cv['feval'] = self.params_cv['feval']
        except KeyError:
            kwargs_cv['metrics'] = self.params_cv['metrics']
        
        if type(self.categorical_feature)==list:
            kwargs_cv['categorical_feature'] = self.categorical_feature
        else:
            kwargs_cv['categorical_feature'] = 'auto'
        
        cvresult = lgb.cv(lgb_param,lgb_train,**kwargs_cv)
        self._params['n_estimators'] = int(len(cvresult[kwargs_cv['metrics'] + \
                                            '-mean'])/(1-1/self.params_cv['cv_folds']))
        return self
    
    
    def tune_params(self):
        """
        tune specified (and default) parameters
        """
        self._start_time = time.time()
        self.default_params() # set default parameters
        self.application_type() # set classification type
        self.score_init() # set initial score
        iround = 0
        while iround<self.max_rounds:
            print('\nLearning rate for iteration %i: %f.' %(iround+1,self._params['learning_rate']))
            while self._step<5:
                istep_time = time.time()
                if self._step==0:
                    self.get_n_estimators()
                else:
                    self.apply_gridsearch(LGBMClassifier(**self._params))
                self.print_progress(istep_time,iround=iround,max_rounds=self.max_rounds) # print params and performance
                self._step+=1
            
            # store model each iteration
            self._params_iround[iround] = {}
            for key,value in self._params.items():
                self._params_iround[iround][key] = value
            self._params_iround[iround]['model_score'] = self.best_score
            
            # check if max_runtime is breached
            if (time.time() - self._start_time) > self.max_runtime:
                print('Tuning stopped after iteration %i. Max runtime of %i sec exceeded.'
                            %(iround+1,self.max_runtime))
                return
            
            # early stopping criterium
            if (iround>=self.running_rounds and 
                    self.best_score==self._params_iround[max(0,iround-self.running_rounds)]['model_score']):
                print('Tuning stopped after iteration %i. No model improvement for %i consecutive rounds.'
                            %(iround+1,self.running_rounds))
                return
            
            # update learning rate and reset n_estimators for next iteration
            if iround<self.max_rounds-1:
                self.update_learning_rate()
            
            if self._stop_learning:
                print('Tuning stopped after iteration %i. Minimum learning rate %f reached.'
                            %(iround+1,self._min_learning_rate))
                return
            
            self._step=0
            iround+=1
        
        return
