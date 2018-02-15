import time
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from mlopt.ml_tune import MLOpt

class XGBoostOpt(MLOpt):
    """
    XGBoost optimizer.
    
    The function "tune_params" will optimize the xgboost classifier as follows:
        1. fix learning rate and number of estimators for tuning tree-based parameters
        2. tune max_depth and min_child_weight
        3. tune gamma
        4. tune subsample and colsample_bytree
        5. tune l2 regularization
        6. reduce learning rate and start over until stopping criterium reached

    Currently only for classification problems.

    Source: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    """
    
    def __init__(self,X,y,params={},params_cv={},tune_range={},max_rounds=2,running_rounds=2,
                 max_runtime=86400,score_type='high',balance_class=False,save_dir=None,model_name=None):
        
        MLOpt.__init__(self,X,y,'xgboost',params=params,params_cv=params_cv,
                       tune_range=tune_range,score_type=score_type,max_runtime=max_runtime,
                       max_rounds=max_rounds,running_rounds=running_rounds,balance_class=balance_class,
                       save_dir=save_dir,model_name=model_name)
    
    
    def get_n_estimators(self,model):
        """
        returns optimal number of estimators using CV on training set
        """
        xgb_param = model.get_xgb_params()
        xgb_param['eta'] = self._params['learning_rate']
        self._params['eta'] = self._params['learning_rate']
        
        if self.balance_class:
            xgb_train = xgb.DMatrix(self.X, label=self.y, weight=self.get_label_weights())
        else:
            xgb_train = xgb.DMatrix(self.X, label=self.y)
        
        kwargs_cv = {'num_boost_round':self.params['n_estimators'],
                     'nfold':self.params_cv['cv_folds'],
                     'early_stopping_rounds':self.params_cv['early_stopping_rounds'],
                     'stratified':self.params_cv['stratified']}
        
        try: # check if custom evalution function is specified
            if callable(self.params_cv['feval']):
                kwargs_cv['feval'] = self.params_cv['feval']
        except KeyError:
            kwargs_cv['metrics'] = self.params_cv['metrics']
        
        if self._greater_is_better:
            kwargs_cv['maximize'] = True
        else:
            kwargs_cv['maximize'] = False
        
        cvresult = xgb.cv(xgb_param,xgb_train,**kwargs_cv)
        self._params['n_estimators'] = int(cvresult.shape[0]/(1-1/self.params_cv['cv_folds']))
        return self
    
    
    def tune_params(self):
        """
        tune specified (and default) parameters
        """
        self._start_time = time.time()
        self.default_params() # set default parameters
        self.score_init() # set initial score
        iround = 0
        while iround<self.max_rounds:
            print('\nLearning rate for iteration %i: %f.' %(iround+1,self._params['learning_rate']))
            while self._step<5:
                istep_time = time.time()
                if self._step==0:
                    xgb = XGBClassifier(**self._params)
                    self.get_n_estimators(xgb)
                else:
                    self.apply_gridsearch(XGBClassifier(**self._params))
                self.print_progress(istep_time,iround=iround,max_rounds=self.max_rounds) # print params and performance
                self._step+=1
            
            # store model each iteration
            self._params_iround[iround] = {}
            for key,value in self._params.items():
                self._params_iround[iround][key] = value
            self._params_iround[iround]['model_score'] = self._temp_score
            
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
