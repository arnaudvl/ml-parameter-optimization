from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import time

from mlopt.ml_tune import MLOpt

class AdaBoostClassifierOpt(MLOpt):
    """
    AdaBoost optimizer.
    
    The function "tune_params" will optimize by default the paramaters:
        - n_estimators (number of boosting rounds)
        - learning_rate
    """
    
    def __init__(self,X,y,params={},params_tree={},params_cv={},tune_range={},score_type='high',
                 save_dir=None,model_name=None):
        
        MLOpt.__init__(self,X,y,'adaboost',params=params,params_cv=params_cv,
                       tune_range=tune_range,score_type=score_type,save_dir=save_dir,model_name=model_name)
        
        self.params_tree = params_tree
        
        self._params_ada_tree = None
    
    def tune_params(self):
        """
        tune specified (and default) parameters
        """
        self._start_time = time.time()
        self.default_params() # set default parameters
        self.score_init() # set initial score
        self._params_ada_tree = self.set_default(self.params_tree,self._params_default_ada_tree)
        tree = DecisionTreeClassifier(**self._params_ada_tree) # define tree classifier
        self._params['base_estimator'] = tree
        adaboost = AdaBoostClassifier(**self._params)
        self.apply_gridsearch(adaboost)
        self.print_progress(self._start_time)
        return self


class KNNOpt(MLOpt):
    """
    K-Nearest Neighbours optimizer.
    
    The function "tune_params" will optimize by default the paramaters:
        - n_neighbors (called as 'knn__n_neighbors' for pipeline)
    """
    
    def __init__(self,X,y,params={},params_cv={},tune_range={},score_type='high',dim_reduction=None,
                 save_dir=None,model_name=None):
        
        MLOpt.__init__(self,X,y,'knn',params=params,params_cv=params_cv,
                       tune_range=tune_range,score_type=score_type,dim_reduction=dim_reduction,
                       save_dir=save_dir,model_name=model_name)
        
    def tune_params(self):
        """
        tune specified (and default) parameters
        """
        self._start_time = time.time()
        self.default_params() # set default parameters
        self.score_init() # set initial score
        if self.dim_reduction is not None:
            knn = Pipeline([('dimred',self.dim_reduction_method())
                            ('knn',KNeighborsClassifier(**self._params))])
            self._pipeline = True
        else:
            knn = KNeighborsClassifier(**self._params)
        self.apply_gridsearch(knn)
        self.print_progress(self._start_time)
        return self


class LogisticRegressionOpt(MLOpt):
    """
    Logistic Regression optimizer.
    
    The function "tune_params" will optimize by default the paramaters:
        - C (called as 'lr__C' for pipeline)
    """
    
    def __init__(self,X,y,params={},params_cv={},tune_range={},score_type='high',dim_reduction=None,
                 save_dir=None,model_name=None):
        
        MLOpt.__init__(self,X,y,'logistic-regression',params=params,params_cv=params_cv,
                       tune_range=tune_range,score_type=score_type,dim_reduction=dim_reduction,
                       save_dir=save_dir,model_name=model_name)
    
    def tune_params(self):
        """
        tune specified (and default) parameters
        """
        self._start_time = time.time()
        self.default_params() # set default parameters
        self.score_init() # set initial score
        if self.dim_reduction is not None:
            lr = Pipeline([('dimred',self.dim_reduction_method())
                           ('lr',LogisticRegression(**self._params))])
            self._pipeline = True
        else:
            lr = LogisticRegression(**self._params)
        self.apply_gridsearch(lr)
        self.print_progress(self._start_time)
        return self


class RandomForestClassifierOpt(MLOpt):
    """
    Random Forest optimizer.
    
    The function "tune_params" will optimize by default the paramaters:
        - n_estimators (number of trees)
        - max_features (fraction of features used for each node split)
        - min_samples_leaf (min amount of samples at the end of each leaf as fraction of samples)
    """
    
    def __init__(self,X,y,params={},params_cv={},tune_range={},score_type='high',save_dir=None,model_name=None):
        
        MLOpt.__init__(self,X,y,'random-forest',params=params,params_cv=params_cv,
                       tune_range=tune_range,score_type=score_type,save_dir=save_dir,model_name=model_name)
    
    def tune_params(self):
        """
        tune specified (and default) parameters
        """
        self._start_time = time.time()
        self.default_params() # set default parameters
        self.score_init() # set initial score
        rf = RandomForestClassifier(**self._params)
        self.apply_gridsearch(rf)
        self.print_progress(self._start_time)
        return self


class SVCOpt(MLOpt):
    """
    Support Vector Classification optimizer.
    
    The function "tune_params" will optimize by default the paramaters:
        - C (called as 'svc__C' for pipeline)
        - gamma (called as 'svc__gamma' for pipeline)
    """
    
    def __init__(self,X,y,params={},params_cv={},tune_range={},score_type='high',dim_reduction=None,
                 save_dir=None,model_name=None):
        
        MLOpt.__init__(self,X,y,'svc',params=params,params_cv=params_cv,
                       tune_range=tune_range,score_type=score_type,dim_reduction=dim_reduction,
                       save_dir=save_dir,model_name=model_name)
    
    def tune_params(self):
        """
        tune specified (and default) parameters
        """
        self._start_time = time.time()
        self.default_params() # set default parameters
        self.score_init() # set initial score
        if self.dim_reduction is not None:
            svc = Pipeline([('dimred',self.dim_reduction_method())
                            ('svc',SVC(**self._params))])
            self._pipeline = True
        else:
            svc = SVC(**self._params)
        self.apply_gridsearch(svc)
        self.print_progress(self._start_time)
        return self
