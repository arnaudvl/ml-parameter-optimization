# ml-parameter-optimization
Hyperparameter optimization for machine learning algorithms.

## Getting started

### Prerequisites

Make sure you have up-to-date versions installed of:

  - lightgbm
  - numpy
  - pandas
  - scikit-learn
  - scipy
  - xgboost

### Installation

Clone the repository in your local workspace:

```
git clone https://github.com/arnaudvl/ml-parameter-optimization
```

## Functionality

There are 3 modules in mlopt that can be used for hyperparameter tuning: lgb_tune, sklearn_tune and xgb_tune.

sklearn_tune covers the adaboost, k-nearest neighbour, logistic regression, random forest and support vecor machine algorithms. Calling the function tune_params starts the tuning process using gridsearch.

The lightgbm (lgb_tune) and xgboost (xgb_tune) algorithms cannot efficiently be tuned using gridsearch given the large amount of hyperparameters. As a result, the parameters are tuned iteritavely. See the example for a full explanation.
