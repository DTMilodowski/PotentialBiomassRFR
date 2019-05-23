import numpy as np
import sys
import xarray as xr #xarray to read all types of formats
from affine import Affine
import pickle

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

import useful
import set_training_areas
import cal_val as cv
import map_figures as mf

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from hyperopt import tpe, Trials, fmin, hp, STATUS_OK,space_eval
from hyperopt.pyll.base import scope
from functools import partial

country_code = sys.argv[1]#'WAFR'
version = sys.argv[2]#'002'
iterations = int(sys.argv[3])#5
training_sample_size = 250000
#load = sys.argv[4]#'new'

path2alg = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/saved_algorithms'
path2data = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
path2agb = path2data+'agb/'
path2calval = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/calval/'
path2output = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/output/'

"""
#===============================================================================
PART A: LOAD IN DATA, PROCESS AS REQUIRED AND SUBSET THE TRAINING DATA
#-------------------------------------------------------------------------------
"""
print('Loading data')
#pca = joblib.load('%s/%s_%s_pca_pipeline.pkl' % (path2alg,country_code,version))
pca = make_pipeline(StandardScaler(),PCA(n_components=0.999))

# load all predictors to generate preprocessing minmax scaler transformation
predictors_full,landmask = useful.get_predictors(country_code, training_subset=False)
#get the agb data
agb = xr.open_rasterio('%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code))[0]
yall = agb.values[landmask]

# Fit PCA transformation on predictor variables
pca.fit(predictors_full)
joblib.dump(pca,'%s/%s_%s_pca_pipeline.pkl' % (path2alg,country_code,version))

# First run of random forest regression model, with inital training set.
# Create the random forest object with predefined parameters
rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=20, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
           oob_score=True, random_state=None, verbose=0, warm_start=False)

# Get additional data masks
initial_training_mask = useful.get_mask(country_code,mask_def=5)

# get subset of predictors for initial training set
X = predictors_full[initial_training_mask[landmask]]
y = yall[initial_training_mask[landmask]]

"""
#===============================================================================
PART B: BAYESIAN HYPERPARAMETER OPTIMISATION
#-------------------------------------------------------------------------------
"""
print('Hyperparameter optimisation')
#split train and test subset, specifying random seed for reproducability
# due to processing limitations, we use only <500000 in the initial
# hyperparameter optimisation. Random seed used so that same split can be
# applied later for independent validations set
sss = StratifiedShuffleSplit(n_splits=1,train_size=0.75,test_size=0.25,random_state=2345)
idx_train, idx_test = sss.split(X,lc)
X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]
lc_train, lc_test = lc[idx_train], lc[idx_test]
# create pca transform for testing against RF performance
Xpca_train = pca.transform(X_train)

n_predictors = X_train.shape[1]
n_pca_predictors = Xpca_train.shape[1]

# set up hyperparameterspace for optimisation
rf = RandomForestRegressor(criterion="mse",bootstrap=True,n_jobs=-1)
space = hp.choice([
            {'preprocessing':'none',
             'params': { "max_depth":scope.int(hp.quniform("max_depth",20,500,1)),              # ***maximum number of branching levels within each tree
                    "max_features":scope.int(hp.quniform("max_features",int(n_predictors/5),n_predictors+1,1)),      # ***the maximum number of variables used in a given tree
                    "min_samples_leaf":scope.int(hp.quniform("min_samples_leaf",1,50,1)),    # ***The minimum number of samples required to be at a leaf node
                    "min_samples_split":scope.int(hp.quniform("min_samples_split",2,200,1)),  # ***The minimum number of samples required to split an internal node
                    "n_estimators":scope.int(hp.quniform("n_estimators",80,120,1)),          # ***Number of trees in the random forest
                    "min_impurity_decrease":hp.uniform("min_impurity_decrease",0.0,0.2),
                    "n_jobs":hp.choice("n_jobs",[20,20]) }
            },
            {'preprocessing':'pca',
             'params': { "max_depth":scope.int(hp.quniform("max_depth",20,500,1)),              # ***maximum number of branching levels within each tree
                    "max_features":scope.int(hp.quniform("max_features",int(n_pca_predictors/5),n_pca_predictors+1,1)),      # ***the maximum number of variables used in a given tree
                    "min_samples_leaf":scope.int(hp.quniform("min_samples_leaf",1,50,1)),    # ***The minimum number of samples required to be at a leaf node
                    "min_samples_split":scope.int(hp.quniform("min_samples_split",2,200,1)),  # ***The minimum number of samples required to split an internal node
                    "n_estimators":scope.int(hp.quniform("n_estimators",80,120,1)),          # ***Number of trees in the random forest
                    "min_impurity_decrease":hp.uniform("min_impurity_decrease",0.0,0.2),
                    "n_jobs":hp.choice("n_jobs",[20,20]) }
            }
        ])

# define a function to quantify the objective function
best = -np.inf
seed=0
def f(params):
    global best
    # print starting point
    if np.isfinite(best)==False:
        print('starting point:', params)

    # otherwise run the cross validation for this parameter set
    # - subsample from training set for this iteration
    sss_iter = StratifiedShuffleSplit(n_splits=1,train_size=training_sample_size,
                                            test_size=0,random_state=seed)
    seed+=1
    idx_iter, idx_temp = sss_iter.split(X_train,lc_train)
    y_iter = y[idx_iter]
    if space['preprocessing']=='pca':
        X_iter = Xpca_train[idx_iter]
    else:
        X_iter = X_train[idx_iter]

    # - set up random forest regressor

    rf_params=space['params']
    rf = RandomForestRegressor(**rf_params)
    # - apply cross validation procedure
    score = cross_val_score(rf, X_iter, y_iter, cv=5).mean()
    # - if error reduced, then update best model accordingly
    if score > best:
        best = score
        print('new best r^2: ', -best, params)
    return {'loss': -score, 'status': STATUS_OK}

# Set algoritm parameters
# - TPE
# - randomised search used to initialise (n_startup_jobs iterations)
# - percentage of hyperparameter combos identified as "good" (gamma)
# - number of sampled candidates to calculate expected improvement (n_EI_candidates)
trials=Trials()
algorithm = partial(tpe.suggest, n_startup_jobs=30, gamma=0.25, n_EI_candidates=24)
best = fmin(f, param_space, algo=algorithm, max_evals=130, trials=trials)
print('best:')
print(best)

# save trials for future reference
print('saving trials to file for future reference')
pickle.dump(trials, open('%s%s_%s_rf_sentinel_lidar_agb_trials.p' % (path2alg,site_id,version), "wb"))
