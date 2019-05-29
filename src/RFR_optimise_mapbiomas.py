import numpy as np
import sys
import xarray as xr #xarray to read all types of formats
import pickle

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

import useful
import cal_val as cv

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from hyperopt import tpe, rand, fmin, hp, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.base import scope
from functools import partial

country_code = 'BRA'
version = '007'
training_sample_size = 500000
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
mapbiomas = useful.load_mapbiomas(country_code)

#get the agb data
agb = xr.open_rasterio('%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code))[0]
yall = agb.values[landmask]

# Fit PCA transformation on predictor variables
pca.fit(predictors_full)
joblib.dump(pca,'%s/%s_%s_pca_pipeline.pkl' % (path2alg,country_code,version))

# Get additional data masks
initial_training_mask = useful.get_mask(country_code,mask_def=5)
initial_training_mask[mapbiomas==0]=0

# get subset of predictors for initial training set
X = predictors_full[initial_training_mask[landmask]]
y = yall[initial_training_mask[landmask]]
lc = mapbiomas[landmask][initial_training_mask[landmask]]

predictors_full = None
yall = None
agb = None

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
for idx_train, idx_test in sss.split(X,lc):
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    lc_train, lc_test = lc[idx_train], lc[idx_test]
# create pca transform for testing against RF performance
Xpca_train = pca.transform(X_train)

n_predictors = X_train.shape[1]
n_pca_predictors = Xpca_train.shape[1]

# set up hyperparameterspace for optimisation
rf = RandomForestRegressor(criterion="mse",bootstrap=True,n_jobs=-1,n_estimators=80)

#min_samples_split_iter = hp.quniform("min_samples_split",2,200,1)
#min_samples_leaf_iter = hp.quniform("min_samples_leaf",1,min_samples_split_iter,1)
"""
default_params = { "max_depth":scope.int(hp.quniform("max_depth",20,500,1)),              # ***maximum number of branching levels within each tree
                    "max_features":scope.int(hp.quniform("max_features",int(n_predictors/5),n_predictors,1)),      # ***the maximum number of variables used in a given tree
                    "min_samples_leaf":scope.int(hp.quniform("min_samples_leaf",1,50,1)),    # ***The minimum number of samples required to be at a leaf node
                    "min_samples_split":scope.int(hp.quniform("min_samples_split",2,200,1)),  # ***The minimum number of samples required to split an internal node
                    "n_estimators":scope.int(hp.quniform("n_estimators",80,120,1)),          # ***Number of trees in the random forest
                    "min_impurity_decrease":hp.uniform("min_impurity_decrease",0.0,0.2),
                    "n_jobs":hp.choice("n_jobs",[20,20]) }
"""
default_params = { "max_depth":scope.int(hp.quniform("max_depth",20,500,1)),              # ***maximum number of branching levels within each tree
                    "max_features":scope.int(hp.quniform("max_features",int(n_predictors/5),n_predictors,1)),      # ***the maximum number of variables used in a given tree
                    "min_samples_leaf":scope.int(hp.quniform("min_samples_leaf",1,50,1)),    # ***The minimum number of samples required to be at a leaf node
                    "min_samples_split":scope.int(hp.quniform("min_samples_split",2,200,1)),  # ***The minimum number of samples required to split an internal node
                    "n_estimators":scope.int(hp.quniform("n_estimators",80,120,1)),          # ***Number of trees in the random forest
                    "min_impurity_decrease":hp.uniform("min_impurity_decrease",0.0,0.2),
                    "n_jobs":hp.choice("n_jobs",[20,20]) }

pca_params =      { "max_depth":scope.int(hp.quniform("pca-max_depth",20,500,1)),              # ***maximum number of branching levels within each tree
                    "max_features":scope.int(hp.quniform("pca-max_features",int(n_pca_predictors/5),n_pca_predictors,1)),      # ***the maximum number of variables used in a given tree
                    "min_samples_leaf":scope.int(hp.quniform("pca-min_samples_leaf",1,50,1)),    # ***The minimum number of samples required to be at a leaf node
                    "min_samples_split":scope.int(hp.quniform("pca-min_samples_split",2,200,1)),  # ***The minimum number of samples required to split an internal node
                    "n_estimators":scope.int(hp.quniform("pca-n_estimators",60,80,1)),          # ***Number of trees in the random forest
                    "min_impurity_decrease":hp.uniform("pca-min_impurity_decrease",0.0,0.2),
                    "n_jobs":hp.choice("pca-n_jobs",[20,20]) }
"""
space = hp.choice('version',[{'preprocessing':'none',
                                    'params': default_params
                                    },
                                    {'preprocessing':'pca',
                                    'params':pca_params
                                    }])
"""
# define a function to quantify the objective function
def f(params):
    global best_score
    global seed
    global fail_count
    # check the hyperparameter set is sensible
    # - check 1: min_samples_split > min_samples_leaf
    if params['min_samples_split']<params['min_samples_leaf']:
        fail_count+=1
        print("INVALID HYPERPARAMETER SELECTION",params)
        return {'loss': None, 'status': STATUS_FAIL}

    # run the cross validation for this parameter set
    # - subsample from training set for this iteration
    sss_iter = StratifiedShuffleSplit(n_splits=1,train_size=training_sample_size,
                                    test_size=y_train.size-training_sample_size,
                                    random_state=seed)
    for idx_iter, idx_ignore in sss_iter.split(X_train,lc_train):
        y_iter = y_train[idx_iter]
        X_iter = X_train[idx_iter]
    #rf_params=space['preprocessing']['params']
    rf = RandomForestRegressor(**params)
    # - apply cross validation procedure
    score = cross_val_score(rf, X_iter, y_iter, cv=5).mean()
    # - if error reduced, then update best model accordingly
    if score > best_score:
        best_score = score
        print('new best r^2: ', -best_score, params)
    seed+=1
    return {'loss': -score, 'status': STATUS_OK}

# Set algoritm parameters
# - TPE
# - randomised search used to initialise (n_startup_jobs iterations)
# - percentage of hyperparameter combos identified as "good" (gamma)
# - number of sampled candidates to calculate expected improvement (n_EI_candidates)
trials=Trials()
#trials=pickle.load(open('%s/%s_%s_rf_hyperopt_trials' % (path2alg,country_code,version), "wb"))
max_evals_target = 120
spin_up_target = 60
best_score = -np.inf
seed=0
fail_count=0

# Start with randomised search - setting this explicitly to account for some
# iterations not being accepted
print("Starting randomised search (spin up)")
spin_up = spin_up_target+fail_count
best = fmin(f, default_params, algo=rand.suggest, max_evals=spin_up, trials=trials)
while (len(trials.trials)-fail_count)<spin_up_target:
    print('\tTarget: %i; iterations: %i; failures: %i', % (spin_up_target,len(trials.trials),fail_count))
    spin_up+=1
    best = fmin(f, default_params, algo=rand.suggest, max_evals=spin_up, trials=trials)

# Now do the TPE search
print("Starting TPE search")
max_evals = max_evals_target
algorithm = partial(tpe.suggest, n_startup_jobs=spin_up, gamma=0.25, n_EI_candidates=24)
best = fmin(f, default_params, algo=algorithm, max_evals=max_evals, trials=trials)
# Not every hyperparameter set will be accepted, so need to conitnue searching
# until the required number of evaluations is met
max_evals = max_evals_target+fail_count
while (len(trials.trials)-fail_count)<max_evals_target:
    print('\tTarget: %i; iterations: %i; failures: %i', % (max_evals_target,len(trials.trials),fail_count))
    max_evals+=1
    best = fmin(f, default_params, algo=algorithm, max_evals=max_evals, trials=trials)

print('\n\n%i iterations, from which %i failed' % (max_evals,fail_count))
print('best:')
print(best)

# save trials for future reference
print('saving trials to file for future reference')
pickle.dump(trials, open('%s/%s_%s_rf_hyperopt_trials.p' % (path2alg,country_code,version), "wb"))

# plot summary of optimisation runs
print('Basic plots summarising optimisation results')
parameters = ['n_estimators','max_depth', 'max_features', 'min_impurity_decrease','min_samples_leaf', 'min_samples_split']

trace = {}
trace['scores'] = np.zeros(max_evals)
trace['iteration'] = np.arange(max_evals)+1
for pp in parameters:
    trace[pp] = np.zeros(max_evals)

for ii,tt in enumerate(trials.trials):
     trace['scores'][ii] = -tt['result']['loss']
     for pp in parameters:
         trace[pp][ii] = tt['misc']['vals'][pp][0]

df = pd.DataFrame(data=trace)

# plot score for each hyperparameter value tested
fig2, axes = plt.subplots(nrows=3, ncols=2, figsize=(8,8))
cmap = sns.dark_palette('seagreen',as_cmap=True)
for i, val in enumerate(parameters):
    sns.scatterplot(x=val,y='scores',data=df,marker='.',hue='iteration',
                palette=cmap,edgecolor='none',legend=False,ax=axes[i//2,i%2])
    axes[i//2,i%2].set_xlabel(val)
    axes[i//2,i%2].set_ylabel('5-fold C-V score')
plt.tight_layout()
fig2.savefig('%s%s_%s_hyperpar_search_score.png' % (path2calval,country_code,version))

# Plot traces to see progression of hyperparameter selection
fig3, axes = plt.subplots(nrows=3, ncols=2, figsize=(8,8))
for i, val in enumerate(parameters):
    sns.scatterplot(x='iteration',y=val,data=df,marker='.',hue='scores',
                palette=cmap,edgecolor='none',legend=False,ax=axes[i//2,i%2])
    #axes[i//2,i%2].axvline(spin_up,':',color = 0.5)
    axes[i//2,i%2].set_title(val)
plt.tight_layout()
fig3.savefig('%s%s_%s_hyperpar_search_trace.png' % (path2calval,country_code,version))

"""
#===============================================================================
PART C: USE BEST HYPERPARAMETER VALUE FROM THE OPTIMISATION AND FIT NEW RF MODEL
USING THE FULL TRAINING SET AND VALIDATE AGAINST THE EXCLUDED TEST DATASET
Boost number of trees in the forest for this analysis, as only running once
#-------------------------------------------------------------------------------
"""
rf = RandomForestRegressor(bootstrap=True,
            criterion='mse',           # criteria used to choose split point at each node
            max_depth= trace['max_depth'][idx],            # ***maximum number of branching levels within each tree
            max_features=trace['max_features'][idx],       # ***the maximum number of variables used in a given tree
            max_leaf_nodes=None,       # the maximum number of leaf nodes per tree
            min_impurity_decrease=trace['min_impurity_decrease'][idx], # the miminum drop in the impurity of the clusters to justify splitting further
            min_impurity_split=None,   # threshold impurity within an internal node before it will be split
            min_samples_leaf=trace['min_samples_leaf'][idx],       # ***The minimum number of samples required to be at a leaf node
            min_samples_split=trace['min_samples_split'][idx],       # ***The minimum number of samples required to split an internal node
            n_estimators=200,#trace['n_estimators'],          # ***Number of trees in the random forest
            n_jobs=30,                 # The number of jobs to run in parallel for both fit and predict
            oob_score=True,            # use out-of-bag samples to estimate the R^2 on unseen data
            random_state=112358,         # seed used by the random number generator
            )

rf.fit(X_train,y_train)
y_train_rf = rf.predict(X_train)
cal_score = rf.score(X_train,y_train) # calculate coefficeint of determination R^2 of the calibration
print("Calibration R^2 = %.02f" % cal_score)

# fit the validation sample
y_test_rf = rf.predict(X_test)
val_score = rf.score(X_test,y_test)
print("Validation R^2 = %.02f" % val_score)
