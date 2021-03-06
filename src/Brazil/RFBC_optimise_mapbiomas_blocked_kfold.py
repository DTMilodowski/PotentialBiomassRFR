import numpy as np
import sys
import xarray as xr #xarray to read all types of formats
import pandas as pd
from scipy import stats
import pickle

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

sys.path.append('../')
sys.path/append('../random_forest/')
import useful
import cal_val as cv
import random_forest_functions as rff

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from scipy import ndimage as image

from hyperopt import tpe, rand, fmin, hp, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.base import scope
from functools import partial

country_code = 'BRA'
version = '014'

path2alg = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/saved_algorithms'
path2data = '/disk/scratch/local.2/PotentialBiomass/processed/%s/' % country_code
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
initial_training_mask = useful.get_mask(country_code,mask_def=10)
initial_training_mask[mapbiomas==0]=0
initial_training_mask[~landmask]=0
# get subset of predictors for initial training set
X = pca.transform(predictors_full[initial_training_mask[landmask]])
y = yall[initial_training_mask[landmask]]
lc = mapbiomas[landmask][initial_training_mask[landmask]]

# create a blocked sampling grid at ~1 degree resolution
raster_res = agb.attrs['res'][0]
block_res = 2
block_width = int(np.ceil(block_res/raster_res))
blocks_array = np.zeros(agb.values.shape)
block_label = 0
for rr,row in enumerate(np.arange(0,blocks_array.shape[0],block_width)):
    for cc, col in enumerate(np.arange(0,blocks_array.shape[1],block_width)):
        blocks_array[row:row+block_width,col:col+block_width]=block_label
        block_label+=1

# test blocks for training data presence
blocks = blocks_array[landmask][initial_training_mask[landmask]]
blocks_keep,blocks_count = np.unique(blocks,return_counts=True)
# remove blocks with no training data
blocks_array[~np.isin(blocks,blocks_keep)]=np.nan

predictors_full = None
yall = None
agb = None

"""
#===============================================================================
PART B: BAYESIAN HYPERPARAMETER OPTIMISATION
#-------------------------------------------------------------------------------
"""
print('Hyperparameter optimisation')
# create k-folds for hyperparameter optimisation (3 folds)
k = 3
# permute blocks randomly
cal_blocks_array = blocks_array.copy()
blocks_kfold = np.random.permutation(blocks_keep)
blocks_in_fold = int(np.ceil(blocks_kfold.size/k))
kfold_idx = np.zeros(y.size)
for ii in range(0,k):
    blocks_iter = blocks_kfold[ii*blocks_in_fold:(ii+1)*blocks_in_fold]
    print(np.isin(blocks,blocks_iter).sum())
    # label calibration blocks with fold
    cal_blocks_array[np.isin(blocks_array,blocks_iter)]=ii
    kfold_idx[np.isin(blocks,blocks_iter)]=ii

cal_blocks_array[~initial_training_mask] = np.nan
cal_blocks = cal_blocks_array[landmask][initial_training_mask[landmask]]

# now filter the blocks based on proximity to validation data to avoid
# neighbouring pixels biasing the validation
buffer_width = 0.5
buffer = int(np.ceil(buffer_width/raster_res))
val_blocks_array = cal_blocks_array.copy()
for ii in range(0,k):
    cal_data_mask = np.all((cal_blocks_array!=ii,initial_training_mask),axis=0)
    # expand neighbourhood with buffer
    cal_data_mask = image.binary_dilation(cal_data_mask,iterations=buffer)
    val_blocks_array[np.all((cal_data_mask,val_blocks_array==ii),axis=0)]=np.nan

val_blocks = val_blocks_array[landmask][initial_training_mask[landmask]]

# set up hyperparameterspace for optimisation
n_predictors = X.shape[1]
rf = RandomForestRegressor(criterion="mse",bootstrap=True,n_jobs=-1,n_estimators=80)
param_space = { "max_depth":scope.int(hp.quniform("max_depth",10,500,1)),              # ***maximum number of branching levels within each tree
                    "max_features":scope.int(hp.quniform("max_features",int(n_predictors/8),n_predictors,1)),      # ***the maximum number of variables used in a given tree
                    "min_samples_leaf":scope.int(hp.quniform("min_samples_leaf",2,150,1)),    # ***The minimum number of samples required to be at a leaf node
                    "min_samples_split":scope.int(hp.quniform("min_samples_split",3,500,1)),  # ***The minimum number of samples required to split an internal node
                    "n_estimators":scope.int(hp.quniform("n_estimators",100,150,1)),          # ***Number of trees in the random forest
                    "n_jobs":hp.choice("n_jobs",[30,30]),
                    "oob_score":hp.choice("oob_score",[True,True]) }

# define a function to quantify the objective function
def f(params):
    global best_mse
    global fail_count
    # check the hyperparameter set is sensible
    # - check 1: min_samples_split > min_samples_leaf
    if params['min_samples_split']<params['min_samples_leaf']:
        fail_count+=1
        #print("INVALID HYPERPARAMETER SELECTION",params)
        return {'loss': None, 'status': STATUS_FAIL}

    rf = RandomForestRegressor(**params)
    r2_scores = np.zeros(k)
    MSE_scores = np.zeros(k)
    for kk in range(k):
        train_mask = cal_blocks!=kk
        test_mask = val_blocks==kk

        rf1,rf2 = rff.rfbc_fit(rf,X[train_mask],y[train_mask])
        y_rf = rff.rfbc_predict(rf1,rf2,X[test_mask])

        temp1,temp2,r,temp3,temp4 = stats.linregress(y[test_mask],y_rf)
        r2_scores[kk] = r**2
        MSE_scores[kk] = np.mean( (y[test_mask]-y_rf) **2 )
    r2_score=r2_scores.mean()
    mse=MSE_scores.mean()
    # - if error reduced, then update best model accordingly
    if mse < best_mse:
        best_score = score
        print('new best r^2: ', r2_score, '; best RMSE: ', np.sqrt(mse), params)
    return {'loss': mse, 'status': STATUS_OK}

# Set algoritm parameters
# - TPE
# - randomised search used to initialise (n_startup_jobs iterations)
# - percentage of hyperparameter combos identified as "good" (gamma)
# - number of sampled candidates to calculate expected improvement (n_EI_candidates)
trials=Trials()
#trials=pickle.load(open('%s/%s_%s_rf_hyperopt_trials.p' % (path2alg,country_code,version), "rb"))
max_evals_target = 200
spin_up_target = 50
best_mse = np.inf
fail_count=0

# Start with randomised search - setting this explicitly to account for some
# iterations not being accepted
print("Starting randomised search (spin up)")
best = fmin(f, param_space, algo=rand.suggest, max_evals=spin_up_target, trials=trials)
spin_up = spin_up_target+fail_count
while (len(trials.trials)-fail_count)<spin_up_target:
    print('\tTarget: %i; iterations: %i; failures: %i' % (spin_up_target,len(trials.trials),fail_count))
    spin_up+=1
    best = fmin(f, param_space, algo=rand.suggest, max_evals=spin_up, trials=trials)

# Now do the TPE search
print("Starting TPE search")
max_evals = max_evals_target+fail_count
algorithm = partial(tpe.suggest, n_startup_jobs=spin_up, gamma=0.15, n_EI_candidates=80)
best = fmin(f, param_space, algo=algorithm, max_evals=max_evals, trials=trials)
# Not every hyperparameter set will be accepted, so need to conitnue searching
# until the required number of evaluations is met
max_evals = max_evals_target+fail_count
while (len(trials.trials)-fail_count)<max_evals_target:
    print('\tTarget: %i; iterations: %i; failures: %i' % (max_evals_target,len(trials.trials),fail_count))
    max_evals+=1
    best = fmin(f, param_space, algo=algorithm, max_evals=max_evals, trials=trials)

# Now repeat TPE search for another 50 iterations with a refined search window
max_evals_target+=100
max_evals = max_evals_target+fail_count
algorithm = partial(tpe.suggest, n_startup_jobs=spin_up, gamma=0.25, n_EI_candidates=32)
while (len(trials.trials)-fail_count)<max_evals_target:
    print('\tTarget: %i; iterations: %i; failures: %i' % (max_evals_target,len(trials.trials),fail_count))
    max_evals+=1
    best = fmin(f, param_space, algo=algorithm, max_evals=max_evals, trials=trials)

print('\n\n%i iterations, from which %i failed' % (max_evals,fail_count))
print('best:')
print(best)

# save trials for future reference
print('saving trials to file for future reference')
pickle.dump(trials, open('%s/%s_%s_rfbc_hyperopt_trials.p' % (path2alg,country_code,version), "wb"))
# trials = pickle.load(open('%s/%s_%s_rf_hyperopt_trials.p' % (path2alg,country_code,version), "rb"))

# plot summary of optimisation runs
print('Basic plots summarising optimisation results')
parameters = ['n_estimators','max_depth', 'max_features', 'min_samples_leaf', 'min_samples_split']

trace = {}
trace['scores'] = np.zeros(max_evals_target)
trace['RMSE'] = np.zeros(max_evals_target)
trace['iteration'] = np.arange(max_evals_target)+1
for pp in parameters:
    trace[pp] = np.zeros(max_evals_target)
ii=0
for tt in trials.trials:
    if ii < max_evals_target:
        if tt['result']['status']=='ok':
            trace['scores'][ii] = tt['result']['loss']
            trace['RMSE'][ii] = np.sqrt(tt['result']['loss'])
            for pp in parameters:
                trace[pp][ii] = tt['misc']['vals'][pp][0]
            ii+=1

df = pd.DataFrame(data=trace)

# plot score for each hyperparameter value tested
fig2, axes = plt.subplots(nrows=3, ncols=2, figsize=(8,8))
cmap = sns.light_palette('seagreen',as_cmap=True)
for i, val in enumerate(parameters):
    sns.scatterplot(x=val,y='RMSE',data=df,marker='o',hue='iteration',palette=cmap,
                legend=False,ax=axes[i//2,i%2])
    axes[i//2,i%2].set_xlabel(val)
    axes[i//2,i%2].set_ylabel('RMSE')
# Finally plot trace of RMSE through the optimisation process
sns.scatterplot(x='iteration',y='RMSE',data=df,marker='o',hue='iteration',palette=cmap,
                legend=False,ax=axes[2,1])
axes[2,1].set_xlabel('iteration')
axes[2,1].set_ylabel('RMSE')
plt.tight_layout()
fig2.savefig('%s%s_%s_rfbc_hyperpar_search_score.png' % (path2calval,country_code,version))

# Plot traces to see progression of hyperparameter selection
fig3, axes = plt.subplots(nrows=3, ncols=2, figsize=(8,8))
for i, val in enumerate(parameters):
    sns.scatterplot(x='iteration',y=val,data=df,marker='o',hue='RMSE',palette=cmap,
                legend=False,ax=axes[i//2,i%2])
    #axes[i//2,i%2].axvline(spin_up,':',color = 0.5)
    axes[i//2,i%2].set_title(val)
# Finally plot trace of RMSE through the optimisation process
sns.scatterplot(x='iteration',y='RMSE',data=df,marker='o',hue='RMSE',palette=cmap,
                legend=False,ax=axes[2,1])
axes[2,1].set_xlabel('iteration')
axes[2,1].set_ylabel('RMSE')
plt.tight_layout()
fig3.savefig('%s%s_%s_rfbc_hyperpar_search_trace.png' % (path2calval,country_code,version))
