import numpy as np
import sys
import xarray as xr #xarray to read all types of formats
from affine import Affine
import useful
import set_training_areas
import cal_val as cv
import map_figures as mf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import pickle

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

country_code = 'BRA'
version = '013'
iterations = 7

path2alg = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/saved_algorithms'
path2data = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
path2agb = path2data+'agb/'
path2calval = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/calval/'
path2output = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/output/'

pca = joblib.load('%s/%s_%s_pca_pipeline.pkl' % (path2alg,country_code,version))
#pca = make_pipeline(StandardScaler(),PCA(n_components=0.999))

"""
#===============================================================================
PART A: LOAD IN DATA, PROCESS AS REQUIRED AND SUBSET THE TRAINING DATA
#-------------------------------------------------------------------------------
"""
# load all predictors to generate preprocessing minmax scaler transformation
predictors_full,landmask = useful.get_predictors(country_code, training_subset=False)
#get the agb data
agb = xr.open_rasterio('%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code))[0]

# Get additional data masks
initial_training_mask = useful.get_mask(country_code,mask_def=8)
#initial_training_mask = useful.get_mask(country_code,mask_def=9) # this one refines non-forest ares based on protected areas
other_stable_forest_mask = useful.get_mask(country_code,mask_def=7)

# Run PCA transformation on predictor variables
# fit PCA
pca.fit(predictors_full)
#joblib.dump(pca,'%s/%s_%s_pca_pipeline.pkl' % (path2alg,country_code,version))
#Xall = predictors_full
Xall = pca.transform(predictors_full)
yall = agb.values[landmask]

# get subset of predictors for initial training set
X = Xall[initial_training_mask[landmask]]
y = yall[initial_training_mask[landmask]]

# Get additional stable forest areas
Xtest = Xall[other_stable_forest_mask[landmask]]
ytest = yall[other_stable_forest_mask[landmask]]

"""
#===============================================================================
PART B: FIT FIRST RANDOM FOREST MODEL
#-------------------------------------------------------------------------------
"""
# Load trials data from optimisation and retrieve best hyperparameter combination
# but boost number of trees in forest as not running as many times, so can
# afford computational expense
trials = pickle.load(open('%s/%s_%s_rf_hyperopt_trials.p' % (path2alg,country_code,version), "rb"))
parameters = ['n_estimators','max_depth', 'max_features','min_samples_leaf', 'min_samples_split']
trace = {}
n_trials = len(trials)
trace['scores'] = np.zeros(n_trials)
for pp in parameters:
    trace[pp] = np.zeros(n_trials)
for ii,tt in enumerate(trials.trials):
    if tt['result']['status']=='ok':
        trace['scores'][ii] = -tt['result']['loss']
        for pp in parameters:
            trace[pp][ii] = tt['misc']['vals'][pp][0]
    else:
        trace['scores'][ii] = np.nan
        for pp in parameters:
            trace[pp][ii] = np.nan
mask = np.isfinite(trace['scores'])
for key in trace.keys():
    trace[key]=trace[key][mask]

idx = np.argsort(trace['scores'])[-1]
rf = RandomForestRegressor(bootstrap=True,
            criterion='mse',           # criteria used to choose split point at each node
            max_depth= int(trace['max_depth'][idx]),            # ***maximum number of branching levels within each tree
            max_features=int(trace['max_features'][idx]),       # ***the maximum number of variables used in a given tree
            max_leaf_nodes=None,       # the maximum number of leaf nodes per tree
            min_impurity_decrease=0,#trace['min_impurity_decrease'][idx], # the miminum drop in the impurity of the clusters to justify splitting further
            min_impurity_split=None,   # threshold impurity within an internal node before it will be split
            min_samples_leaf=int(trace['min_samples_leaf'][idx]),       # ***The minimum number of samples required to be at a leaf node
            min_samples_split=int(trace['min_samples_split'][idx]),       # ***The minimum number of samples required to split an internal node
            n_estimators=200,#trace['n_estimators'],          # ***Number of trees in the random forest
            n_jobs=10,                 # The number of jobs to run in parallel for both fit and predict
            oob_score=True,            # use out-of-bag samples to estimate the R^2 on unseen data
            random_state=112358,         # seed used by the random number generator
            )

# First run of random forest regression model, with inital training set.
# Create the random forest object with predefined parameters
rf.fit(X,y)

"""
#===============================================================================
PART B: ITERATIVE AUGMENTATION OF TRAINING SET USED TO CALIBRATE RANDOM FOREST
MODEL
#-------------------------------------------------------------------------------
"""
# now iterate, filtering out other stable forest pixels for which the observed biomass
# is not within error of the predicted potential biomass
AGBpot, training_set, rf = useful.iterative_augmentation_of_training_set_obs_vs_pot_v3(ytest,
                                            y, Xtest, X, Xall, iterations,
                                            landmask, initial_training_mask,
                                            other_stable_forest_mask, rf,
                                            percentile_cutoff = 10)
iterations = AGBpot.shape[0]

# Save rf model for future reference
joblib.dump(rf,'%s/%s_%s_rf_iterative.pkl' % (path2alg,country_code,
                                                version))

# convert training set and AGBpot to xdarray for easy plotting and export to
# netcdf
# first deal with metadata and coordinates
tr = agb.attrs['transform']
transform = Affine(tr[0],tr[1],tr[2],tr[3],tr[4],tr[5])
nx, ny = agb.sizes['x'], agb.sizes['y']
col,row = np.meshgrid(np.arange(nx)+0.5, np.arange(ny)+0.5)
lon, lat = transform * (col,row)

coords = {'lat': (['lat'],lat[:,0],{'units':'degrees_north','long_name':'latitude'}),
          'lon': (['lon'],lon[0,:],{'units':'degrees_east','long_name':'longitude'})}

attrs_1={'_FillValue':-9999.,'units':'Mg ha-1'}
attrs_2={'_FillValue':-9999.,'units':'None'}
data_vars = {}

data_vars['AGBobs'] = (['lat','lon'],np.zeros([ny,nx])*np.nan,attrs_1)
for ii in range(0,iterations):
    key = 'AGBpot%i' % (ii+1)
    data_vars[key] = (['lat','lon'],np.zeros([ny,nx])*np.nan,attrs_1)
    key = 'trainset%i' % (ii+1)
    data_vars[key] = (['lat','lon'],np.zeros([ny,nx])*np.nan,attrs_2)

agb_rf = xr.Dataset(data_vars=data_vars,coords=coords)
agb_rf.AGBobs.values[landmask]  = agb.values[landmask]
for ii in range(0,iterations):
    key = 'AGBpot%i' % (ii+1)
    agb_rf[key].values[landmask]= AGBpot[ii][landmask]
    key = 'trainset%i' % (ii+1)
    agb_rf[key].values[landmask] = training_set[ii][landmask]

#save to a nc file
comp = dict(zlib=True, complevel=1)
encoding = {var: comp for var in agb_rf.data_vars}
nc_file = '%s%s_%s_AGB_potential_RFR_worldclim_soilgrids.nc' % (path2output,
                                country_code,version)
agb_rf.to_netcdf(path=nc_file)#,encoding=encoding)

# plot stuff
# Initial cal-val plot
mf.plot_AGBpot_iterations(agb_rf,iterations,country_code,version,path2output = path2output)
mf.plot_training_residuals(agb_rf,iterations,country_code,version,path2output=path2output,vmin=[0,0,-50],vmax=[200,200,50])
mf.plot_training_areas_iterative(agb_rf,iterations,country_code,version,path2output = path2output)
mf.plot_AGB_AGBpot_training(agb_rf,iterations,country_code,version,path2output = path2output)

training_mask_final = (training_set[3]>0)*landmask

X_train, X_test, y_train, y_test = train_test_split(Xall[training_mask_final[landmask]],agb.values[training_mask_final],test_size = 0.25, random_state=29)
rf.fit(X_train,y_train)
y_train_predict = rf.predict(X_train)
y_test_predict = rf.predict(X_test)
cal_r2,val_r2 = cv.cal_val_train_test_post_fit(y_train, y_train_predict, y_test,
                                    y_test_predict, path2calval, country_code,
                                    version, hue_var='density_50')


# Test bias correction
rf.fit(X_train,y_train)
y_oob = rf.oob_prediction_
# New target variable = y_oob - residual
# Note that this is more biased than the RF estimate
y_new = 2*y_oob-y_train
# Fit second random forest regression to predict y_new
rf2 = RandomForestRegressor(bootstrap=True,
            criterion='mse',           # criteria used to choose split point at each node
            max_depth= int(trace['max_depth'][idx]),            # ***maximum number of branching levels within each tree
            max_features=int(trace['max_features'][idx]),       # ***the maximum number of variables used in a given tree
            max_leaf_nodes=None,       # the maximum number of leaf nodes per tree
            min_impurity_decrease=trace['min_impurity_decrease'][idx], # the miminum drop in the impurity of the clusters to justify splitting further
            min_impurity_split=0,   # threshold impurity within an internal node before it will be split
            min_samples_leaf=int(trace['min_samples_leaf'][idx]),       # ***The minimum number of samples required to be at a leaf node
            min_samples_split=int(trace['min_samples_split'][idx]),       # ***The minimum number of samples required to split an internal node
            n_estimators=200,#trace['n_estimators'],          # ***Number of trees in the random forest
            n_jobs=10,                 # The number of jobs to run in parallel for both fit and predict
            oob_score=True,            # use out-of-bag samples to estimate the R^2 on unseen data
            )

rf2.fit(X_train,y_oob)

# Make prediction including bias correction
y_hat_train = 2*rf.predict(X_train)-rf2.predict(X_train)
y_hat_test = 2*rf.predict(X_test)-rf2.predict(X_test)
y_hat_test[y_hat_test<0]=0
y_hat_train[y_hat_train<0]=0
cal_r2,val_r2 = cv.cal_val_train_test_post_fit(y_train, y_hat_train, y_test,
                                    y_hat_test, path2calval, country_code,
                                    version, hue_var='density_50')
