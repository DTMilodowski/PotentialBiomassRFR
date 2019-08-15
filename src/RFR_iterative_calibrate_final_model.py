"""
#===============================================================================
# RFR_iterative: calibrate final model
#-------------------------------------------------------------------------------
# - After selecting the iteration of choice from the training runs, fit final
#   model for expectation value, min and max, based on the uncertainty range in
#   the training data.
# - Save output to netcdf
#===============================================================================
"""
import numpy as np
import sys
import xarray as xr
import fiona
import rasterio
import rasterio.mask
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

country_code = 'EAFR'
version = '001'
iteration_to_use = 3
clip_to_boundary = True
country = 'Kenya'

path2alg = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/saved_algorithms'
path2data = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
path2agb = path2data+'agb/'
path2calval = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/calval/'
path2output = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/output/'

hyperopt_trials = '%s/%s_%s_rf_hyperopt_trials.p' % (path2alg,'BRA','010')
boundaries_shp = '/home/dmilodow/DataStore_DTM/EOlaboratory/Areas/ne_50m_admin_0_tropical_countries_small_islands_removed.shp'

pca = joblib.load('%s/%s_%s_pca_pipeline.pkl' % (path2alg,country_code,version))

"""
#===============================================================================
PART A: LOAD IN DATA, PROCESS AS REQUIRED AND SUBSET THE TRAINING DATA
#-------------------------------------------------------------------------------
"""
print("PART A: LOADING AND PREPROCESSING DATA")
# load all predictors to generate preprocessing minmax scaler transformation
predictors_full,landmask = useful.get_predictors(country_code, training_subset=False)
# training set from intermediate netcdf file
AGBpot_ds = xr.open_dataset('%s%s_%s_AGB_potential_RFR_worldclim_soilgrids.nc' %
                                (path2output, country_code,version))
key = 'trainset%i' % iteration_to_use
training_mask = AGBpot_ds[key].values>0*landmask

# load the AGB data and the uncertainty
agb = xr.open_rasterio('%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code))[0]
agb_unc = xr.open_rasterio('%sAvitabile_AGB_Uncertainty_%s_1km.tif' % (path2agb,country_code))[0]

yall = agb.values[landmask]
yunc = agb_unc.values[landmask]

# get subset of predictors for initial training set
Xall = pca.transform(predictors_full)
X = Xall[training_mask[landmask]]
y = yall[training_mask[landmask]]
ymin = (yall-yunc)[training_mask[landmask]]
ymax = (yall+yunc)[training_mask[landmask]]

"""
#===============================================================================
PART B: FIT RANDOM FOREST MODELS
#-------------------------------------------------------------------------------
"""
print("PART B: Fitting random forest models")
# Load trials data from optimisation and retrieve best hyperparameter combination
# but boost number of trees in forest as not running as many times, so can
# afford computational expense
trials = pickle.load(open(hyperopt_trials, "rb"))
parameters = ['n_estimators','max_depth', 'max_features', 'min_impurity_decrease','min_samples_leaf', 'min_samples_split']
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
            min_impurity_decrease=trace['min_impurity_decrease'][idx], # the miminum drop in the impurity of the clusters to justify splitting further
            min_impurity_split=None,   # threshold impurity within an internal node before it will be split
            min_samples_leaf=int(trace['min_samples_leaf'][idx]),       # ***The minimum number of samples required to be at a leaf node
            min_samples_split=int(trace['min_samples_split'][idx]),       # ***The minimum number of samples required to split an internal node
            n_estimators=int(trace['n_estimators'][idx]),#trace['n_estimators'],          # ***Number of trees in the random forest
            n_jobs=10,                 # The number of jobs to run in parallel for both fit and predict
            oob_score=True,            # use out-of-bag samples to estimate the R^2 on unseen data
            random_state=112358,         # seed used by the random number generator
            )
rf_max = RandomForestRegressor(bootstrap=True,
            criterion='mse',           # criteria used to choose split point at each node
            max_depth= int(trace['max_depth'][idx]),            # ***maximum number of branching levels within each tree
            max_features=int(trace['max_features'][idx]),       # ***the maximum number of variables used in a given tree
            max_leaf_nodes=None,       # the maximum number of leaf nodes per tree
            min_impurity_decrease=trace['min_impurity_decrease'][idx], # the miminum drop in the impurity of the clusters to justify splitting further
            min_impurity_split=None,   # threshold impurity within an internal node before it will be split
            min_samples_leaf=int(trace['min_samples_leaf'][idx]),       # ***The minimum number of samples required to be at a leaf node
            min_samples_split=int(trace['min_samples_split'][idx]),       # ***The minimum number of samples required to split an internal node
            n_estimators=int(trace['n_estimators'][idx]),#trace['n_estimators'],          # ***Number of trees in the random forest
            n_jobs=10,                 # The number of jobs to run in parallel for both fit and predict
            oob_score=True,            # use out-of-bag samples to estimate the R^2 on unseen data
            random_state=112358,         # seed used by the random number generator
            )
rf_min = RandomForestRegressor(bootstrap=True,
            criterion='mse',           # criteria used to choose split point at each node
            max_depth= int(trace['max_depth'][idx]),            # ***maximum number of branching levels within each tree
            max_features=int(trace['max_features'][idx]),       # ***the maximum number of variables used in a given tree
            max_leaf_nodes=None,       # the maximum number of leaf nodes per tree
            min_impurity_decrease=trace['min_impurity_decrease'][idx], # the miminum drop in the impurity of the clusters to justify splitting further
            min_impurity_split=None,   # threshold impurity within an internal node before it will be split
            min_samples_leaf=int(trace['min_samples_leaf'][idx]),       # ***The minimum number of samples required to be at a leaf node
            min_samples_split=int(trace['min_samples_split'][idx]),       # ***The minimum number of samples required to split an internal node
            n_estimators=int(trace['n_estimators'][idx]),#trace['n_estimators'],          # ***Number of trees in the random forest
            n_jobs=10,                 # The number of jobs to run in parallel for both fit and predict
            oob_score=True,            # use out-of-bag samples to estimate the R^2 on unseen data
            random_state=112358,         # seed used by the random number generator
            )

# Fit the random forest model with the mean, min and max
rf.fit(X,y)
rf_min.fit(X,ymin)
rf_max.fit(X,ymax)

# Save rf models for future reference
joblib.dump(rf,'%s/%s_%s_rf_iterative_mean.pkl' % (path2alg,country_code,
                                                version))
joblib.dump(rf_min,'%s/%s_%s_rf_iterative_min.pkl' % (path2alg,country_code,
                                                version))
joblib.dump(rf_max,'%s/%s_%s_rf_iterative_max.pkl' % (path2alg,country_code,
                                                version))

"""
#===============================================================================
PART C: PRODUCE FULL POTENTIAL BIOMASS MAPS AND SAVE TO NETCDF
#-------------------------------------------------------------------------------
"""
print("PART C: Produce full maps and uncertainty bounds")
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
data_vars['AGBobs_max'] = (['lat','lon'],np.zeros([ny,nx])*np.nan,attrs_1)
data_vars['AGBobs_min'] = (['lat','lon'],np.zeros([ny,nx])*np.nan,attrs_1)
data_vars['AGBpot'] = (['lat','lon'],np.zeros([ny,nx])*np.nan,attrs_1)
data_vars['AGBpot_max'] = (['lat','lon'],np.zeros([ny,nx])*np.nan,attrs_1)
data_vars['AGBpot_min'] = (['lat','lon'],np.zeros([ny,nx])*np.nan,attrs_1)
data_vars['training']= (['lat','lon'],np.zeros([ny,nx])*np.nan,attrs_2)

agb_rf = xr.Dataset(data_vars=data_vars,coords=coords)

agb_rf.AGBobs.values[landmask]  = yall
agb_rf.AGBobs_min.values[landmask]  = yall-yunc
agb_rf.AGBobs_max.values[landmask]  = yall+yunc

agb_rf.AGBpot.values[landmask]  = rf.predict(Xall)
agb_rf.AGBpot_min.values[landmask]  = rf_min.predict(Xall)
agb_rf.AGBpot_max.values[landmask]  = rf_max.predict(Xall)

agb_rf.training.values[landmask]  = training_mask[landmask]

#save to a nc file
comp = dict(zlib=True, complevel=1)
encoding = {var: comp for var in agb_rf.data_vars}
nc_file = '%s%s_%s_AGB_potential_RFR_worldclim_soilgrids_final.nc' % (path2output,
                                country_code,version)
agb_rf.to_netcdf(path=nc_file)#,encoding=encoding)

"""
#===============================================================================
PART D: PLOTTING
#-------------------------------------------------------------------------------
"""
print("PART D: Plot figures")
# plot stuff
# Initial cal-val plot
if clip_to_boundary:
    # load template raster
    template = rasterio.open('%s/Avitabile_AGB_%s_1km.tif' % (path2agb,country_code))
    mask = np.zeros(template.shape)
    # - load shapefile
    boundaries = fiona.open(boundaries_shp)
    # - for country of interest, make mask
    for feat in boundaries:
        name = feat['properties']['admin']
        if name==country:
            image,transform = rasterio.mask.mask(template,[feat['geometry']],crop=False)
            mask[image[0]>=0]=1

    agb_rf.AGBobs.values[mask==0]  = np.nan
    agb_rf.AGBobs_min.values[mask==0]  = np.nan
    agb_rf.AGBobs_max.values[mask==0]  = np.nan

    agb_rf.AGBpot.values[mask==0]  = np.nan
    agb_rf.AGBpot_min.values[mask==0]  = np.nan
    agb_rf.AGBpot_max.values[mask==0]  = np.nan

    mf.plot_AGB_AGBpot_training_final(agb_rf,country,version,path2output = path2output,
                            clip=True,mask=mask)
    mf.plot_AGBpot_uncertainty(agb_rf,country,version,path2output = path2output,
                            clip=True,mask=mask)
    mf.plot_AGBseq_final(agb_rf,country,version,path2output = path2output,
                            clip=True,mask=mask)

else:
    mf.plot_AGB_AGBpot_training_final(agb_rf,country_code,version,
                            path2output = path2output)
    mf.plot_AGBpot_uncertainty(agb_rf,country_code,version,
                            path2output = path2output)
    mf.plot_AGBseq_final(agb_rf,country_code,version,path2output = path2output)

cal_r2,val_r2 = cv.cal_val_train_test(Xall[training_mask[landmask]],
                                agb.values[training_mask],rf,path2calval,
                                country_code, version)
