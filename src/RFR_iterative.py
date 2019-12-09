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

country_code = sys.argv[1]#'WAFR'
version = sys.argv[2]#'002'
iterations = int(sys.argv[3])#5
#load = sys.argv[4]#'new'

path2alg = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/saved_algorithms'
path2data = '/disk/scratch/local.2/PotentialBiomass/processed/%s/' % country_code
path2agb = path2data+'agb/'
path2calval = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/calval/'
path2output = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/output/'

agb_source = 'avitabile'
#agb_source = 'globbiomass'

# load hyperopt trials object to get hyperparameters for rf
hyperopt_trials = '%s/%s_%s_rf_hyperopt_trials.p' % (path2alg,'BRA','013')
trials = pickle.load(open(hyperopt_trials, "rb"))
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

#pca = joblib.load('%s/%s_%s_pca_pipeline.pkl' % (path2alg,country_code,version))
pca = make_pipeline(StandardScaler(),PCA(n_components=0.999))

# load all predictors to generate preprocessing minmax scaler transformation
predictors_full,landmask = useful.get_predictors(country_code, training_subset=False)
#get the agb data
if agb_source == 'avitabile':
    agb = xr.open_rasterio('%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code))[0]
elif agb_source == 'globbiomass':
    agb_avitabile = xr.open_rasterio('%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code))[0]
    agb = xr.open_rasterio('%s/globiomass/%s_globiomass_agb_1km.tif' % (path2agb,country_code))[0]
    agb.values=agb.values.astype('float')
    agb.values[np.isnan(agb_avitabile.values)]=np.nan
else:
    agb = xr.open_rasterio('%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code))[0] # default is avitabile

# Get additional data masks
initial_training_mask = useful.get_mask(country_code,mask_def=1)
other_stable_forest_mask = useful.get_mask(country_code,mask_def=2)

# Run PCA transformation on predictor variables
# fit PCA
pca.fit(predictors_full)
joblib.dump(pca,'%s/%s_%s_pca_pipeline.pkl' % (path2alg,country_code,version))
Xall = pca.transform(predictors_full)

#yall = agb_globiomass.values[landmask]
yall = agb.values[landmask]

# First run of random forest regression model, with inital training set.
# Create the random forest object with predefined parameters
rf = RandomForestRegressor(bootstrap=True,
            criterion='mse',           # criteria used to choose split point at each node
            max_depth= int(trace['max_depth'][idx]),            # ***maximum number of branching levels within each tree
            max_features=int(trace['max_features'][idx]),       # ***the maximum number of variables used in a given tree
            min_samples_leaf=int(trace['min_samples_leaf'][idx]),       # ***The minimum number of samples required to be at a leaf node
            min_samples_split=int(trace['min_samples_split'][idx]),       # ***The minimum number of samples required to split an internal node
            n_estimators=int(trace['n_estimators'][idx]),#trace['n_estimators'],          # ***Number of trees in the random forest
            n_jobs=10,                 # The number of jobs to run in parallel for both fit and predict
            oob_score=True,            # use out-of-bag samples to estimate the R^2 on unseen data
            random_state=112358,         # seed used by the random number generator
            )

# get subset of predictors for initial training set
X = Xall[initial_training_mask[landmask]]
y = yall[initial_training_mask[landmask]]

# fit the model
rf.fit(X,y)

# Iterative augmentation of training dataset (getting variance is memory & processor intensive)
# - get additional stable forest areas
Xtest = Xall[other_stable_forest_mask[landmask]]
ytest = yall[other_stable_forest_mask[landmask]]

# now iterate, filtering out other stable forest pixels for which the observed biomass
# is not within error of the predicted potential biomass
AGBpot, training_set, rf = useful.iterative_augmentation_of_training_set_obs_vs_pot_v3(ytest,
                                            y, Xtest, X, Xall, iterations,
                                            landmask, initial_training_mask,
                                            other_stable_forest_mask, rf,
                                            percentile_cutoff = 10)
iterations = AGBpot.shape[0]

# Save rf model for future reference
joblib.dump(rf,'%s/%s_%s_%s_rf_iterative.pkl' % (path2alg,country_code,version,agb_source))

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
#agb_rf.AGBobs.values[landmask]  = agb_globiomass.values[landmask]
agb_rf.AGBobs.values[landmask]  = agb.values[landmask]
for ii in range(0,iterations):
    key = 'AGBpot%i' % (ii+1)
    agb_rf[key].values[landmask]= AGBpot[ii][landmask]
    key = 'trainset%i' % (ii+1)
    agb_rf[key].values[landmask] = training_set[ii][landmask]

#save to a nc file
comp = dict(zlib=True, complevel=1)
encoding = {var: comp for var in agb_rf.data_vars}
#nc_file = '%s%s_%s_AGB_globiomass_potential_RFR_worldclim_soilgrids.nc' % (path2output,country_code,version)
nc_file = '%s%s_%s_AGB_%s_potential_RFR_worldclim_soilgrids.nc' % (path2output,country_code,version,agb_source)
agb_rf.to_netcdf(path=nc_file)#,encoding=encoding)

# plot stuff
# Initial cal-val plot
mf.plot_AGBpot_iterations(agb_rf,iterations,country_code,version,path2output = path2output,agb_source=agb_source)
mf.plot_training_residuals(agb_rf,iterations,country_code,version,path2output=path2output,vmin=[0,0,-50],vmax=[200,200,50])
mf.plot_training_areas_iterative(agb_rf,iterations,country_code,version,path2output = path2output)
mf.plot_AGB_AGBpot_training(agb_rf,iterations,country_code,version,path2output = path2output)

training_mask_final = (training_set[-1]>0)*landmask
#cal_r2,val_r2 = cv.cal_val_train_test(Xall[training_mask_final[landmask]],agb_globiomass.values[training_mask_final],
#                                rf,path2calval, country_code, version)
cal_r2,val_r2 = cv.cal_val_train_test(Xall[training_mask_final[landmask]],agb.values[training_mask_final],
                                rf,path2calval, country_code, version)
