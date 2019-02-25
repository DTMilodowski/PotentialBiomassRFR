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
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

country_code = sys.argv[1]#'WAFR'
version = sys.argv[2]#'002'
iterations = int(sys.argv[3])#5
load = sys.argv[4]#'new'

path2alg = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/saved_algorithms'
path2data = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
path2agb = path2data+'agb/'
path2calval = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/calval/'
path2output = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/output/'

pca = joblib.load('%s/%s_%s_pca_pipeline.pkl' % (path2alg,country_code,version))

# load all predictors to generate preprocessing minmax scaler transformation
predictors_full,landmask = useful.get_predictors(country_code, training_subset=False)
#get the agb data
agb = xr.open_rasterio('%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code))[0]

# Get additional data masks
initial_training_mask = useful.get_mask(country_code,mask_def=1)
other_stable_forest_mask = useful.get_mask(country_code,mask_def=2)

# Run PCA transformation on predictor variables
Xall = pca.transform(predictors_full)

yall = agb.values[landmask]

# First run of random forest regression model, with inital training set.
# Create the random forest object with predefined parameters
rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=20, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
           oob_score=True, random_state=None, verbose=0, warm_start=False)

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
AGBpot, training_set, rf = useful.iterative_augmentation_of_training_set_obs_vs_pot(ytest, y, Xtest, X, Xall, iterations,
                                            landmask, initial_training_mask,
                                            other_stable_forest_mask, rf,stopping_condition=0.01)
iterations = AGBpot.shape[0]

# Save rf model for future reference
joblib.dump(rf,'%s/%s_%s_rf_iterative.pkl' % (path2alg,country_code,
                                                version))

# Initial cal-val plot
training_mask_final = (training_set[-1]>0)*landmask
cal_r2,val_r2 = cv.cal_val_train_test(Xall[training_mask_final[landmask]],agb.values[training_mask_final],
                                rf,path2calval, country_code, version)

# convert training set and AGBpot to xdarray for easy plotting and export to
# netcdf
# first deal with metadata and coordinates
tr = agb.attrs['transform']#Affine.from_gdal(*agb.attrs['transform'])
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
mf.plot_AGBpot_iterations(agb_rf,iterations,country_code,version,path2output = path2output)
mf.plot_training_residuals(agb_rf,iterations,country_code,version,path2output=path2output,vmin=[0,0,-50],vmax=[200,200,50])
mf.plot_training_areas_iterative(agb_rf,iterations,country_code,version,path2output = path2output)
mf.plot_AGB_AGBpot_training(agb_rf,iterations,country_code,version,path2output = path2output)
