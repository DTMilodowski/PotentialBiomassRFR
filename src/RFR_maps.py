"""
30/11/2018 - DTM
Rewritten for F2020 project, including paths, extent processing, outputs, present
day only (no time series)

14/11/2018 - JFE
This files loads the fitted models and produces annual AGB maps
"""

from useful import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import pandas as pd

country_code = sys.argv[1]
version = sys.argv[2]

path2alg = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/saved_algorithms/'
path2calval = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/calval/'
path2data = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
path2agb = path2data+'agb/'
path2output = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/output/'

agb = xr.open_rasterio('%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code))[0].values

#load the fitted rfs and pca
rf_med = joblib.load('%s%s_%s_rf_mean.pkl' % (path2alg,country_code,version)).best_estimator_
#rf_upp = joblib.load('%s%s_rf_upp.pkl' % (path2alg,country_code)).best_estimator_
#rf_low = joblib.load('%s%s_rf_low.pkl' % (path2alg,country_code)).best_estimator_
pca = joblib.load('%s%s_%s_pca_pipeline.pkl' % (path2alg,country_code,version))

"""
lat = np.arange(90-0.125,-90,-0.25)
lon = np.arange(-180+0.125,180,0.25)
"""
# Generate lat/lon grid for target
transform = Affine.from_gdal(*agb.attrs['transform'])
nx, ny = agb.sizes['x'], agb.sizes['y']
lon, lat = np.meshgrid(np.arange(nx)+0.5, np.arange(ny)+0.5) * transform
#get areas
areas = get_areas()

predictors,landmask = get_predictors(country_code)

#transform the data
X = pca.transform(predictors)

#create coordinates
coords = {'time': (['time'],years.astype('d'),{'units':'year'}),
              'lat': (['lat'],lat,{'units':'degrees_north','long_name':'latitude'}),
              'lon': (['lon'],lon,{'units':'degrees_east','long_name':'longitude'})}

#create empty variable to store results
attrs={'_FillValue':-9999.,'units':'Mg ha-1'}
data_vars = {}

data_vars['AGB_mean'] = (['lat','lon'],np.zeros([lat.size,lon.size])-9999.,attrs)
data_vars['AGB_pot'] = (['lat','lon'],np.zeros([lat.size,lon.size])-9999.,attrs)

"""
for lvl in ['mean','upper','lower']:
    data_vars['AGB_'+lvl] = (['time','lat','lon'],np.zeros([years.size,lat.size,lon.size])-9999.,attrs)
"""
agb_rf = xr.Dataset(data_vars=data_vars,coords=coords)

agb_rf.AGB_mean.values[landmask]  = agb[landmask]
agb_rf.AGBpot_mean.values[landmask]  = rf_med.predict(X)
#agb_rf.AGB_upper[yy].values[landmask] = rf_upp.predict(X)
#agb_rf.AGB_lower[yy].values[landmask] = rf_low.predict(X)

#save to a nc file
encoding = {'AGB_mean':{'zlib':True,'complevel':1},
            'AGB_upper':{'zlib':True,'complevel':1},
            'AGB_lower':{'zlib':True,'complevel':1},}


agb_rf.to_netcdf('%s%s_%s_AGB_potential_RFR_worldclim_soilgrids.nc' % (path2output,country_code,version),encoding=encoding)
