"""
14/11/2018 - JFE
This files loads the fitted models and produces annual AGB maps
"""

from useful import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import pandas as pd


pca = joblib.load('/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/saved_algorithms/pca_pipeline.pkl')

#load the fitted rfs
rf_med = joblib.load('/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/saved_algorithms/rf_mean.pkl')
rf_upp = joblib.load('/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/saved_algorithms/rf_upper.pkl')
rf_low = joblib.load('/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/saved_algorithms/rf_lower.pkl')

#iterate over years
years = np.arange(1850,2016)
lat = np.arange(90-0.125,-90,-0.25)
lon = np.arange(-180+0.125,180,0.25)

#get areas
areas = get_areas()

for yy, year in enumerate(years):
    predictors,landmask = get_predictors(y0=year)

    #transform the data
    X = pca.transform(predictors)

    if yy == 0:
        #create coordinates
        coords = {'time': (['time'],years.astype('d'),{'units':'year'}),
                  'lat': (['lat'],lat,{'units':'degrees_north','long_name':'latitude'}),
                  'lon': (['lon'],lon,{'units':'degrees_east','long_name':'longitude'})}

        #create empty variable to store results
        attrs={'_FillValue':-9999.,'units':'Mg ha-1'}
        data_vars = {}
        for lvl in ['mean','upper','lower']:
            data_vars['AGB_'+lvl] = (['time','lat','lon'],np.zeros([years.size,lat.size,lon.size])-9999.,attrs)
            data_vars['ts_' + lvl] = (['time'],np.zeros(years.size),{'units': 'Pg','long_name': 'total AGB'})

        agb_hist = xr.Dataset(data_vars=data_vars,coords=coords)

    agb_hist.AGB_mean[yy].values[landmask]  = rf_med.predict(X)
    agb_hist.AGB_upper[yy].values[landmask] = rf_upp.predict(X)
    agb_hist.AGB_lower[yy].values[landmask] = rf_low.predict(X)

    #get time series
    agb_hist.ts_mean[yy] = (agb_hist.AGB_mean[yy].values*areas)[landmask].sum()*1e-13
    agb_hist.ts_upper[yy] = (agb_hist.AGB_upper[yy].values*areas)[landmask].sum()*1e-13
    agb_hist.ts_lower[yy] = (agb_hist.AGB_lower[yy].values*areas)[landmask].sum()*1e-13


#save to a nc file
encoding = {'AGB_mean':{'zlib':True,'complevel':1},
            'AGB_upper':{'zlib':True,'complevel':1},
            'AGB_lower':{'zlib':True,'complevel':1},}

agb_hist.to_netcdf('/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/output/AGB_hist.nc',encoding=encoding)
