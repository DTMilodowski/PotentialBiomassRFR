"""
14/11/2018 - JFE
This files loads the best model from the GridSearchCV performed on the
RandomForestRegressor.
"""

from useful import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import pandas as pd

pca = joblib.load('/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/saved_algorithms/pca_pipeline.pkl')

#load the fitted  the fitted rf_grid
rf = joblib.load('/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/saved_algorithms/rf_grid.pkl').best_estimator_


#iterate over years
years = np.arange(1850,2016)
for yy, year in enumerate(years):
    predictors,landmask = get_predictors(y0=year)

    #transform the data
    X = pca.transform(predictors)

    if yy == 0:
        #get the agb data
        y = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Map_0.25d.tif')[0].values[landmask]

        #create coordinates
        lat = xr.IndexVariable('lat',np.arange(90-0.125,-90,-0.25),attrs={'units':'degrees_north'})
        lon = xr.IndexVariable('lon',np.arange(-180+0.125,180,0.25),attrs={'units':'degrees_east'})
        time= xr.IndexVariable('time',years,attrs={'units':'years'})

        #create empty variable to store results
        agb_hist = xr.DataArray(data=np.zeros([time.size,lat.size,lon.size])-9999.,
                                dims=('time','lat','lon'),
                                coords={'time':time,'lat':lat,'lon':lon},
                                attrs={'_FillValue':-9999.,'units':'Mg ha-1','description':'AGB reconstructed using LUH data'},
                                name='AGB_LUH')

    agb_hist[yy].values[landmask] = rf.predict(X)

#save to a nc file
agb_hist.to_netcdf('/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/output/AGB_hist.nc')
