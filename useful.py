"""
12/11/2018 - JFE
This file contains the definition of some useful functions
for the pantrop-AGB-LUH work
"""

import xarray as xr #xarray to read all types of formats
import glob
import numpy as np
import sys

def get_predictors(y0=2000,y1=None,luh_file='/disk/scratch/local.2/jexbraya/LUH2/states.nc',return_landmask = True):

    #check that both years are defined, if not assume only one year of data needed
    if y1 == None:
        y1 = y0
    #check that first year is before the last, otherwise invert
    if y0 > y1:
        y0,y1 = y1,y0

    print('Getting data for years %4i - %4i' % (y0,y1))

    ### first get datasets
    #LUH2 data provided by Uni of Maryland
    luh = xr.open_dataset(luh_file,decode_times=False)
    luh_mask = ~luh.primf[0].isnull()
    print('Loaded LUH data')

    #define time - start in 850 for the historical, 2015 for the SSPs
    if luh_file == '/disk/scratch/local.2/jexbraya/LUH2/states.nc':
        luh_time = 850+luh.time.values
    else:
        luh_time = 2015+luh.time.values

    #worldclim2 data regridded to 0.25x0.25
    wc2 = xr.concat([xr.open_rasterio(f) for f in sorted(glob.glob('/disk/scratch/local.2/jexbraya/WorldClim2/0.25deg/*tif'))],dim='band')
    wc2_mask = wc2[0]!=wc2[0,0,0]
    for ii in range(wc2.shape[0]):
        wc2_mask = wc2_mask & (wc2[ii]!=wc2[ii,0,0])
    print('Loaded WC2 data')

    #soilgrids data regridded to 0.25x0.25
    soil= xr.concat([xr.open_rasterio(f) for f in sorted(glob.glob('/disk/scratch/local.2/jexbraya/soilgrids/0.25deg/*tif'))],dim='band')
    soil_mask = soil[0]!=soil[0,0,0]
    for ii in range(soil.shape[0]):
        soil_mask = soil_mask & (soil[ii]!=soil[ii,0,0])
    print('Loaded SOILGRIDS data')

    #also load the AGB data to only perform the PCA for places where there is both AGB and uncertainty
    agb = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Map_0.25d.tif')
    agb_mask = agb[0]!=agb.nodatavals[0]

    unc = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Uncertainty_0.25d.tif')
    unc_mask = unc[0]!=unc.nodatavals[0]

    #create the land mask knowing that top left pixels (NW) are all empty
    landmask = (luh_mask.values & wc2_mask.values & soil_mask.values & agb_mask.values & unc_mask.values)

    #define the LUH variables of interest
    luh_pred = ['primf','primn','secdf','secdn','urban','c3ann','c4ann','c3per','c4per','c3nfx','pastr','range','secma']
    #create the empty array to store the predictors
    predictors = np.zeros([landmask.sum(),len(luh_pred)+soil.shape[0]+wc2.shape[0]])

    #iterate over variables to create the large array with data
    counter = 0

    #first LUH
    for landuse in luh_pred:
        #get average over the period
        if y0 != y1:
            predictors[:,counter] = luh[landuse][(luh_time>=y0) & (luh_time<=y1)].mean(axis=0).values[landmask]
        else:
            predictors[:,counter] = luh[landuse][(luh_time==y0)].values[0][landmask]
        counter += 1
    print('Extracted LUH data')
    #then wc2
    for bi in wc2:
        predictors[:,counter] = bi.values[landmask]
        counter += 1
    print('Extracted WC2 data')
    #then soil properties
    for sp in soil:
        predictors[:,counter] = sp.values[landmask]
        counter += 1
    print('Extracted SOILGRIDS data')

    if return_landmask:
        return(predictors,landmask)
    else:
        return(predictors)
