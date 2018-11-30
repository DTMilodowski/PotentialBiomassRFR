"""
30/11/2018 - DTM
Rewritten some of the functions specific to Forests2020 potential biomass work
- no LUH data
- restricted set of soilgrids parameters

12/11/2018 - JFE
This file contains the definition of some useful functions
for the pantrop-AGB-LUH work
"""

import xarray as xr #xarray to read all types of formats
import glob
import numpy as np
import sys

# Load predictor variables
def get_predictors(country_code,return_landmask = True):
    path = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
    path2wc = path+'wc2/'
    path2sg = path+'soilgrids/'
    path2agb = path+'agb/'

    wc2 = xr.concat([xr.open_rasterio(f) for f in sorted(glob.glob(path2wc+'*tif'))],dim='band')
    #worldclim2 data
    wc2_mask = wc2[0]!=wc2[0,0,0]
    for ii in range(wc2.shape[0]):
        wc2_mask = wc2_mask & (wc2[ii]!=wc2[ii,0,0])
    print('Loaded WC2 data')

    #soilgrids data - filter out a bunch of variables correlated with land cover
    soilfiles_all = glob.glob(path2sg+'*tif')
    soilfiles = []
    #             %sand %silt %clay %D2Rhorizon %probRhorizon %D2bedrock
    filtervars = ['SNDPPT','SLTPPT','CLYPPT','BDRICM','BDRLOG','BDTICM']
    for ff in range(len(soilfiles_all)):
        if soilfiles_all[ff].split('/')[-1].split('.')[0].split('_')[0] in filtervars:
            soilfiles.append(soilfiles_all[ff])

    soil= xr.concat([xr.open_rasterio(f) for f in sorted(soilfiles)],dim='band')
    soil_mask = soil[0]!=soil[0,0,0]
    for ii in range(soil.shape[0]):
        soil_mask = soil_mask & (soil[ii]!=soil[ii,0,0])
    print('Loaded SOILGRIDS data')

    #also load the AGB data to only perform the PCA for places where there is both AGB and uncertainty
    agb = xr.open_rasterio('%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code))
    agb_mask = agb[0]!=agb.nodatavals[0]

    unc = xr.open_rasterio('%sAvitabile_AGB_Uncertainty_%s_1km.tif' % (path2agb,country_code))
    unc_mask = unc[0]!=unc.nodatavals[0]

    #create the land mask knowing that top left pixels (NW) are all empty
    # for now, ignoring uncertainty as want to include N. Australia in training
    landmask = (wc2_mask.values & soil_mask.values & agb_mask.values)# & unc_mask.values)

    #create the empty array to store the predictors
    predictors = np.zeros([landmask.sum(),soil.shape[0]+wc2.shape[0]])

    #iterate over variables to create the large array with data
    counter = 0

    #first wc2
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


# get areas for a global grid with 0.25x0.25 res
# edited to take in  optional grid dimensions
default_lat = np.arange(90-1/8.,-90.,-1/4.)
default_lon = np.arange(-180+1/8.,180.,1/4.)
def get_areas(latorig = default_lat, lonorig = default_lon):
    #latorig = np.arange(90-1/8.,-90.,-1/4.)
    #lonorig = np.arange(-180+1/8.,180.,1/4.)
    areas = np.zeros([latorig.size,lonorig.size])
    res = np.abs(latorig[1]-latorig[0])
    for la,latval in enumerate(latorig):
        areas[la]= (6371e3)**2 * ( np.deg2rad(0+res/2.)-np.deg2rad(0-res/2.) ) * (np.sin(np.deg2rad(latval+res/2.))-np.sin(np.deg2rad(latval-res/2.)))

    return areas
