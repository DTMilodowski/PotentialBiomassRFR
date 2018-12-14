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
from affine import Affine
import glob
import numpy as np
import sys
import set_training_areas
import rasterio

# Load predictor variables
def get_predictors(country_code,return_landmask = True, training_subset=False, subset_def=1):

    path = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
    path2wc = path+'wc2/'
    path2sg = path+'soilgrids/'
    path2agb = path+'agb/'

    #worldclim2 data
    nodata=[]
    for ff in sorted(glob.glob(path2wc+'*tif')):
        nodata.append(rasterio.open(ff).nodatavals[0])

    wc2 = xr.concat([xr.open_rasterio(f) for f in sorted(glob.glob(path2wc+'*tif'))],dim='band')
    wc2_mask = wc2[0]!=nodata[0]#wc2[0]!=wc2[0,0,0]
    for ii in range(wc2.shape[0]):
        wc2_mask = wc2_mask & (wc2[ii]!=nodata[ii])#(wc2[ii]!=wc2[ii,0,0])
    print('Loaded WC2 data')

    #soilgrids data - filter out a bunch of variables correlated with land cover
    soilfiles_all = glob.glob(path2sg+'*tif')
    soilfiles = []
    #             %sand %silt %clay %D2Rhorizon %probRhorizon %D2bedrock
    filtervars = ['SNDPPT','SLTPPT','CLYPPT','BDRICM','BDRLOG','BDTICM']
    for ff in range(len(soilfiles_all)):
        if soilfiles_all[ff].split('/')[-1].split('.')[0].split('_')[0] in filtervars:
            soilfiles.append(soilfiles_all[ff])

    nodata=[]
    for ff in sorted(soilfiles):
        nodata.append(rasterio.open(ff).nodatavals[0])

    soil= xr.concat([xr.open_rasterio(f) for f in sorted(soilfiles)],dim='band')
    soil_mask = soil[0]!=nodata[0]#soil[0]!=soil[0,0,0]
    for ii in range(soil.shape[0]):
        soil_mask = soil_mask & soil[ii]!=nodata[0]#(soil[ii]!=soil[ii,0,0])
    print('Loaded SOILGRIDS data')

    #also load the AGB data to only perform the PCA for places where there is both AGB and uncertainty
    agb_file = '%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code)
    agb = xr.open_rasterio(agb_file)
    agb_mask = agb.values[0]!=np.float32(agb.nodatavals[0])

    unc_file = '%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code)
    unc = xr.open_rasterio(unc_file)
    unc_mask = unc[0]!=unc.nodatavals[0]

    #create the land mask knowing that top left pixels (NW) are all empty
    # for now, ignoring uncertainty as want to include N. Australia in training
    if training_subset:
        training_mask = set_training_areas.set(path,subset=subset_def)
        landmask = (training_mask & wc2_mask.values & soil_mask.values & agb_mask)# & unc_mask.values)
    else:
        landmask = (wc2_mask.values & soil_mask.values & agb_mask)# & unc_mask.values)

    #create the empty array to store the predictors
    predictors = np.zeros([landmask.sum(),soil.shape[0]+wc2.shape[0]])

    # check the mask dimensions
    if len(landmask.shape)>2:
        print('\t\t caution shape of landmask is: ', landmask.shape)
        landmask = landmask[0]

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
