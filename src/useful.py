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
import forestci as fci
from copy import deepcopy

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

# equivalent function to above, but for specified mask, rather than creating mask within function
def get_predictors_for_defined_mask(country_code,mask):

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

    landmask = (mask & wc2_mask.values & soil_mask.values & agb_mask)# & unc_mask.values)

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

    return(predictors)

# Get a defined mask
# - country code is a prefix identifying the study area
# - mask_def is the option to select different data masks as required
# --- 0 (default) -> landmask
# --- 1           -> training mask (hinterland forest landscapes
#                    and stable sparsely vegetated ESA-CCI classes)
# --- 2           -> other stable forest areas from ESA-CCI
def get_mask(country_code, mask_def=0):

    path = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
    path2wc = path+'wc2/'; path2sg = path+'soilgrids/'; path2agb = path+'agb/'

    #worldclim2 data
    nodata=[]
    ff = sorted(glob.glob(path2wc+'*tif'))[0]
    nodata = rasterio.open(ff).nodatavals[0]
    wc2 = xr.open_rasterio(ff)
    wc2_mask = wc2.values!=nodata

    #soilgrids data - filter out a bunch of variables correlated with land cover
    soilfiles_all = glob.glob(path2sg+'*tif')
    soilfiles = []
    filtervar = 'SNDPPT'
    for ff in range(len(soilfiles_all)):
        if soilfiles_all[ff].split('/')[-1].split('.')[0].split('_')[0] == filtervar:
            soilfiles.append(soilfiles_all[ff])
    ff = sorted(soilfiles)[0]
    nodata = rasterio.open(ff).nodatavals[0]
    soil= xr.open_rasterio(ff)
    soil_mask = soil.values!=nodata

    #also load the AGB data to only perform the PCA for places where there is both AGB and uncertainty
    agb_file = '%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code)
    agb = xr.open_rasterio(agb_file)
    agb_mask = agb.values[0]!=np.float32(agb.nodatavals[0])

    # land mask
    if(mask_def == 0):
        mask = (training_mask & wc2_mask & soil_mask & agb_mask)
    # training mask (HFL based)
    elif(mask_def == 1):
        mask = set_training_areas.set(path,subset=1)
    # other stable forest
    elif(mask_def == 2):
        landmask = (wc2_mask & soil_mask & agb_mask)
        trainmask = set_training_areas.set(path,subset=1)
        mask = set_training_areas.get_stable_forest_outside_training_areas(path,trainmask,landmask)
    else:
        mask = (training_mask & wc2_mask & soil_mask & agb_mask)
    # check the mask dimensions
    if len(mask.shape)>2:
        print('\t\t caution shape of landmask is: ', landmask.shape)
        mask = mask[0]

    return mask


#===============================================================================
# AUXILLIARY FUNCTIONS TO CALCULATE USEFUL INFORMATION
#-------------------------------------------------------------------------------
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

"""
#===============================================================================
# FUNCTIONS THAT ARE CALLED DURING THE ITERATIVE TRAINING DATA SELECTION PROCESS
#-------------------------------------------------------------------------------
"""
# This uses the confidence intervals from the random forest to trim the training
# set
def iterative_augmentation_of_training_set_rfci(ytest, y, Xtest, X, Xall, iterations,
                                            landmask, initial_training_mask,
                                            other_stable_forest_mask, rf, yscaler,
                                            memory_limit = 8000):
    # define host array for potential biomass estimates at each iteration
    AGBpot = np.zeros((iterations,landmask.shape[0],landmask.shape[1]))*np.nan
    AGBpot[0][landmask] = yscaler.inverse_transform(rf.predict(Xall).reshape(-1, 1)).ravel()

    # Also track which pixels are included in the additional training set
    # pixels marked 1 are in the initial training set
    # pixels marked 2 are in the additional training set
    training_set = np.zeros((iterations,landmask.shape[0],landmask.shape[1]))
    training_set[0] = initial_training_mask.copy()

    # keep track of row and column indices for updating training set maps
    cols,rows = np.meshgrid(np.arange(landmask.shape[1]),np.arange(landmask.shape[0]))
    col_idx = cols[other_stable_forest_mask]
    row_idx = rows[other_stable_forest_mask]

    # now iterate, stopping after specified number of iterations, or if
    # training set stabilises sooner
    print("starting iterative training selection")
    ii=1
    while(ii<iterations):
        # track number of additional training pixels considered in this iteration
        n_additional = ytest.size
        # predict potential biomass
        # - (ii) for additional training subset
        ytest_pr = rf.predict(Xtest)
        # calculate variance in estimate for this subset
        var = fci.random_forest_error(rf, X, Xtest, memory_constrained = True,
                                        memory_limit = memory_limit)
        # Filter out pixels with biomass below potential (accounting for error)
        subset = (ytest[:,0]+np.sqrt(var)) >= ytest_pr

        # Check how many pixels were removed from the additional training set
        n_removed = n_additional-subset.sum()
        print("Iteration %i complete, removed %i out of %i pixels from training" %
                (ii+1,n_removed, n_additional))
        # repeat iterations unit either n_iter is reached, or reached steady state
        if(n_removed==0):
            # skip to end of loop
            ii = iterations
        else:
            # record AGBpot & training set for this next iteration
            AGBpot[ii][landmask] = yscaler.inverse_transform(rf.predict(Xall).reshape(-1, 1)).ravel()
            training_set[ii] = initial_training_mask.copy()
            training_set[ii,row_idx,col_idx] = 2

            # refine subset
            ytest = ytest[subset]
            Xtest = Xtest[subset,:]
            col_idx = col_idx[subset]
            row_idx = row_idx[subset]

            # fit new random forest with new training subset
            Xiter=np.concatenate((X,Xtest),axis=0)
            yiter=np.concatenate((y,ytest),axis=0)
            rf.fit(Xiter,yiter)
            ii+=1
    np.save('AGBpot_test.npy',AGBpot)
    np.save('training_test.npy',training_set)
    return AGBpot, training_set, rf

# This version is simpler - it simply filters out additional stable forest for
# which the AGBobs < AGBpot in each iteration. It is also much faster! Unlike
# the previous function, it does not need the data to be scaled, either
def iterative_augmentation_of_training_set_obs_vs_pot(ytest, y, Xtest, X, Xall, iterations,
                                            landmask, initial_training_mask,
                                            other_stable_forest_mask, rf,stopping_condition=0.05):
    # define host array for potential biomass estimates at each iteration
    AGBpot = np.zeros((iterations+1,landmask.shape[0],landmask.shape[1]))*np.nan
    AGBpot[0][landmask] = rf.predict(Xall)

    # Also track which pixels are included in the additional training set
    # pixels marked 1 are in the initial training set
    # pixels marked 2 are in the additional training set
    training_set = np.zeros((iterations+1,landmask.shape[0],landmask.shape[1]))
    training_set[0] = initial_training_mask.copy()

    # keep track of row and column indices for updating training set maps
    cols,rows = np.meshgrid(np.arange(landmask.shape[1]),np.arange(landmask.shape[0]))
    col_idx = cols[other_stable_forest_mask]
    row_idx = rows[other_stable_forest_mask]

    # now iterate, stopping after specified number of iterations, or if
    # training set stabilises sooner
    print("starting iterative training selection")

    # predict potential biomass for additional training subset
    ytest_pr = rf.predict(Xtest)
    # calculate RMSE of subset
    rmse_pr = np.sqrt(np.mean((ytest_pr-ytest)**2))
    # copy rf model
    rf_pr = deepcopy(rf)

    ii=0
    while(ii<iterations):
        # track number of additional training pixels considered in this iteration
        n_additional = ytest.size
        # Filter out pixels with biomass below potential (accounting for error)
        residual = ytest - ytest_pr
        residual_as_fraction = residual/ytest
        threshold = np.percentile(residual_as_fraction,25)

        #subset = (ytest >= ytest_pr)
        subset = np.all((residual_as_fraction >= threshold,ytest >= ytest_pr),axis=0)

        # refine subset
        ytest = ytest[subset]
        Xtest = Xtest[subset,:]
        col_idx = col_idx[subset]
        row_idx = row_idx[subset]

        # fit new random forest with new training subset
        Xiter=np.concatenate((X,Xtest),axis=0)
        yiter=np.concatenate((y,ytest),axis=0)
        rf.fit(Xiter,yiter)

        # record AGBpot & training set for this next iteration
        AGBpot[ii+1][landmask] = rf.predict(Xall)
        training_set[ii+1] = initial_training_mask.copy()
        training_set[ii+1,row_idx,col_idx] = 2


        # predict potential biomass for additional training subset
        ytest_pr = rf.predict(Xtest)
        # calculate RMSE of subset
        rmse = np.sqrt(np.mean((ytest_pr-ytest)**2))

        # Check how many pixels were removed from the additional training set
        n_removed = n_additional-subset.sum()
        print("Iteration %i complete, removed %i out of %i pixels from training, %.1f" %
                (ii+1,n_removed, n_additional,float(n_removed)/float(n_additional)*100.))
        print("RMSE previous: %.02f; RMSE updated: %.02f; percentage change: %.02f" %
                (rmse_pr, rmse,(rmse-rmse_pr)/(rmse_pr)*100.))

        # repeat iterations unit either n_iter is reached, or reached steady state
        #if(float(n_remove)/float(n_additional)<stopping_condition):
        if((rmse_pr-rmse)/(rmse_pr)<=stopping_condition):
            # if rmse increases, revert back to previous iteration
            if((rmse_pr-rmse)/(rmse_pr)<=0) :
                AGBpot=AGBpot[:ii+2]
                training_set = training_set[:ii+2]
                #rf=deepcopy(rf_pr)
            else:
                AGBpot=AGBpot[:ii+2]
                training_set = training_set[:ii+2]
            # skip to end of loop
            ii = iterations
        else:
            rf_pr = deepcopy(rf)
            rmse_pr = rmse
            n_additional_previous = n_additional
            ii+=1
    np.save('AGBpot_test.npy',AGBpot)
    np.save('training_test.npy',training_set)
    return AGBpot, training_set, rf
