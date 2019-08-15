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
#import forestci as fci
from copy import deepcopy

# Load predictor variables
def get_predictors(country_code,return_landmask = True, training_subset=False,
                    subset_def=1,apply_unc_mask=True):

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
    agb_mask[agb.values[0]<-3e38]=False

    unc_file = '%sAvitabile_AGB_Uncertainty_%s_1km.tif' % (path2agb,country_code)
    unc = xr.open_rasterio(unc_file)
    unc_mask = unc.values[0]!=np.float32(unc.nodatavals[0])
    unc_mask[unc.values[0]<-3e38]=False
    #create the land mask knowing that top left pixels (NW) are all empty
    # for now, ignoring uncertainty as want to include N. Australia in training
    if training_subset:
        training_mask = set_training_areas.set(path,subset=subset_def)
        if apply_unc_mask:
            landmask = (training_mask & wc2_mask.values & soil_mask.values & agb_mask & unc_mask)
        else:
            landmask = (training_mask & wc2_mask.values & soil_mask.values & agb_mask)
    else:
        if apply_unc_mask:
            landmask = (wc2_mask.values & soil_mask.values & agb_mask & unc_mask)
        else:
            landmask = (wc2_mask.values & soil_mask.values & agb_mask)

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
# --- 3           -> mapbiomas stable natural land cover classes
# --- 4           -> other stable forest areas from ESA-CCI
# --- 5           -> mapbiomas and hinterland forest landscapes
# --- 6           -> other stable forest areas from mapbiomas
# --- 7           -> other stable forest areas from mapbiomas & ESACCI
# --- 8           -> mapbiomas and hinterland forest landscapes & ESACCI natural
#                    non-forest
# --- 9           -> mapbiomas and hinterland forest landscapes and ESACCI natural
#                    non-forest with non-forest classes filtered according to
#                    protected areas
# --- 10          -> mapbiomas and hinterland forest landscapes within mapbiomas
#                    extent only
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
    # training mask (mapbiomas based)
    elif(mask_def == 3):
        landmask = (wc2_mask & soil_mask & agb_mask)
        mask = set_training_areas.set(path,subset=3)
    # other stable forest (initially mapbiomas based forest extent)
    elif(mask_def == 4):
        landmask = (wc2_mask & soil_mask & agb_mask)
        trainmask = set_training_areas.set(path,subset=3)
        mask = set_training_areas.get_stable_forest_outside_training_areas(path,trainmask,landmask,method==4)
    # training mask (mapbiomas & HFL & ESA-CCI bare areas)
    elif(mask_def == 5):
        landmask = (wc2_mask & soil_mask & agb_mask)
        mask = set_training_areas.set(path,subset=4)
    # other stable forest (ESA CCI)
    elif(mask_def == 6):
        landmask = (wc2_mask & soil_mask & agb_mask)
        trainmask = set_training_areas.set(path,subset=4)
        mask = set_training_areas.get_stable_forest_outside_training_areas(path,trainmask,landmask,method=1)
    # other stable forest (Mapbiomas & ESA CCI)
    elif(mask_def == 7):
        landmask = (wc2_mask & soil_mask & agb_mask)
        trainmask = set_training_areas.set(path,subset=4)
        mask = set_training_areas.get_stable_forest_outside_training_areas(path,trainmask,landmask,method=5)
    # training mask (mapbiomas & HFL & ESA-CCI natural non-forest areas)
    elif(mask_def == 8):
        landmask = (wc2_mask & soil_mask & agb_mask)
        mask = set_training_areas.set(path,subset=5)
    elif(mask_def == 9):
        landmask = (wc2_mask & soil_mask & agb_mask)
        mask = set_training_areas.set(path,subset=6)
    elif(mask_def == 10):
        landmask = (wc2_mask & soil_mask & agb_mask)
        mask = set_training_areas.set(path,subset=7)
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
"""
# This version is simpler - it simply filters out additional stable forest for
# which the AGBobs < AGBpot in each iteration. It is also much faster! Unlike
# the previous function, it does not need the data to be scaled, either.
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

# as above, but using a slightly different protocol
# The iterative scheme works as follows:
# 1) add possible additional training pixels to the training dataset.
# 2) assess model quality with 5-fold cross validation, using root-mean-squared-
#    error as the score metric
# 3) calculate residuals, as fraction of AGB estimate
# 4) remove additional training data with lowest X percent of the residuals
# 5) reassess model quality, repeating until root-mean-square error is within
#    threshold for acceptance (i.e. when removing pixels makes little difference
#    to predictive accuracy of model)
def iterative_augmentation_of_training_set_obs_vs_pot_v2(ytest, y, Xtest, X, Xall, iterations,
                                            landmask, initial_training_mask,
                                            other_stable_forest_mask, rf,
                                            stopping_condition=0.05,percentile_cutoff=20):
    """
    # start with reference run using initial training data only
    """
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

    """
    # Iteration 1 - add candidate pixels
    """
    # fit new random forest with new training subset
    Xiter=np.concatenate((X,Xtest),axis=0)
    yiter=np.concatenate((y,ytest),axis=0)
    rf.fit(Xiter,yiter)

    # record AGBpot & training set for this next iteration
    AGBpot[1][landmask] = rf.predict(Xall)
    training_set[1] = initial_training_mask.copy()
    training_set[1,row_idx,col_idx] = 2

    # calculate RMSE of subset
    ytest_predict = rf.predict(Xtest)
    rmse = np.sqrt(np.mean((ytest_predict-ytest)**2))
    n_additional = ytest.size

    # now iterate, stopping after specified number of iterations, or if
    # training set stabilises sooner
    print("starting iterative training selection")

    ii=1
    while(ii<iterations):

        # update rmse_pr with presemt rmse for comparison later in the loop
        rmse_pr=rmse
        n_additional_previous = n_additional

        # Filter out pixels with sufficiently negative residuals
        residual = ytest - ytest_predict
        residual_as_fraction = residual/ytest
        threshold = np.percentile(residual_as_fraction,percentile_cutoff)

        #subset = (ytest >= ytest_pr)
        #subset = np.all((residual_as_fraction >= threshold,ytest >= ytest_pr),axis=0)
        subset = residual_as_fraction >= threshold

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
            else:
                AGBpot=AGBpot[:ii+2]
                training_set = training_set[:ii+2]
            # skip to end of loop
            ii = iterations
        else:
            ii+=1
    np.save('AGBpot_test.npy',AGBpot)
    np.save('training_test.npy',training_set)
    return AGBpot, training_set, rf

# as above, but using a slightly different protocol
# The iterative scheme works as follows:
# 1) calculate residuals from naive initial model
# 2) add possible additional training pixels to the training dataset with
#    predicted AGB < observed AGB.
# 2) fit new model including augmented training set
# 3) check updated residuals for added training data; filter the added pixels
#    to remove the lowest X percent (normalised by observed AGB), AGBpred<AGBobs
# 4) repeat 2 & 3 for specified number of iterations (no cutoff applied - this
#    can be specified later once algorithm behaviour has been assessed)
def iterative_augmentation_of_training_set_obs_vs_pot_v3(ytest, y, Xtest, X, Xall, iterations,
                                            landmask, initial_training_mask,
                                            other_stable_forest_mask, rf,
                                            percentile_cutoff=10):
    """
    # start with reference run using initial training data only
    """
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

    """
    # Iteration 1 - add candidate pixels (stable forest with potential AGB<=AGBobs)
    """
    # Filter out pixels with biomass below potential in naive model
    Xadditional_forest = Xtest.copy()
    ytest_predict = rf.predict(Xtest)
    subset = (ytest >= ytest_predict)
    ytest = ytest[subset]
    Xtest = Xtest[subset,:]
    col_idx = col_idx[subset]
    row_idx = row_idx[subset]


    y_added_stable_forest_predict_previous = np.mean(rf.predict(Xadditional_forest))

    # fit new random forest with new training subset
    Xiter=np.concatenate((X,Xtest),axis=0)
    yiter=np.concatenate((y,ytest),axis=0)
    rf.fit(Xiter,yiter)

    # record AGBpot & training set for this next iteration
    AGBpot[1][landmask] = rf.predict(Xall)
    training_set[1] = initial_training_mask.copy()
    training_set[1,row_idx,col_idx] = 2

    # calculate RMSE of subset
    ytest_predict = rf.predict(Xtest)
    rmse = np.sqrt(np.mean((ytest_predict-ytest)**2))
    n_additional = ytest.size
    y_added_stable_forest_predict = np.mean(rf.predict(Xadditional_forest))

    print("Iteration 1 complete")
    print("RMSE: %.02f" % rmse)
    print("AGBpot of added pixels previous: %.02f; updated: %.02f" %
            (y_added_stable_forest_predict_previous,y_added_stable_forest_predict))

    # now iterate, stopping after specified number of iterations, or if
    # training set stabilises sooner
    print("starting iterative training refinement")

    ii=1
    while(ii<iterations):

        # update rmse_pr with presemt rmse for comparison later in the loop
        rmse_previous=rmse
        y_added_stable_forest_predict_previous = y_added_stable_forest_predict
        n_additional_previous = n_additional

        # Filter out pixels with sufficiently negative residuals
        residual = ytest - ytest_predict
        residual_as_fraction = residual/ytest
        threshold = np.percentile(residual_as_fraction,percentile_cutoff)

        #subset = residual_as_fraction >= threshold
        #subset = np.any((residual_as_fraction >= threshold,ytest >= ytest_predict),axis=0)
        subset = np.all((residual_as_fraction >= threshold,residual>0),axis=0)

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
        ytest_predict = rf.predict(Xtest)
        # calculate RMSE of subset
        rmse = np.sqrt(np.mean((ytest_predict-ytest)**2))
        # predict potential biomass for the full added stable forest class
        y_added_stable_forest_predict = np.mean(rf.predict(Xadditional_forest))

        # Check how many pixels were removed from the additional training set
        n_removed = n_additional-subset.sum()
        print("Iteration %i complete, removed %i out of %i pixels from training, %.1f" %
                (ii+1,n_removed, n_additional,float(n_removed)/float(n_additional)*100.))
        print("RMSE previous: %.02f; RMSE updated: %.02f; percentage change: %.02f" %
                (rmse_previous, rmse,(rmse-rmse_previous)/(rmse_previous)*100.))
        print("AGBpot of added pixels previous: %.02f; updated: %.02f" %
                (y_added_stable_forest_predict_previous,y_added_stable_forest_predict))

        ii+=1
    np.save('AGBpot_test.npy',AGBpot)
    np.save('training_test.npy',training_set)
    return AGBpot, training_set, rf

"""
#===============================================================================
# FUNCTIONS FOR QUICKLY LOADING DATASET IF NEEDED
#-------------------------------------------------------------------------------
"""
# load in mapbiomas for a given timestep
def load_mapbiomas(country_code,timestep=-1,aggregate=0):
    path = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
    mbfiles = sorted(glob.glob('%s/mapbiomas/*tif' % path))
    # option 0 -> no aggregation
    if aggregate == 0:
        mb = xr.open_rasterio(mbfiles[0]).values[timestep]
        lc = mb.copy()
    # option 1 -> aggregate to 8 classes
    elif aggregate == 1:
        mb = xr.open_rasterio(mbfiles[0]).values[timestep]
        lc = np.zeros(mb.shape)*np.nan
        lc[np.all((mb>=1,mb<=5),axis=0)] = 1                # Natural forest
        lc[np.all((mb>=11,mb<=13),axis=0)] = 2              # Natural non-forest
        lc[mb==9]= 3                                        # Plantation forest
        lc[mb==15] = 4                                      # Pasture
        lc[np.all((mb>=18,mb<=21),axis=0)] = 5              # Agriculture
        #lc[mb==21] = 6                                     # Mosaic agro-pastoral
        lc[mb==24] = 6                                      # Urban
        lc[np.any((mb==23,mb==29,mb==30,mb==25),axis=0)] = 7# other
    # option 2 -> aggregation to 8 classes above, but filtering this so that
    # only keep pixels that have consistent land cover from 2000-2008
    elif aggregate == 2:
        mb = xr.open_rasterio(mbfiles[0]).values[15:24] # 2000-2008 inclusive
        lc = np.zeros(mb.shape)*np.nan
        lc[np.all((mb[0]>=1,mb[0]<=5),axis=0)] = 1                      # Natural forest
        lc[np.all((mb[0]>=11,mb[0]<=13),axis=0)] = 2                    # Natural non-forest
        lc[mb[0]==9] = 3                                              # Plantation forest
        lc[mb[0]==15] = 4                                               # Pasture
        lc[np.all((mb[0]>=18,mb[0]<=20),axis=0)] = 5                    # Agriculture
        lc[mb[0]==21] = 6                                               # Mosaic agro-pastoral
        lc[mb[0]==24] = 7                                               # Urban
        lc[np.any((mb[0]==23,mb[0]==29,mb[0]==30,mb[0]==25),axis=0)]    # other
        for ii in range(1,mb.shape[0]):
            lc[lc!=mb[ii]] = np.nan
    else:
        mb = xr.open_rasterio(mbfiles[0]).values[timestep]
        lc = mb.copy()
    return lc


#-------------------------------------------------------------------------------
# Script to load in ESA-CCI landcover data for a given timestep
# Note that the legend for landcover types is:
#  1. Agriculture
#    10, 11, 12 Rainfed cropland
#    20 Irrigated cropland
#    30 Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover) (<50%)
#    40 Mosaic natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland (< 50%)
#  -----------------------
#  2. Forest
#    50 Tree cover, broadleaved, evergreen, closed to open (>15%)
#    60, 61, 62 Tree cover, broadleaved, deciduous, closed to open (> 15%)
#    70, 71, 72 Tree cover, needleleaved, evergreen, closed to open (> 15%)
#    80, 81, 82 Tree cover, needleleaved, deciduous, closed to open (> 15%)
#    90 Tree cover, mixed leaf type (broadleaved and needleleaved)
#   100 Mosaic tree and shrub (>50%) / herbaceous cover (< 50%)
#   160 Tree cover, flooded, fresh or brakish water
#   170 Tree cover, flooded, saline water
#  -----------------------
#  3. Grassland
#   110 Mosaic herbaceous cover (>50%) / tree and shrub (<50%)
#   130 Grassland
#  -----------------------
# 4. Wetland
#   180 Shrub or herbaceous cover, flooded, fresh-saline or brakishwater
#  -----------------------
# 5. Settlement
#   190 Urban
#  -----------------------
# 6. Shrub
#   120, 121, 122 Shrubland
#  -----------------------
# 7. Lichens/Mosses
#   140 Lichens and mosses
#  -----------------------
# 8. Sparse
#   150, 151, 152, 153 Sparse vegetation (tree, shrub, herbaceous cover)
#  -----------------------
# 9. Bare
#   200, 201, 202 Bare areas
#  -----------------------
# 10. Water
#   210 Water
#  -----------------------
# 11. Ice
#   220 Permanent snow and ice
#  -----------------------
def load_esacci(country_code,year=2015,aggregate=0):
    path = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
    files = sorted(glob.glob('%s/esacci/*%i*lccs-class*tif' % (path,year)))
    # option 0 -> no aggregation
    if aggregate == 0:
        landcover = xr.open_rasterio(files[0]).values[0]
    # option 1 -> aggregate to 8 classes
    elif aggregate == 1:
        landcover = xr.open_rasterio(files[0]).values[0]
        # Agri
        landcover[np.all((landcover>=10,landcover<50),axis=0)]=1
        # Forest
        landcover[np.all((landcover>=50,landcover<110),axis=0)]=2
        landcover[np.any((landcover==160,landcover==170),axis=0)]=2
        # Grass
        landcover[np.any((landcover==110,landcover==130),axis=0)]=3
        # Wetland
        landcover[landcover==180]=4
        # Settlement
        landcover[landcover==190]=5
        # Shrub
        landcover[np.all((landcover>=120,landcover<130),axis=0)]=6
        # Lichens
        landcover[landcover==140]=7
        # Sparse
        landcover[np.all((landcover>=150,landcover<160),axis=0)]=8
        # Bare
        landcover[np.all((landcover>=200,landcover<210),axis=0)]=9
        # Water
        landcover[landcover==210]=10
        # Ice
        landcover[landcover==220]=11
    else:
        landcover = xr.open_rasterio(files[0]).values[0]
    return landcover
