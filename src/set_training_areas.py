"""
30/11/2018 - DTM
Create training mask for RFR routine

"""
import numpy as np
import xarray as xr
import glob
def set(path,subset=1):
    # subset 1: HFL and ESACCI)
    if subset == 1:
        lcfiles = sorted(glob.glob('%s/esacci/*lccs-class*tif' % path))
        lc = xr.open_rasterio(lcfiles[0]).values[0]
        forestmask=np.ones(lc.shape)
        sparsemask=np.ones(lc.shape)

        sparsemask = (lc>=120)*(lc<130) + (lc>=140)*(lc<160) + (lc>=200)*(lc<210) # shrub, lichen, sparse, bare
        #sparsemask = (lc>=140)*(lc<160) + (lc>=200)*(lc<210)# lichen, sparse, bare
        forestmask =(lc>=50)*(lc<=90) + (lc==160)  + (lc==170)

        # loop through years and only keep pixels tha do not change from year to year
        lc_start = 1993
        lc_end = 2015
        for ff in range(len(lcfiles)):
            lc_p = lc.copy()
            lc = xr.open_rasterio(lcfiles[ff]).values[0]
            forestmask *= (lc==lc_p)
            sparsemask *= (lc==lc_p)
        # load hinterland forests
        hfl = xr.open_rasterio(glob.glob('%s/forestcover/HFL*tif' % path)[0]).values[0]
        forestmask=forestmask*(hfl==1)
        # merge masks
        training_mask=sparsemask+forestmask

    # subset 2: HFL only
    elif subset == 2:
        # load hinterland forests
        hfl = xr.open_rasterio(glob.glob('%s/forestcover/HFL*tif' % path)[0]).values[0]
        training_mask=(hfl==1)

    return training_mask

# A function that revises the training areas to account for forested areas that
# are not well represented in parameter space
# Simple criteria for additional training data:
# 1) continuous forest (based on ESA-CCI)
# 2) AGBpot < AGBobs
# returns both the mask for training, and a coded array indicating whether the
# training data are part of the original training set (1 = intact forest, 2 =
# sparse), or are additional forest pixels (3 = additional forest)
def set_revised(path,AGBobs,AGBpot,landmask):

        lcfiles = sorted(glob.glob('%s/esacci/*lccs-class*tif' % path))
        lc = xr.open_rasterio(lcfiles[0]).values[0]
        forestmask=np.ones(lc.shape)
        sparsemask=np.ones(lc.shape)
        training_flag=np.zeros(lc.shape)

        sparsemask = (lc>=120)*(lc<130) + (lc>=140)*(lc<160) + (lc>=200)*(lc<210) # shrub, lichen, sparse, bare
        #sparsemask = (lc>=140)*(lc<160) + (lc>=200)*(lc<210)# lichen, sparse, bare
        forestmask =(lc>=50)*(lc<=90) + (lc==160)  + (lc==170)

        # loop through years and only keep pixels tha do not change from year to year
        lc_start = 1993
        lc_end = 2015

        for ff in range(len(lcfiles)):
            lc_p = lc.copy()
            lc = xr.open_rasterio(lcfiles[ff]).values[0]
            forestmask *= (lc==lc_p)
            sparsemask *= (lc==lc_p)

        # load hinterland forests
        hfl = xr.open_rasterio(glob.glob('%s/forestcover/HFL*tif' % path)[0]).values[0]
        forestmask_init=forestmask*(hfl==1)
        forestmask_added = forestmask*(hfl!=1)

        forestmask_added = forestmask_added*(AGBpot<AGBobs)

        training_flag[forestmask_init] = 1
        training_flag[sparsemask] = 2
        training_flag[forestmask_added] = 3

        # merge masks
        training_mask=(sparsemask+forestmask_init+forestmask_added)*landmask

        return training_mask, training_flag

def get_stable_forest_outside_training_areas(path,trainmask_init,landmask):

        lcfiles = sorted(glob.glob('%s/esacci/*lccs-class*tif' % path))
        lc = xr.open_rasterio(lcfiles[0]).values[0]
        forestmask=np.ones(lc.shape)
        training_flag=np.zeros(lc.shape)

        forestmask =(lc>=50)*(lc<=90) + (lc==160)  + (lc==170)

        # loop through years and only keep pixels tha do not change from year to year
        lc_start = 1993
        lc_end = 2015

        for ff in range(len(lcfiles)):
            lc_p = lc.copy()
            lc = xr.open_rasterio(lcfiles[ff]).values[0]
            forestmask *= (lc==lc_p)

        # mask out initial training dataset from stable forest
        return forestmask*(trainmask_init!=1)*landmask
