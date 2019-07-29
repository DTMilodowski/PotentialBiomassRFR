"""
30/11/2018 - DTM
Create training mask for RFR routine

"""
import numpy as np
import xarray as xr
from scipy import ndimage
import glob
def set(path,subset=1):
    # subset 1: HFL and ESACCI)
    if subset == 1:
        lcfiles = sorted(glob.glob('%s/esacci/*lccs-class*tif' % path))
        lc = xr.open_rasterio(lcfiles[0]).values[0]

        sparsemask = (lc>=120)*(lc<130) + (lc>=140)*(lc<160) + (lc>=200)*(lc<210) # shrub, lichen, sparse, bare
        #sparsemask = (lc>=140)*(lc<160) + (lc>=200)*(lc<210)# lichen, sparse, bare
        forestmask =(lc>=50)*(lc<=90) + (lc==160)  + (lc==170)

        # loop through years and only keep pixels tha do not change from year to year
        for ff in range(len(lcfiles)):
            lc_p = lc.copy()
            lc = xr.open_rasterio(lcfiles[ff]).values[0]
            forestmask *= (lc==lc_p)
            sparsemask *= (lc==lc_p)


        humanmask += np.any((lc==190,np.all((lc>=10,lc<=40),axis=0)),axis=0)
        struct = ndimage.generate_binary_structure(2, 2)
        buffermask = ndimage.binary_dilation(humanmask, structure=struct,iterations=2)
        buffermask = buffermask == False

        # load hinterland forests
        hfl = xr.open_rasterio(glob.glob('%s/forestcover/HFL*tif' % path)[0]).values[0]
        forestmask=forestmask*(hfl==1)
        # merge masks
        training_mask=sparsemask+forestmask
        training_mask*=buffermask

    # subset 2: HFL only
    elif subset == 2:
        # load hinterland forests
        hfl = xr.open_rasterio(glob.glob('%s/forestcover/HFL*tif' % path)[0]).values[0]
        training_mask=(hfl==1)

    # subset 3: Mapbiomas
    elif subset == 3:
        mbfiles = sorted(glob.glob('%s/mapbiomas/*tif' % path))
        mb = xr.open_rasterio(mbfiles[0]).values

        lc = mb[0]
        forest = np.all((lc>=2,lc<=5),axis=0)
        nonforest = np.any((lc==10,lc==11,lc==12),axis=0)
        bare = np.any((lc==23,lc==29,lc==32),axis=0)

        for yy in range(mb.shape[0]):
            lc_p = lc.copy()
            lc = mb[yy]
            forest *= (lc==lc_p)
            nonforest *= (lc==lc_p)
            bare *= (lc==lc_p)

        humanmask = np.any((np.all((lc>=13,lc<=21),axis=0),lc==9,lc==24,lc==30),axis=0)
        struct = ndimage.generate_binary_structure(2, 2)
        buffermask = ndimage.binary_dilation(humanmask, structure=struct,iterations=2)
        buffermask = buffermask == False

        training_mask = forest+nonforest+bare
        training_mask*=buffermask

    # MAPBIOMAS & HFL
    elif subset == 4:
        mbfiles = sorted(glob.glob('%s/mapbiomas/*tif' % path))
        mb = xr.open_rasterio(mbfiles[0]).values

        lc = mb[0]
        forest = np.all((lc>=2,lc<=5),axis=0)
        nonforest = np.any((lc==10,lc==11,lc==12),axis=0)
        bare = np.any((lc==23,lc==29,lc==32),axis=0)
        nodata = lc==0

        for yy in range(mb.shape[0]):
            lc_p = lc.copy()
            lc = mb[yy]
            update = (lc==lc_p)
            forest *= update
            nonforest *= update
            bare *= update
            nodata *= update

        humanmask = np.any((np.all((lc>=13,lc<=21),axis=0),lc==9,lc==24,lc==30),axis=0)

        # load hinterland forests
        hfl = xr.open_rasterio(glob.glob('%s/forestcover/HFL*tif' % path)[0]).values[0]
        hfl_mask=(hfl==1)

        hfl_outside_biomas_extent = hfl_mask*nodata

        # load ESA-CCI data
        lcfiles = sorted(glob.glob('%s/esacci/*lccs-class*tif' % path))
        lc = xr.open_rasterio(lcfiles[0]).values[0]
        esacci_bare = (lc>=200)*(lc<210) + (lc == 220) # bare, ice

        # loop through years and only keep pixels tha do not change from year to year
        for ff in range(len(lcfiles)):
            lc_p = lc.copy()
            lc = xr.open_rasterio(lcfiles[ff]).values[0]
            esacci_bare *= (lc==lc_p)

        humanmask += np.any((lc==190,np.all((lc>=10,lc<=40),axis=0)),axis=0)*nodata

        struct = ndimage.generate_binary_structure(2, 2)
        buffermask = ndimage.binary_dilation(humanmask, structure=struct,iterations=2)
        buffermask = buffermask == False

        esacci_bare_outside_biomas_extent = esacci_bare*nodata


        training_mask = forest*hfl_mask + nonforest + bare + hfl_outside_biomas_extent + esacci_bare_outside_biomas_extent
        training_mask*=buffermask

    # MAPBIOMAS & HFL & ESA-CCI (outside Brazil)
    elif subset == 5:
        mbfiles = sorted(glob.glob('%s/mapbiomas/*tif' % path))
        mb = xr.open_rasterio(mbfiles[0]).values

        lc = mb[0]
        forest = np.all((lc>=2,lc<=5),axis=0)
        nonforest = np.any((lc==10,lc==11,lc==12),axis=0)
        bare = np.any((lc==23,lc==29,lc==32),axis=0)
        nodata = lc==0

        for yy in range(mb.shape[0]):
            lc_p = lc.copy()
            lc = mb[yy]
            update = (lc==lc_p)
            forest *= update
            nonforest *= update
            bare *= update
            nodata *= update

        humanmask = np.any((np.all((lc>=13,lc<=21),axis=0),lc==9,lc==24,lc==30),axis=0)

        # load hinterland forests
        hfl = xr.open_rasterio(glob.glob('%s/forestcover/HFL*tif' % path)[0]).values[0]
        hfl_mask=(hfl==1)

        hfl_outside_biomas_extent = hfl_mask*nodata

        # load ESA-CCI data
        lcfiles = sorted(glob.glob('%s/esacci/*lccs-class*tif' % path))
        lc = xr.open_rasterio(lcfiles[0]).values[0]
        esacci_bare = (lc>=200)*(lc<210) + (lc == 220) # bare, ice
        esacci_natural_non_forest = (lc>=110)*(lc<160) + (lc>=200)*(lc<210) # shrub, lichen, sparse, bare


        # loop through years and only keep pixels tha do not change from year to year
        for ff in range(len(lcfiles)):
            lc_p = lc.copy()
            lc = xr.open_rasterio(lcfiles[ff]).values[0]
            esacci_natural_non_forest *= (lc==lc_p)

        humanmask += np.any((lc==190,np.all((lc>=10,lc<=40),axis=0)),axis=0)*nodata

        struct = ndimage.generate_binary_structure(2, 2)
        buffermask = ndimage.binary_dilation(humanmask, structure=struct,iterations=2)
        buffermask = buffermask == False

        esacci_nf_outside_biomas_extent = esacci_natural_non_forest*nodata


        training_mask = forest*hfl_mask + nonforest + bare + hfl_outside_biomas_extent + esacci_nf_outside_biomas_extent
        training_mask*=buffermask

    # MAPBIOMAS & HFL & ESA-CCI (outside Brazil), non-forest filtered by IUCN
    # protected areas
    elif subset == 6:
        mbfiles = sorted(glob.glob('%s/mapbiomas/*tif' % path))
        mb = xr.open_rasterio(mbfiles[0]).values

        lc = mb[0]
        forest = np.all((lc>=2,lc<=5),axis=0)
        nonforest = np.any((lc==10,lc==11,lc==12),axis=0)
        bare = np.any((lc==23,lc==29,lc==32),axis=0)
        nodata = lc==0

        for yy in range(mb.shape[0]):
            lc_p = lc.copy()
            lc = mb[yy]
            update = (lc==lc_p)
            forest *= update
            nonforest *= update
            bare *= update
            nodata *= update

        humanmask = np.any((np.all((lc>=13,lc<=21),axis=0),lc==9,lc==24,lc==30),axis=0)

        # load hinterland forests
        hfl = xr.open_rasterio(glob.glob('%s/forestcover/HFL*tif' % path)[0]).values[0]
        hfl_mask=(hfl==1)

        hfl_outside_biomas_extent = hfl_mask*nodata

        # load ESA-CCI data
        lcfiles = sorted(glob.glob('%s/esacci/*lccs-class*tif' % path))
        lc = xr.open_rasterio(lcfiles[0]).values[0]
        esacci_bare = (lc>=200)*(lc<210) + (lc == 220) # bare, ice
        esacci_natural_non_forest = (lc>=110)*(lc<160) + (lc>=200)*(lc<210) # shrub, lichen, sparse, bare


        # loop through years and only keep pixels tha do not change from year to year
        for ff in range(len(lcfiles)):
            lc_p = lc.copy()
            lc = xr.open_rasterio(lcfiles[ff]).values[0]
            esacci_natural_non_forest *= (lc==lc_p)

        humanmask += np.any((lc==190,np.all((lc>=10,lc<=40),axis=0)),axis=0)*nodata

        struct = ndimage.generate_binary_structure(2, 2)
        buffermask = ndimage.binary_dilation(humanmask, structure=struct,iterations=2)
        buffermask = buffermask == False

        esacci_nf_outside_biomas_extent = esacci_natural_non_forest*nodata

        # load IUCN protected areas (gridded)
        pa = xr.open_rasterio(glob.glob('%s/IUCN_protected_areas/*tif' % path)[0]).values[0]
        pa_mask = pa==1


        training_mask = forest*hfl_mask + nonforest + bare*pa_mask + hfl_outside_biomas_extent + esacci_nf_outside_biomas_extent*pa_mask
        training_mask*=buffermask



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
        training_flag=np.zeros(lc.shape)

        sparsemask = (lc>=120)*(lc<130) + (lc>=140)*(lc<160) + (lc>=200)*(lc<210) # shrub, lichen, sparse, bare
        #sparsemask = (lc>=140)*(lc<160) + (lc>=200)*(lc<210)# lichen, sparse, bare
        forestmask =(lc>=50)*(lc<=90) + (lc==160)  + (lc==170)

        # loop through years and only keep pixels tha do not change from year to year
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

def get_stable_forest_outside_training_areas(path,trainmask_init,landmask,method = 1):

    # ESA-CCI basis
    if method == 1:
        lcfiles = sorted(glob.glob('%s/esacci/*lccs-class*tif' % path))
        lc = xr.open_rasterio(lcfiles[0]).values[0]
        forestmask =(lc>=50)*(lc<=90) + (lc==160)  + (lc==170)

        # loop through years and only keep pixels that do not change from year to year
        for ff in range(len(lcfiles)):
            lc_p = lc.copy()
            lc = xr.open_rasterio(lcfiles[ff]).values[0]
            forestmask *= (lc==lc_p)

    # MapBiomas basis
    elif method == 2:
        mbfiles = sorted(glob.glob('%s/mapbiomas/*tif' % path))
        mb = xr.open_rasterio(mbfiles[0]).values

        lc = mb[0]
        forestmask = np.all((lc>=2,lc<=5),axis=0)

        for yy in range(mb.shape[0]):
            lc_p = lc.copy()
            lc = mb[yy]
            update = (lc==lc_p)
            forestmask *= update

    # MapBiomas basis with buffer around agriculture pasture and urban areas
    # (3km - see Chaplin-Kramer et al, Nature Comm., 2015)
    elif method == 3:
        mbfiles = sorted(glob.glob('%s/mapbiomas/*tif' % path))
        mb = xr.open_rasterio(mbfiles[0]).values

        lc = mb[0]
        forestmask = np.all((lc>=2,lc<=5),axis=0)

        for yy in range(mb.shape[0]):
            lc_p = lc.copy()
            lc = mb[yy]
            update = (lc==lc_p)
            forestmask *= update

        humanmask = np.any((np.all((lc>=13,lc<=21),axis=0),lc==9,lc==24,lc==30),axis=0)
        struct = ndimage.generate_binary_structure(2, 2)
        buffermask = ndimage.binary_dilation(humanmask, structure=struct,iterations=2)
        buffermask = buffermask == False

        forestmask*=buffermask

    # MapBiomas & ESA-CCI outside of Brazil
    elif method == 4:
        mbfiles = sorted(glob.glob('%s/mapbiomas/*tif' % path))
        mb = xr.open_rasterio(mbfiles[0]).values

        lc = mb[0]
        forestmask = np.all((lc>=2,lc<=5),axis=0)
        nodata = lc==0

        for yy in range(mb.shape[0]):
            lc_p = lc.copy()
            lc = mb[yy]
            update = (lc==lc_p)
            forestmask *= update
            nodata *= update

        lcfiles = sorted(glob.glob('%s/esacci/*lccs-class*tif' % path))
        lc = xr.open_rasterio(lcfiles[0]).values[0]
        lcmask =(lc>=50)*(lc<=90) + (lc==160)  + (lc==170) + (lc>=120)*(lc<130) + (lc>=140)*(lc<160) + (lc>=200)*(lc<210)

        # loop through years and only keep pixels that do not change from year to year
        for ff in range(len(lcfiles)):
            lc_p = lc.copy()
            lc = xr.open_rasterio(lcfiles[ff]).values[0]
            lcmask *= (lc==lc_p)

        forestmask = forestmask + lcmask*nodata

    # MapBiomas & ESA-CCI outside of Brazil with buffer around agriculture
    # pasture and urban areas (3km - see Chaplin-Kramer et al, Nature Comm.,
    # 2015)
    elif method == 5:
            mbfiles = sorted(glob.glob('%s/mapbiomas/*tif' % path))
            mb = xr.open_rasterio(mbfiles[0]).values

            lc = mb[0]
            forestmask = np.all((lc>=2,lc<=5),axis=0)
            nodata = lc==0

            for yy in range(mb.shape[0]):
                lc_p = lc.copy()
                lc = mb[yy]
                update = (lc==lc_p)
                forestmask *= update
                nodata *= update

            humanmask = np.any((np.all((lc>=13,lc<=21),axis=0),lc==9,lc==24,lc==30),axis=0)

            lcfiles = sorted(glob.glob('%s/esacci/*lccs-class*tif' % path))
            lc = xr.open_rasterio(lcfiles[0]).values[0]
            lcmask =(lc>=50)*(lc<=90) + (lc==160)  + (lc==170) + (lc>=120)*(lc<130) + (lc>=140)*(lc<160) + (lc>=200)*(lc<210)

            # loop through years and only keep pixels that do not change from year to year
            for ff in range(len(lcfiles)):
                lc_p = lc.copy()
                lc = xr.open_rasterio(lcfiles[ff]).values[0]
                lcmask *= (lc==lc_p)

            humanmask += np.any((lc==190,np.all((lc>=10,lc<=40),axis=0)),axis=0)*nodata
            struct = ndimage.generate_binary_structure(2, 2)
            buffermask = ndimage.binary_dilation(humanmask, structure=struct,iterations=2)
            buffermask = buffermask == False

            forestmask = forestmask + lcmask*nodata
            forestmask *= buffermask

    # mask out initial training dataset from stable forest
    return forestmask*(trainmask_init!=1)*landmask
