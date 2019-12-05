"""
LAND_COVER_FUNCTIONS.PY
--------------------------------------------------------------------------------
A set of functions to process land cover datasets to derive informative data
layers from land cover and related databases
--------------------------------------------------------------------------------
"""
# Import Libraries
import numpy as np
from scipy import spatial
import xarray as xr
import glob
"""
create_KDTree
-------------
"""
def create_KDTree(pts,max_pts_per_tree = 10**6):
    npts = pts.shape[0]
    ntrees = int(np.ceil(npts/float(max_pts_per_tree)))
    trees = []
    starting_ids = []
    print('Building %i kd-trees' % ntrees)
    for tt in range(0,ntrees):
        print('\t- tree %i of %i\r' % (tt+1,ntrees),end='\r')
        i0=tt*max_pts_per_tree
        i1 = (tt+1)*max_pts_per_tree
        if i1 < pts.shape[0]:
            trees.append(spatial.cKDTree(pts[i0:i1,0:2],leafsize=32,balanced_tree=True))
        else:
            trees.append(spatial.cKDTree(pts[i0:,0:2],leafsize=32,balanced_tree=True))
        starting_ids.append(i0)

    return np.asarray(starting_ids,dtype='int'), trees


def build_trees(mask_array,dx,dy):
    y = np.arange(mask_array.shape[0])*dy
    x = np.arange(mask_array.shape[1])*dx
    xx,yy = np.meshgrid(x,y)
    xx_1 = xx[mask_array==1]
    yy_1 = yy[mask_array==1]
    pts = np.array([xx_1,yy_1]).transpose()

    # put coordinates into a kd-tree for efficient spatial searching
    starting_ids, trees = create_KDTree(pts)
    return trees


"""
distance_to_mask
----------------
This function takes an binary mask and finds the distance of each pixel to the
nearest pixel with value == 1. If NaN values are present, these pixels are
ignored.
Optional arguments dx and dy are there so that the resolution of the raster can
be accounted for if desired
"""
def distance_to_mask(mask_array,dx=1,dy=1):
    # get coordinates of pixels in mask==1
    y = np.arange(mask_array.shape[0])*dy
    x = np.arange(mask_array.shape[1])*dx
    xx,yy = np.meshgrid(x,y)
    xx_1 = xx[mask_array==1]
    yy_1 = yy[mask_array==1]
    pts = np.array([xx_1,yy_1]).transpose()

    # put coordinates into a kd-tree for efficient spatial searching
    starting_ids, trees = create_KDTree(pts)
    N_trees = len(trees)

    # now loop through mask and get distance to nearest neighbour from kd-tree
    print('Calculating distance to specified mask...')
    distance = np.zeros(mask_array.shape)*np.nan
    count = 0
    print_at = 10000
    interval = 10000
    for ii in range(0,y.size):
        for jj in range(0,x.size):
            if count==print_at:
                print('\t{0:.3f}%\r'.format(float(ii*x.size+jj)/float((y.size*x.size))*100),end='\r')
                print_at = print_at+interval
            if np.isfinite(mask_array[ii,jj]):
                if mask_array[ii,jj]==1:
                    distance[ii,jj]=0
                else:
                    distance[ii,jj],temp=trees[0].query([x[jj],y[ii]],k=1)
                    for tt in range(1,N_trees):
                        distance_iter,temp = trees[tt].query([x[jj],y[ii]],k=1)
                        distance[ii,jj]=np.min([distance[ii,jj],distance_iter])
            count = count +1
    return distance

"""
neighbourhood_pixel_counts
--------------------------
This function takes a binary mask, and finds the number of pixels with value==1
within a defined radius. The radius should be specified in the same units as
optional arguments dx and dy
Optional arguments:
- dx (default = 1 i.e. deal in pixels)
- dy  (default = 1 i.e. deal in pixels)
to provide the area contributed by the pixels in the neighbourhood, rather than
a simple count
"""
def neighbourhood_pixel_counts(mask_array,radius,dx=1,dy=1):
    # get coordinates of pixels in mask==1
    y = np.arange(mask_array.shape[0])*dy
    x = np.arange(mask_array.shape[1])*dx
    xx,yy = np.meshgrid(x,y)
    xx_1 = xx[mask_array==1]
    yy_1 = yy[mask_array==1]
    pts = np.array([xx_1,yy_1]).transpose()

    # put coordinates into a kd-tree for efficient spatial searching
    starting_ids, trees = create_KDTree(pts)
    N_trees = len(trees)

    # now loop through mask and get number of neighbours within specified radius
    counts = np.zeros(mask_array.shape)
    for ii in range(0,y.size):
        for jj in range(0,x.size):
            if np.isfinite(mask_array[ii,jj]):
                #print(N_trees)
                for tt in range(0,N_trees):
                    counts[ii,jj] += len(trees[tt].query_ball_point([x[jj],y[ii]],radius))
            else:
                counts[ii,jj]=np.nan
    counts
    return counts


"""
load_mapbiomas
--------------------------------------------------------------------------------
load in mapbiomas for a given timestep
Timestep 0 -> 1985
Timestep -1 -> 2017
"""
def load_mapbiomas(timestep=-1,aggregate=0):
    path = '/disk/scratch/local.2/PotentialBiomass/processed/BRA/'
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
    # option 3 -> aggregation to 11 classes:
    # Forest formation, Savanna formation, Mangrove, Plantation, Wetland,
    #       Grassland, Pasture, Agriculture, Mosaic agri-pasture,Urban, Other
    elif aggregate == 3:
        mb = xr.open_rasterio(mbfiles[0]).values[timestep]
        lc = np.zeros(mb.shape)*np.nan
        lc[mb==3] = 1                                                # Forest formation
        lc[mb==4] = 2                                                # Savanna formation
        lc[mb==5] = 3                                                # Mangrove
        lc[mb==9] = 4                                                # Plantation
        lc[mb==11] = 5                                               # Wetland
        lc[mb==12] = 6                                               # Grassland
        lc[mb==15] = 7                                               # Pasture
        lc[np.any((mb==19,mb==20),axis=0)] = 8                      # Agriculture
        lc[mb==21] = 9                                               # Mosaic Agriculture-Pasture
        lc[mb==24] = 10                                              # Urban
        lc[np.any((mb==23,mb==29,mb==30,mb==32,mb==13),axis=0)] = 11 # Other
    else:
        mb = xr.open_rasterio(mbfiles[0]).values[timestep]
        lc = mb.copy()
    return lc


"""
load_esacci
--------------------------------------------------------------------------------
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
"""
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
