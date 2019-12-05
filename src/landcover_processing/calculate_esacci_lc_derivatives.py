"""
CALCULATE_MAPBIOMAS_DERIVATIVES.PY
================================================================================
This script takes land use data from the Mapbiomas land use
database and derives the following spatial-temporal products
1) Distance from non-forest
2) Distance to human activity
Takes one command line argument (year)
--------------------------------------------------------------------------------
"""
# Import libraries
import numpy as np
import xarray as xr
from scipy import ndimage
from affine import Affine
import sys
sys.path.append('../')
import useful as useful
import raster_io as io
import land_cover_functions as lcf
import cython_functions as cf
"""
PART A: LOAD ESACCI DATA
--------------------------------------------------------------------------------
"""
year = int(sys.argv[1])
country_code = sys.argv[2]
lc = useful.load_esacci(country_code,year=year,aggregate=1)
n_years = lc.shape[0]
"""
PART B: DISTANCE TO NONFOREST (AND SIMILAR)
--------------------------------------------------------------------------------
"""
nonforest_mask = np.all((np.isfinite(lc),lc!=2),axis=0)
human_mask = np.any((lc==1,lc==5),axis=0)
nodata_mask = ~np.isfinite(lc)

print("Building trees\t")
nonforest_boundary = nonforest_mask.astype(int)-ndimage.binary_erosion(nonforest_mask).astype(int)
human_boundary = human_mask.astype(int)-ndimage.binary_erosion(human_mask).astype(int)
trees_nf = lcf.build_trees(nonforest_boundary,1,1)
trees_human = lcf.build_trees(human_boundary,1,1)

print("calculating distance to nonforest pixels")
distance_to_nonforest = cf.distance_to_mask(nonforest_mask.astype(int),nodata_mask.astype(int),1,1,trees_nf)
print("calculating distance to human pixels")
distance_to_human = cf.distance_to_mask(human_mask.astype(int),nodata_mask.astype(int),1,1,trees_human)

"""
PART C: WRITE OUTPUTS TO GEOTIFF FILE
--------------------------------------------------------------------------------
"""
print('writing output layers')
# load template
template_file = '/disk/scratch/local.2/PotentialBiomass/processed/BRA/mapbiomas/BRAZIL_mapbiomas_1km.tif'
template = xr.open_rasterio(template_file)[0]

dist2nf = io.copy_xarray_template(template)
dist2nf.values = distance_to_nonforest.copy()
io.write_xarray_to_GeoTiff(dist2nf,'%s_esacci_dist2nonforest_%i' % (country_code,year))

dist2hm = io.copy_xarray_template(template)
dist2hm.values = distance_to_human.copy()
io.write_xarray_to_GeoTiff(dist2hm,'%s_esacci_dist2human_%i' % (country_code,year))
