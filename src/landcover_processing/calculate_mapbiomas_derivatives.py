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
import raster_io as io
import land_cover_functions as lcf
import cython_functions as cf
"""
PART A: LOAD MAPBIOMAS DATA
--------------------------------------------------------------------------------
"""
year = int(sys.argv[1])
tstep = year-1985
mb = lcf.load_mapbiomas(timestep=tstep,aggregate=3)
n_years = mb.shape[0]
"""
PART B: DISTANCE TO NONFOREST (AND SIMILAR)
--------------------------------------------------------------------------------
"""
nonforest_mask = np.all((np.isfinite(mb),mb!=1),axis=0)
human_mask = np.any((mb==4,mb==7,mb==8,mb==9,mb==10),axis=0)
nodata_mask = ~np.isfinite(mb)

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
io.write_xarray_to_GeoTiff(dist2nf,'mapbiomas_dist2nonforest_%i' % year)

dist2hm = io.copy_xarray_template(template)
dist2hm.values = distance_to_human.copy()
io.write_xarray_to_GeoTiff(dist2hm,'mapbiomas_dist2human_%i' % year)
"""
tr = template.attrs['transform']
transform = Affine(tr[0],tr[1],tr[2],tr[3],tr[4],tr[5])
nx, ny = template.sizes['x'], template.sizes['y']
col,row = np.meshgrid(np.arange(nx)+0.5, np.arange(ny)+0.5)
lon, lat = transform * (col,row)
coords = {'lat': (['lat'],lat[:,0],{'units':'degrees_north','long_name':'latitude'}),
          'lon': (['lon'],lon[0,:],{'units':'degrees_east','long_name':'longitude'})}

attrs={'_FillValue':-9999.,'units':'~1 km pixels'}
data_vars = {}
ny=mb.shape[0];nx=mb.shape[1]
data_vars['Distance_to_nonforest'] = (['lat','lon'],np.zeros([ny,nx])*np.nan,attrs)
data_vars['Distance_to_human'] = (['lat','lon'],np.zeros([ny,nx])*np.nan,attrs)

ds = xr.Dataset(data_vars=data_vars,coords=coords)
ds['Distance_to_nonforest'].values = distance_to_nonforest.copy()
ds['Distance_to_human'].values = distance_to_human.copy()

#save to a nc file
nc_file = 'mapbiomas_deforestation_test_%i.nc' % year
ds.to_netcdf(path=nc_file)
"""
