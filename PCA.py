"""
09/11/2018 - JFE
This script performs a PCA to reduce the dimensionality of the
predictors for the pantropical-AGB-LUH study.
The pipeline used to perform the PCA is saved and will be loaded
in the script performing the training.
"""

import xarray as xr #xarray to read all types of formats
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline


### first get datasets
#LUH2 data provided by Uni of Maryland
luh = xr.open_dataset('/disk/scratch/local.2/jexbraya/LUH2/states.nc',decode_times=False)
#worldclim2 data regridded to 0.25x0.25 
wc2 = xr.concat([xr.open_rasterio(f) for f in sorted(glob.glob('/disk/scratch/local.2/jexbraya/WorldClim2/0.25deg/*tif'))],dim='band')
#soilgrids data regridded to 0.25x0.25
soil= xr.concat([xr.open_rasterio(f) for f in sorted(glob.glob('/disk/scratch/local.2/jexbraya/soilgrids/0.25deg/*tif'))],dim='band')

#create the land mask knowing that northernmost pixels (row id 0) are all empty
