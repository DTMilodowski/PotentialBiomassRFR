"""
create_methods_figures.py
--------------------------------------------------------------------------------
GENERATE FIGURES REQUIRED FOR METHODS DESCRIPTION
This is a set of scripts to produce figures used to descript potential Biomass
methodology

21/03/2019 - D. T. Milodowski
--------------------------------------------------------------------------------
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import simple_figures as sfig
from matplotlib.colors import ListedColormap

# some colormaps
warm = ListedColormap(sns.cubehelix_palette(20))
wet = ListedColormap(sns.cubehelix_palette(20, start=.5, rot=-.75))
#---------------------------
# Plot some Worldclim2 data
path2wc = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/BRA/wc2/'
# Annual mean temperature
filename = '%swc2.0_bio_30s_01_BRA.tif' % path2wc
savefile = '../figures/wc2.0_bio_30s_01_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values[ds.values==ds.nodatavals]=np.nan
fig_wc01 = sfig.plot_xarray(ds,savefile,cmap=warm)

# Diurnal temp range
filename = '%swc2.0_bio_30s_02_BRA.tif' % path2wc
savefile = '../figures/wc2.0_bio_30s_02_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values[ds.values==ds.nodatavals]=np.nan
fig_wc02 = sfig.plot_xarray(ds,savefile,cmap=warm)

# seasonality
filename = '%swc2.0_bio_30s_04_BRA.tif' % path2wc
savefile = '../figures/wc2.0_bio_30s_04_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values[ds.values==ds.nodatavals]=np.nan
fig_wc04 = sfig.plot_xarray(ds,savefile,cmap=warm)

# max temperature
filename = '%swc2.0_bio_30s_05_BRA.tif' % path2wc
savefile = '../figures/wc2.0_bio_30s_05_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values[ds.values==ds.nodatavals]=np.nan
fig_wc05 = sfig.plot_xarray(ds,savefile,cmap=warm)

# min temperature
filename = '%swc2.0_bio_30s_06_BRA.tif' % path2wc
savefile = '../figures/wc2.0_bio_30s_06_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values[ds.values==ds.nodatavals]=np.nan
fig_wc06 = sfig.plot_xarray(ds,savefile,cmap=warm)

# Annual range
filename = '%swc2.0_bio_30s_07_BRA.tif' % path2wc
savefile = '../figures/wc2.0_bio_30s_07_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values[ds.values==ds.nodatavals]=np.nan
fig_wc07 = sfig.plot_xarray(ds,savefile,cmap=warm)


# Annual precip
filename = '%swc2.0_bio_30s_12_BRA.tif' % path2wc
savefile = '../figures/wc2.0_bio_30s_12_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values[ds.values==ds.nodatavals]=np.nan
fig_wc12 = sfig.plot_xarray(ds,savefile,cmap=wet)

# Seasonality
filename = '%swc2.0_bio_30s_15_BRA.tif' % path2wc
savefile = '../figures/wc2.0_bio_30s_15_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values[ds.values==ds.nodatavals]=np.nan
fig_wc15 = sfig.plot_xarray(ds,savefile,cmap=wet)
