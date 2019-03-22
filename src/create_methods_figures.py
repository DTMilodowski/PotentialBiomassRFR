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
soil = ListedColormap(sns.cubehelix_palette(20, start=22, rot=0.1))
greens = ListedColormap(sns.cubehelix_palette(8, start=2, rot=0, dark=0.4, light=.95))

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

#---------------------------
# Plot some SoilGrids data
path2sg = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/BRA/soilgrids/'

filename = '%sBDRICM_M_1km_ll_BRA.tif' % path2sg
savefile = '../figures/BDRICM_M_1km_ll_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values = ds.values.astype('float')
ds.values[ds.values==ds.nodatavals]=np.nan
fig_sg01 = sfig.plot_xarray(ds,savefile,cmap=soil)

filename = '%sSNDPPT_M_sl1_1km_ll_BRA.tif' % path2sg
savefile = '../figures/SNDPPT_M_sl1_1km_ll_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values = ds.values.astype('float')
ds.values[ds.values==ds.nodatavals]=np.nan
fig_sg02 = sfig.plot_xarray(ds,savefile,cmap=soil)

filename = '%sSLTPPT_M_sl1_1km_ll_BRA.tif' % path2sg
savefile = '../figures/SLTPPT_M_sl1_1km_ll_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values = ds.values.astype('float')
ds.values[ds.values==ds.nodatavals]=np.nan
fig_sg03 = sfig.plot_xarray(ds,savefile,cmap=soil)

filename = '%sCLYPPT_M_sl1_1km_ll_BRA.tif' % path2sg
savefile = '../figures/CLYPPT_M_sl1_1km_ll_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values = ds.values.astype('float')
ds.values[ds.values==ds.nodatavals]=np.nan
fig_sg04 = sfig.plot_xarray(ds,savefile,cmap=soil)
ds_temp=ds.copy()
#----------------------------
# Plot landcover data
filename =  '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/BRA/forestcover/HFL_2013_BRA.tif'
savefile = '../figures/HFL_2013_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values = ds.values.astype('float')
ds.values[ds.values==ds.nodatavals]=0
ds.values[np.isnan(ds_temp.values)]=np.nan
fig_lc01 = sfig.plot_xarray(ds,savefile,cmap=greens)

path2lc = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/BRA/esacci/'
filename = '%sESACCI-LC-L4-LCCS-Map-P1Y-2008-v2.0.7-1km-mode-lccs-class-BRA.tif' % path2lc
savefile = '../figures/ESACCI_2008_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values = sfig.aggregate_classes(ds.values)
lc,lc_labs=sfig.esa_cci_colormap(ds.values)
fig_lc02 = sfig.plot_xarray(ds,savefile,cmap=lc)

filename = '%sESACCI-LC-L4-LCCS-Map-P1Y-2009-v2.0.7-1km-mode-lccs-class-BRA.tif' % path2lc
savefile = '../figures/ESACCI_2009_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values = sfig.aggregate_classes(ds.values)
fig_lc03 = sfig.plot_xarray(ds,savefile,cmap=lc)

filename = '%sESACCI-LC-L4-LCCS-Map-P1Y-2010-v2.0.7-1km-mode-lccs-class-BRA.tif' % path2lc
savefile = '../figures/ESACCI_2010_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values = sfig.aggregate_classes(ds.values)
fig_lc04 = sfig.plot_xarray(ds,savefile,cmap=lc)

filename = '%sESACCI-LC-L4-LCCS-Map-P1Y-2011-v2.0.7-1km-mode-lccs-class-BRA.tif' % path2lc
savefile = '../figures/ESACCI_2011_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values = sfig.aggregate_classes(ds.values)
fig_lc05 = sfig.plot_xarray(ds,savefile,cmap=lc)

#-------------------------------------
# Plot biomass
filename =  '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/BRA/agb/Avitabile_AGB_BRA_1km.tif'
savefile = '../figures/AGB_Avitabile_BRA.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds.values = ds.values.astype('float')
ds.values[ds.values<0]=np.nan
fig_obs = sfig.plot_xarray(ds,savefile,cmap='viridis',vmin=0,vmax=400)

# Plot potential biomass
filename =  '../output/BRA_003_AGB_potential_RFR_worldclim_soilgrids.nc'
savefile = '../figures/AGBpot_Avitabile_BRA.png'
ds=xr.open_dataset(filename)
ds['AGBpot3'].values = ds['AGBpot3'].values.astype('float')
ds['AGBpot3'].values[ds['AGBpot3'].values<0]=np.nan
fig_pot = sfig.plot_xarray(ds['AGBpot3'],savefile,cmap='viridis',vmin=0,vmax=400)

#

# Plot restoration potential
savefile = '../figures/AGBseq_Avitabile_BRA.png'
seq=(ds['AGBpot3']-ds['AGBobs'])/2.
fig_seq = sfig.plot_xarray(seq,savefile,cmap='plasma',vmin=0,vmax=100)


savefile = '../figures/AGBseq_Brazil.png'
cbar_kwargs={'label': 'AGB$_{seq}$ / Mg(C) ha$^{-1}$','orientation':'horizontal'}
fig_indo_seq,ax = sfig.plot_xarray(seq,savefile,cmap='plasma',extend ='max',
                                vmin=0,vmax=125,cbar_kwargs=cbar_kwargs,
                                figsize_x=8,figsize_y = 8,add_colorbar =True,
                                title= 'Sequestration potential')

## "Key countries"
# Indonesia
area_file = '/home/dmilodow/DataStore_DTM/EOlaboratory/EOlab/IndonesiaPotentialAGB/indonesia_cell_areas_data.tif'
areas = xr.open_rasterio(area_file).sel(band=1)

filename =  '/home/dmilodow/DataStore_DTM/EOlaboratory/EOlab/IndonesiaPotentialAGB/indonesia_AGBobs_data.tif'
savefile = '../figures/AGBobs_Indo.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds = ds.rename(x='longitude',y='latitude')
ds.values = ds.values.astype('float')
ds.values[ds.values==ds.nodatavals]=np.nan
cbar_kwargs={'label': 'AGB$_{obs}$ / Mg(C) ha$^{-1}$','orientation':'vertical'}
fig_indo_obs,ax = sfig.plot_xarray(ds,savefile,cmap='viridis',extend ='max',
                                vmin=0,vmax=250,cbar_kwargs=cbar_kwargs,
                                figsize_x=16,figsize_y = 6,add_colorbar =True,
                                title= 'Observed AGB (mid 2000s)')

filename =  '/home/dmilodow/DataStore_DTM/EOlaboratory/EOlab/IndonesiaPotentialAGB/indonesia_AGBpot_data.tif'
savefile = '../figures/AGBpot_Indo.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds = ds.rename(x='longitude',y='latitude')
ds.values = ds.values.astype('float')
ds.values[ds.values==ds.nodatavals]=np.nan
cbar_kwargs={'label': 'AGB$_{pot}$ / Mg(C) ha$^{-1}$','orientation':'vertical'}
fig_indo_pot,ax = sfig.plot_xarray(ds,savefile,cmap='viridis',extend ='max',
                                vmin=0,vmax=250,cbar_kwargs=cbar_kwargs,
                                figsize_x=16,figsize_y = 6,add_colorbar =True,
                                title= 'Potential AGB')

filename =  '/home/dmilodow/DataStore_DTM/EOlaboratory/EOlab/IndonesiaPotentialAGB/indonesia_AGBseq_data.tif'
savefile = '../figures/AGBseq_Indo.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds = ds.rename(x='longitude',y='latitude')
ds.values = ds.values.astype('float')
ds.values[ds.values==ds.nodatavals]=np.nan
cbar_kwargs={'label': 'AGB$_{seq}$ / Mg(C) ha$^{-1}$','orientation':'vertical'}
fig_indo_seq,ax = sfig.plot_xarray(ds,savefile,cmap='plasma',extend ='max',
                                vmin=0,vmax=125,cbar_kwargs=cbar_kwargs,
                                figsize_x=16,figsize_y = 6,add_colorbar =True,
                                title= 'Sequestration potential')

# Ghana
area_file = '/home/dmilodow/DataStore_DTM/EOlaboratory/EOlab/GhanaPotentialAGB/ghana_cell_areas_data.tif'
areas = xr.open_rasterio(area_file).sel(band=1)

filename =  '/home/dmilodow/DataStore_DTM/EOlaboratory/EOlab/GhanaPotentialAGB/ghana_AGBobs_data.tif'
savefile = '../figures/AGBobs_Ghana.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds = ds.rename(x='longitude',y='latitude')
ds.values = ds.values.astype('float')
ds.values[ds.values==ds.nodatavals]=np.nan
cbar_kwargs={'label': 'AGB$_{obs}$ / Mg(C) ha$^{-1}$','orientation':'vertical'}
fig_indo_obs,ax = sfig.plot_xarray(ds,savefile,cmap='viridis',extend ='max',
                                vmin=0,vmax=250,cbar_kwargs=cbar_kwargs,
                                figsize_x=16,figsize_y = 6,add_colorbar =True,
                                title= 'Observed AGB (mid 2000s)')

filename =  '/home/dmilodow/DataStore_DTM/EOlaboratory/EOlab/GhanaPotentialAGB/ghana_AGBpot_data.tif'
savefile = '../figures/AGBpot_Ghana.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds = ds.rename(x='longitude',y='latitude')
ds.values = ds.values.astype('float')
ds.values[ds.values==ds.nodatavals]=np.nan
cbar_kwargs={'label': 'AGB$_{pot}$ / Mg(C) ha$^{-1}$','orientation':'vertical'}
fig_indo_pot,ax = sfig.plot_xarray(ds,savefile,cmap='viridis',extend ='max',
                                vmin=0,vmax=250,cbar_kwargs=cbar_kwargs,
                                figsize_x=16,figsize_y = 6,add_colorbar =True,
                                title= 'Potential AGB')

filename =  '/home/dmilodow/DataStore_DTM/EOlaboratory/EOlab/GhanaPotentialAGB/ghana_AGBseq_data.tif'
savefile = '../figures/AGBseq_Ghana.png'
ds=xr.open_rasterio(filename).sel(band=1)
ds = ds.rename(x='longitude',y='latitude')
ds.values = ds.values.astype('float')
ds.values[ds.values==ds.nodatavals]=np.nan
cbar_kwargs={'label': 'AGB$_{seq}$ / Mg(C) ha$^{-1}$','orientation':'vertical'}
fig_indo_seq,ax = sfig.plot_xarray(ds,savefile,cmap='plasma',extend ='max',
                                vmin=0,vmax=125,cbar_kwargs=cbar_kwargs,
                                figsize_x=16,figsize_y = 6,add_colorbar =True,
                                title= 'Sequestration potential')
