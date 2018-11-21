"""
21/11/2018 - JFE
this scripts plots maps showing the 21st century evol of AGB according to LUH2
dataset for 6 ssp scenarios for the whole region
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from useful import *
from mpl_toolkits.axes_grid1 import AxesGrid
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#get areas and landmask to calculate continental numbers
areas = get_areas()
_,landmask = get_predictors()

#create the figure / ax
fig=plt.figure('bars');fig.clf()
ax = fig.add_subplot(111)

#load agb
med = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Map_0.25d.tif')
med.values[med.values == med.nodatavals[0]] = np.nan
med[0].values[~landmask] = np.nan

#load uncertainty to write some stats
unc = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Uncertainty_0.25d.tif')

#create 2d array to separate per continent
lon2d,lat2d = np.meshgrid(med.x,med.y)

mask_pantrop = np.ones(lon2d.shape,dtype='bool')
mask_america = lon2d<-25.
mask_africa  = (lon2d>-25.) & (lon2d<58)
mask_asia    = lon2d>58.


#colorblind friendly figures
cols = {'ssp126': np.array([230,159,0])/255.,
        'ssp434': np.array([86,180,233])/255.,
        'ssp245': np.array([0,158,115])/255.,
        'ssp460': np.array([240,228,66])/255.,
        'ssp370': np.array([0,114,178])/255.,
        'ssp585': np.array([213,94,0])/255.}
#loop over the different scenarios
scenlong = {'ssp126': 'SSP1-2.6',
            'ssp434': 'SSP4-3.4',
            'ssp245': 'SSP2-4.5',
            'ssp460': 'SSP4-6.0',
            'ssp370': 'SSP3-7.0',
            'ssp585': 'SSP5-8.5'}

#create an axes grid
fig = plt.figure('all maps',figsize=(12,8));fig.clf()

#create a figure using the axesgrid to make the colorbar fit on the axis
projection = ccrs.PlateCarree()
axes_class = (GeoAxes,dict(map_projection=projection))

axgr = AxesGrid(fig,111,nrows_ncols=(3,2),axes_class=axes_class,label_mode='',cbar_mode='single',cbar_pad = 0.25,cbar_size="3%",axes_pad=.75)


for sc,scen in enumerate(['ssp126','ssp434','ssp245','ssp460','ssp370','ssp585']):

    ssp = xr.open_dataset('/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/output/AGB_%s.nc' % scen)
    toplot = ssp.AGB_mean[-1].copy()
    toplot.values = (toplot.values-med[0].values)*.48

    toplot.plot.imshow(ax=axgr[sc],vmin=-50.,vmax=50.,extend='both',cbar_ax=axgr.cbar_axes[0],
                        interpolation='nearest',cbar_kwargs={'label':'Mg C ha$^{-1}$'},
                        cmap='bwr_r',xlim = (-120,160),ylim=(-60,40),
                        yticks=np.arange(-60,41,20),xticks=np.arange(-120,161,40),
                        add_labels=False)

    axgr[sc].yaxis.set_major_formatter(LatitudeFormatter())
    axgr[sc].xaxis.set_major_formatter(LongitudeFormatter())
    axgr[sc].set_title(scenlong[scen])
#    axgr[sc].coastlines(resolution='110m',lw=.75)
    axgr[sc].add_feature(cfeat.LAND,facecolor='silver',zorder=-1)
    axgr[sc].add_feature(cfeat.OCEAN,facecolor='k',zorder=-1)

fig.show()
fig.savefig('end_of_century.png',dpi=300,bbox_inches='tight')
