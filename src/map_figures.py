"""
# Scripts to produce figures for the potential biomass maps
# 06/02/2019 DTM
# based on scripts origianally written by JFE
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid

#country_code = 'WAFR'
#version = '002'
#iterations = 5
#path = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/output/'

"""
Plot the iterative changes in AGBpot, alongside the original AGB map for
comparison. Note that a maximum of five iterations are plotted - the first, and
the final four (or fewer).
"""
"""
def plot_AGBpot_iterations(path,country_code,version,iterations,vmin=0,vmax=200):
    ncfile = '%s%s_%s_AGB_potential_RFR_worldclim_soilgrids_2pass.nc' % (path,country_code,version)
    #ncfile = 'test.nc'
    nc = xr.open_dataset(ncfile)
    #nc.values[nc.values == nc.nodatavals[0]] = np.nan
"""
def plot_AGBpot_iterations(nc,iterations,country_code,version,path2output = './',vmin=0,vmax=200):
    #create a figure using the axesgrid to make the colorbar fit on the axis
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,dict(map_projection=projection))

    #create figure
    fig = plt.figure('AGBpot_iterative',figsize=(10,10))
    fig.clf()

    #create axes grid
    axgr = AxesGrid(fig,111,nrows_ncols=(3,2),axes_class=axes_class,label_mode='',
                    cbar_mode='each',cbar_pad = 0.25,cbar_size="3%",axes_pad=.5)

    #plot setup
    xlim = (np.min(nc.lon.values),np.max(nc.lon.values))
    ylim = (np.min(nc.lat.values),np.max(nc.lat.values))
    cmap = 'viridis'
    titles = ['a) AGB$_{obs}$','b) AGB$_{pot}$ 1','c) AGB$_{pot}$ 2',
                'd) AGB$_{pot}$ 3','e) AGB$_{pot}$ 4',
                'f) AGB$_{pot}$ 5']
    maps2plot = [nc.AGBobs]


    if iterations <= 5:
        print('<=5')
        for ii in range(0,iterations):
            maps2plot.append(nc['AGBpot%i' % (ii+1)])
    # only going to plot first and final four iterations if there are a lot
    else:
        maps2plot.append(nc['AGBpot1'])
        for ii in range(iterations-4,iterations):
            maps2plot.append(nc['AGBpot%i' % (ii+1)])

    # now plot maps onto axes
    for mm,map2plot in enumerate(maps2plot):
        (map2plot*.48).plot.imshow(ax=axgr[mm],cbar_ax=axgr.cbar_axes[mm],vmin=vmin,vmax=vmax,
                            extend='max',interpolation='nearest',cbar_kwargs={'label':'Mg C ha$^{-1}$'},
                            cmap=cmap,xticks=np.arange(-120,161,5),yticks=np.arange(-60,41,5),
                            add_labels=False,ylim=ylim,xlim=xlim)

        #set labels
        axgr[mm].yaxis.set_major_formatter(LatitudeFormatter())
        axgr[mm].xaxis.set_major_formatter(LongitudeFormatter())
        axgr[mm].text(0.98,0.02,titles[mm],transform=axgr[mm].transAxes,ha='right',
                        va='bottom',weight='bold')

    #fig.show()
    fig.savefig('%s/%s_%s_iterative_AGBpot.png' % (path2output,country_code,version),bbox_inches='tight',dpi=300)

"""
Plot the iterative changes in training areas, alongside the original AGB map for
comparison. Note that a maximum of five iterations are plotted - the first, and
the final four (or fewer).
"""
def plot_training_areas_iterative(nc,iterations,country_code,version,path2output='./'):
    #create a figure using the axesgrid to make the colorbar fit on the axis
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,dict(map_projection=projection))

    #create figure
    fig = plt.figure('training_iterative',figsize=(10,10))
    fig.clf()

    #create axes grid
    axgr = AxesGrid(fig,111,nrows_ncols=(3,2),axes_class=axes_class,label_mode='',
                    cbar_mode='each',cbar_pad = 0.25,cbar_size="3%",axes_pad=.5)

    #plot setup
    xlim = (np.min(nc.lon.values),np.max(nc.lon.values))
    ylim = (np.min(nc.lat.values),np.max(nc.lat.values))
    cmap = 'plasma'
    titles = ['a) AGB$_{obs}$','b) training iter 1$','c) training iter 2$',
                'd) training iter 3$','e) training iter 4$',
                'f) training iter 5$']
    maps2plot = [nc.AGBobs]


    if iterations <= 5:
        for ii in range(0,iterations):
            maps2plot.append(nc['trainset%i' % (ii+1)])
    # only going to plot first and final four iterations if there are a lot
    else:
        maps2plot.append(nc['trainset1'])
        for ii in range(iterations-4,iterations):
            maps2plot.append(nc['trainset%i' % (ii+1)])

    # now plot maps onto axes
    for mm,map2plot in enumerate(maps2plot):
        (map2plot*.48).plot.imshow(ax=axgr[mm],cbar_ax=axgr.cbar_axes[mm],
                            interpolation='nearest',cbar_kwargs={'label':'Mg C ha$^{-1}$'},
                            cmap=cmap,xticks=np.arange(-120,161,40),yticks=np.arange(-60,41,20),
                            add_labels=False,ylim=ylim,xlim=xlim)

        #set labels
        axgr[mm].yaxis.set_major_formatter(LatitudeFormatter())
        axgr[mm].xaxis.set_major_formatter(LongitudeFormatter())
        axgr[mm].text(0.98,0.02,titles[mm],transform=axgr[mm].transAxes,ha='right',
                        va='bottom',weight='bold')

    #fig.show()
    fig.savefig('%s/%s_%s_iterative_training.png' % (path2output,country_code,version),bbox_inches='tight',dpi=300)

"""
Plot the original AGB map, the potential map and the training set used
"""
def plot_AGB_AGBpot_training(nc,iterations,country_code,version,path2output='./',vmin=0,vmax=200):
    #create a figure using the axesgrid to make the colorbar fit on the axis
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,dict(map_projection=projection))

    #create figure
    fig = plt.figure('AGBpot_final',figsize=(10,6))
    fig.clf()

    #create axes grid
    axgr = AxesGrid(fig,111,nrows_ncols=(3,1),axes_class=axes_class,label_mode='',
                    cbar_mode='each',cbar_pad = 0.25,cbar_size="3%",axes_pad=.5)

    #plot setup
    xlim = (np.min(nc.lon.values),np.max(nc.lon.values))
    ylim = (np.min(nc.lat.values),np.max(nc.lat.values))
    cmap = 'viridis'
    titles = ['a) AGB$_{obs}$','b) AGB$_{pot}$','c) training set']
    maps2plot = [nc.AGBobs]
    maps2plot.append(nc['AGBpot%i' % (iterations)])
    maps2plot.append(nc['trainset%i' % (iterations)])

    # now plot maps onto axes
    for mm,map2plot in enumerate(maps2plot):
        if mm<2:
            (map2plot*.48).plot.imshow(ax=axgr[mm],cbar_ax=axgr.cbar_axes[mm],vmin=vmin,vmax=vmax,
                            extend='max',interpolation='nearest',cbar_kwargs={'label':'Mg C ha$^{-1}$'},
                            cmap=cmap,xticks=np.arange(-120,161,40),yticks=np.arange(-60,41,20),
                            add_labels=False,ylim=ylim,xlim=xlim)
        else:
            (map2plot*.48).plot.imshow(ax=axgr[mm],cbar_ax=axgr.cbar_axes[mm],
                            interpolation='nearest',cbar_kwargs={'label':'Mg C ha$^{-1}$'},
                            cmap=cmap,xticks=np.arange(-120,161,40),yticks=np.arange(-60,41,20),
                            add_labels=False,ylim=ylim,xlim=xlim)

        #set labels
        axgr[mm].yaxis.set_major_formatter(LatitudeFormatter())
        axgr[mm].xaxis.set_major_formatter(LongitudeFormatter())
        axgr[mm].text(0.98,0.02,titles[mm],transform=axgr[mm].transAxes,ha='right',
                        va='bottom',weight='bold')

    #fig.show()
    fig.savefig('%s/%s_%s_AGB_AGBpot_training.png' % (path2output,country_code,version),bbox_inches='tight',dpi=300)

"""
Plot residuals from training data spatially
"""
def plot_training_residuals(nc,iteration,country_code,version,path2output='./',vmin=[0,0,-50],vmax=[200,200,50]):

    #create a figure using the axesgrid to make the colorbar fit on the axis
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,dict(map_projection=projection))

    #create figure
    fig = plt.figure('training_residuals',figsize=(8,12))
    fig.clf()

    #create axes grid
    axgr = AxesGrid(fig,111,nrows_ncols=(3,1),axes_class=axes_class,label_mode='',
                    cbar_mode='each',cbar_pad = 0.25,cbar_size="3%",axes_pad=.5)

    #plot setup
    xlim = (np.min(nc.lon.values),np.max(nc.lon.values))
    ylim = (np.min(nc.lat.values),np.max(nc.lat.values))
    cmap = ['viridis','viridis','bwr']
    titles = ['a) training data', 'b) predicted', 'c) residuals']
    residuals = nc['AGBpot%i' % (iteration)]-nc.AGBobs
    residuals.values[nc['trainset%i' % iteration].values<=0]=np.nan
    training = nc.AGBobs
    training.values[nc['trainset%i' % iteration].values<=0]=np.nan
    predicted = nc['AGBpot%i' % (iteration)]
    predicted.values[nc['trainset%i' % iteration].values<=0]=np.nan
    maps2plot = [training,predicted,residuals]

    # now plot maps onto axes
    for mm,map2plot in enumerate(maps2plot):
        (map2plot*.48).plot.imshow(ax=axgr[mm],cbar_ax=axgr.cbar_axes[mm],vmin=vmin[mm],vmax=vmax[mm],
                            extend='max',interpolation='nearest',cbar_kwargs={'label':'Mg C ha$^{-1}$'},
                            cmap=cmap[mm],xticks=np.arange(-120,161,5),yticks=np.arange(-60,41,5),
                            add_labels=False,ylim=ylim,xlim=xlim)

        #set labels
        axgr[mm].yaxis.set_major_formatter(LatitudeFormatter())
        axgr[mm].xaxis.set_major_formatter(LongitudeFormatter())
        axgr[mm].text(0.98,0.02,titles[mm],transform=axgr[mm].transAxes,ha='right',
                        va='bottom',weight='bold')
        #add grey mask for land regions outside the study, and black for the oceans
        axgr[mm].add_feature(cfeat.LAND,zorder=-1,facecolor='silver')
        axgr[mm].add_feature(cfeat.OCEAN,zorder=-1,facecolor='gray')


    #fig.show()
    fig.savefig('%s/%s_%s_training_residuals.png' % (path2output,country_code,version),bbox_inches='tight',dpi=300)
