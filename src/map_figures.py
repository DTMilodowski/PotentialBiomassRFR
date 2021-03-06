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
def plot_AGBpot_iterations(nc,iterations,country_code,version,path2output = './',agb_source='',vmin=0,vmax=200):
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
    fig.savefig('%s/%s_%s%s_iterative_AGBpot.png' % (path2output,country_code,version,agb_source),bbox_inches='tight',dpi=300)

"""
Plot the iterative changes in training areas, alongside the original AGB map for
comparison. Note that a maximum of five iterations are plotted - the first, and
the final four (or fewer).
"""
def plot_training_areas_iterative(nc,iterations,country_code,version,path2output='./',agb_source=''):
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
    fig.savefig('%s/%s_%s%s_iterative_training.png' % (path2output,country_code,version,agb_source),bbox_inches='tight',dpi=300)

"""
Plot the original AGB map, the potential map and the training set used
"""
def plot_AGB_AGBpot_training(nc,iterations,country_code,version,path2output='./',agb_source='',vmin=0,vmax=200):
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
    fig.savefig('%s/%s_%s%s_AGB_AGBpot_training.png' % (path2output,country_code,version,agb_source),bbox_inches='tight',dpi=300)

def plot_AGB_AGBpot_training_final(nc,country_code,version,path2output='./',agb_source='',
                                    vmin=0,vmax=200,
                                    clip=False,mask=np.array([])):
    #create a figure using the axesgrid to make the colorbar fit on the axis
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,dict(map_projection=projection))

    #create figure
    fig = plt.figure('AGBpot_final',figsize=(10,5))
    fig.clf()

    #create axes grid
    axgr = AxesGrid(fig,111,nrows_ncols=(1,3),axes_class=axes_class,label_mode='',
                    cbar_mode='single',cbar_pad = 0.,cbar_size="3%",axes_pad=.6,
                    cbar_location = 'bottom')

    #plot setup
    if clip: # use mask to limit plot extent
        lonmask = np.any(mask==1,axis=0)
        latmask = np.any(mask==1,axis=1)
        xlim = (np.min(nc.lon.values[lonmask]),np.max(nc.lon.values[lonmask]))
        ylim = (np.min(nc.lat.values[latmask]),np.max(nc.lat.values[latmask]))
    else:
        xlim = (np.min(nc.lon.values),np.max(nc.lon.values))
        ylim = (np.min(nc.lat.values),np.max(nc.lat.values))
    cmap = 'viridis'
    titles = ['a) AGB$_{obs}$','b) AGB$_{pot}$','c) training\nset']
    maps2plot = [nc.AGBobs]
    maps2plot.append(nc['AGBpot'])
    maps2plot.append(nc['training'])

    # now plot maps onto axes
    for mm,map2plot in enumerate(maps2plot):
        if mm<2:
            (map2plot*.48).plot.imshow(ax=axgr[mm],cbar_ax=axgr.cbar_axes[mm],vmin=vmin,vmax=vmax,
                            extend='max',interpolation='nearest',
                            cbar_kwargs={'label':'AGB / Mg C ha$^{-1}$','orientation':'horizontal'},
                            cmap=cmap,xticks=np.arange(-120,161,20),yticks=np.arange(-60,41,20),
                            add_labels=False,ylim=ylim,xlim=xlim)
        else:
            (map2plot*.48).plot.imshow(ax=axgr[mm],interpolation='nearest',add_colorbar=False,
                            cmap=cmap,xticks=np.arange(-120,161,20),yticks=np.arange(-60,41,20),
                            add_labels=False,ylim=ylim,xlim=xlim)

        #set labels
        axgr[mm].yaxis.set_major_formatter(LatitudeFormatter())
        axgr[mm].xaxis.set_major_formatter(LongitudeFormatter())
        axgr[mm].text(0.98,0.98,titles[mm],transform=axgr[mm].transAxes,ha='right',
                        va='top',weight='bold')

    #fig.show()
    fig.savefig('%s/%s_%s%s_AGB_AGBpot_training_final.png' % (path2output,country_code,version,agb_source),bbox_inches='tight',dpi=300)

def plot_AGB_AGBpot_final(nc,country_code,version,path2output='./',agb_source='',
                                    vmin=0,vmax=200,
                                    clip=False,mask=np.array([])):
    #create a figure using the axesgrid to make the colorbar fit on the axis
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,dict(map_projection=projection))

    #create figure
    fig = plt.figure('AGBpot_final',figsize=(10,5))
    fig.clf()

    #create axes grid
    axgr = AxesGrid(fig,111,nrows_ncols=(1,2),axes_class=axes_class,label_mode='',
                    cbar_mode='single',cbar_pad = 0.3,cbar_size="5%",axes_pad=.6,
                    cbar_location = 'right')

    #plot setup
    if clip: # use mask to limit plot extent
        lonmask = np.any(mask==1,axis=0)
        latmask = np.any(mask==1,axis=1)
        xlim = (np.min(nc.lon.values[lonmask]),np.max(nc.lon.values[lonmask]))
        ylim = (np.min(nc.lat.values[latmask]),np.max(nc.lat.values[latmask]))
    else:
        xlim = (np.min(nc.lon.values),np.max(nc.lon.values))
        ylim = (np.min(nc.lat.values),np.max(nc.lat.values))
    cmap = 'viridis'
    titles = ['a) AGB$_{obs}$','b) AGB$_{pot}$']
    maps2plot = [nc.AGBobs,nc.AGBpot]

    # now plot maps onto axes
    for mm,map2plot in enumerate(maps2plot):
        if mm<2:
            (map2plot*.48).plot.imshow(ax=axgr[mm],cbar_ax=axgr.cbar_axes[mm],vmin=vmin,vmax=vmax,
                            extend='max',interpolation='nearest',
                            cbar_kwargs={'label':'AGB / Mg C ha$^{-1}$','orientation':'vertical'},
                            cmap=cmap,xticks=np.arange(-120,161,20),yticks=np.arange(-60,41,20),
                            add_labels=False,ylim=ylim,xlim=xlim)
        else:
            (map2plot*.48).plot.imshow(ax=axgr[mm],interpolation='nearest',add_colorbar=False,
                            cmap=cmap,xticks=np.arange(-120,161,20),yticks=np.arange(-60,41,20),
                            add_labels=False,ylim=ylim,xlim=xlim)

        #set labels
        axgr[mm].yaxis.set_major_formatter(LatitudeFormatter())
        axgr[mm].xaxis.set_major_formatter(LongitudeFormatter())
        axgr[mm].text(0.98,0.98,titles[mm],transform=axgr[mm].transAxes,ha='right',
                        va='top',weight='bold')

    #fig.show()
    fig.savefig('%s/%s_%s%s_AGB_AGBpot_final.png' % (path2output,country_code,version,agb_source),bbox_inches='tight',dpi=300)

"""
Plot defecit
"""
import seaborn as sns
def plot_AGBdef_final(nc,country_code,version,path2output='./',agb_source='',clip=False,mask=np.array([])):
    #create a figure using the axesgrid to make the colorbar fit on the axis
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,dict(map_projection=projection))

    #create figure
    fig = plt.figure('AGBpot_final',figsize=(5,5))
    fig.clf()

    #create axes grid
    axgr = AxesGrid(fig,111,nrows_ncols=(1,1),axes_class=axes_class,label_mode='',
                    cbar_mode='single',cbar_pad = 0.,cbar_size="3%",axes_pad=.6,
                    cbar_location = 'bottom')
    axgr[0].set_facecolor('0.5')
    axgr.cbar_axes[0].set_facecolor('0.5')
    #plot setup
    if clip: # use mask to limit plot extent
        lonmask = np.any(mask==1,axis=0)
        latmask = np.any(mask==1,axis=1)
        xlim = (np.min(nc.lon.values[lonmask]),np.max(nc.lon.values[lonmask]))
        ylim = (np.min(nc.lat.values[latmask]),np.max(nc.lat.values[latmask]))
    else:
        xlim = (np.min(nc.lon.values),np.max(nc.lon.values))
        ylim = (np.min(nc.lat.values),np.max(nc.lat.values))
    cmap = 'viridis'
    title = 'AGB$_{def}$'
    map2plot = nc['AGBobs']-nc['AGBpot']
    # now plot maps onto axes
    (map2plot*.48).plot.imshow(ax=axgr[0],cbar_ax=axgr.cbar_axes[0],robust=True,
                    interpolation='nearest',cbar_kwargs={'orientation':'horizontal',
                    'label':'AGB$_{def}$ / Mg C ha$^{-1}$'},
                    cmap=sns.diverging_palette(10,240,as_cmap=True),
                    xticks=np.arange(-120,161,20),yticks=np.arange(-60,41,20),
                    add_labels=False,ylim=ylim,xlim=xlim)
    #set labels
    axgr[0].yaxis.set_major_formatter(LatitudeFormatter())
    axgr[0].xaxis.set_major_formatter(LongitudeFormatter())
    axgr[0].text(0.98,0.98,title,transform=axgr[0].transAxes,ha='right',
                        va='top',weight='bold')

    #fig.show()
    fig.savefig('%s/%s_%s%s_AGBdef_final.png' % (path2output,country_code,version,agb_source),bbox_inches='tight',dpi=300)

"""
Plot sequestration potential
"""
def plot_AGBseq_final(nc,country_code,version,path2output='./',agb_source='',clip=False,mask=np.array([])):
    #create a figure using the axesgrid to make the colorbar fit on the axis
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,dict(map_projection=projection))

    #create figure
    fig = plt.figure('AGBseq_final',figsize=(5,5))
    fig.clf()

    #create axes grid
    axgr = AxesGrid(fig,111,nrows_ncols=(1,1),axes_class=axes_class,label_mode='',
                    cbar_mode='single',cbar_pad = 0.,cbar_size="3%",axes_pad=.6,
                    cbar_location = 'bottom')
    axgr[0].set_facecolor('0.5')
    axgr.cbar_axes[0].set_facecolor('0.5')
    #plot setup
    if clip: # use mask to limit plot extent
        lonmask = np.any(mask==1,axis=0)
        latmask = np.any(mask==1,axis=1)
        xlim = (np.min(nc.lon.values[lonmask]),np.max(nc.lon.values[lonmask]))
        ylim = (np.min(nc.lat.values[latmask]),np.max(nc.lat.values[latmask]))
    else:
        xlim = (np.min(nc.lon.values),np.max(nc.lon.values))
        ylim = (np.min(nc.lat.values),np.max(nc.lat.values))
    cmap = 'viridis'
    title = 'Sequestration\n potential'
    map2plot = nc['AGBpot']-nc['AGBobs']
    # now plot maps onto axes
    (map2plot*.48).plot.imshow(ax=axgr[0],cbar_ax=axgr.cbar_axes[0],robust=True,
                    interpolation='nearest',cbar_kwargs={'orientation':'horizontal',
                    'label':'Sequestration potential / Mg C ha$^{-1}$'},
                    cmap=sns.diverging_palette(275,150,l=66,s=90,as_cmap=True),
                    xticks=np.arange(-120,161,20),yticks=np.arange(-60,41,20),
                    add_labels=False,ylim=ylim,xlim=xlim)
    #set labels
    axgr[0].yaxis.set_major_formatter(LatitudeFormatter())
    axgr[0].xaxis.set_major_formatter(LongitudeFormatter())
    axgr[0].text(0.98,0.98,title,transform=axgr[0].transAxes,ha='right',
                        va='top',weight='bold')

    #fig.show()
    fig.savefig('%s/%s_%s%s_AGBseq_final.png' % (path2output,country_code,version,agb_source),bbox_inches='tight',dpi=300)

"""
Plot uncertainty
"""
def plot_AGBpot_uncertainty(nc,country_code,version,path2output='./',agb_source='',clip=False,
    mask=np.array([])):
    #create a figure using the axesgrid to make the colorbar fit on the axis
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,dict(map_projection=projection))

    #create figure
    fig = plt.figure('AGBunc_final',figsize=(7,5))
    fig.clf()

    #create axes grid
    axgr = AxesGrid(fig,111,nrows_ncols=(1,2),axes_class=axes_class,label_mode='',
                    cbar_mode='each',cbar_pad = 0.5,cbar_size="3%",axes_pad=.5,
                    cbar_location = 'bottom')

    #plot setup
    if clip: # use mask to limit plot extent
        lonmask = np.any(mask==1,axis=0)
        latmask = np.any(mask==1,axis=1)
        xlim = (np.min(nc.lon.values[lonmask]),np.max(nc.lon.values[lonmask]))
        ylim = (np.min(nc.lat.values[latmask]),np.max(nc.lat.values[latmask]))
    else:
        xlim = (np.min(nc.lon.values),np.max(nc.lon.values))
        ylim = (np.min(nc.lat.values),np.max(nc.lat.values))
    cmap = 'viridis'
    titles = ['a) uncertainty\nrange','b) relative\nuncertainty\nrange']
    unc = nc.AGBpot_max-nc.AGBpot_min
    unc_frac = unc/nc.AGBpot
    maps2plot = [unc,unc_frac]

    (unc*.48).plot.imshow(ax=axgr[0],cbar_ax=axgr.cbar_axes[0],
                            vmin=0,extend='max',interpolation='nearest',cmap=cmap,
                            cbar_kwargs={'label':'Uncertainty range / Mg C ha$^{-1}$','orientation':'horizontal'},
                            xticks=np.arange(-120,161,20),yticks=np.arange(-60,41,20),
                            add_labels=False,ylim=ylim,xlim=xlim)
    (unc_frac).plot.imshow(ax=axgr[1],cbar_ax=axgr.cbar_axes[1],vmin=0,vmax=2,
                            extend='max',interpolation='nearest',cmap=cmap,
                            cbar_kwargs={'label':'Uncertainty / AGB$_{pot}$','orientation':'horizontal'},
                            xticks=np.arange(-120,161,20),yticks=np.arange(-60,41,20),
                            add_labels=False,ylim=ylim,xlim=xlim)
    for mm,map2plot in enumerate(maps2plot):
        #set labels
        axgr[mm].yaxis.set_major_formatter(LatitudeFormatter())
        axgr[mm].xaxis.set_major_formatter(LongitudeFormatter())
        axgr[mm].text(0.98,0.98,titles[mm],transform=axgr[mm].transAxes,
                        ha='right',va='top',weight='bold')

    #fig.show()
    fig.savefig('%s/%s_%s%s_AGBpot_uncertainty.png' % (path2output,country_code,version,agb_source),bbox_inches='tight',dpi=300)


"""
Plot residuals from training data spatially
"""
def plot_training_residuals(nc,iteration,country_code,version,path2output='./',agb_source='',
    vmin=[0,0,-50],vmax=[200,200,50],clip=False,mask=np.array([])):

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
    if clip: # use mask to limit plot extent
        lonmask = np.any(mask==1,axis=0)
        latmask = np.any(mask==1,axis=1)
        xlim = (np.min(nc.lon.values[lonmask]),np.max(nc.lon.values[lonmask]))
        ylim = (np.min(nc.lat.values[latmask]),np.max(nc.lat.values[latmask]))
    else:
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
    fig.savefig('%s/%s_%s%s_training_residuals.png' % (path2output,country_code,version,agb_source),bbox_inches='tight',dpi=300)
