"""
20/11/2018 - JFE
this scripts plots bars showing the 21st century evol of AGB according to LUH2
dataset for 6 ssp scenarios for the whole region, and separated by continents
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from useful import *
from mpl_toolkits.axes_grid1 import AxesGrid
import pandas as pd

#get areas and landmask to calculate continental numbers
areas = get_areas()
_,landmask = get_predictors()

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
fig = plt.figure('bars',figsize=(12,8));fig.clf()

titles = ['a) Pantropical','b) Americas','c) Africa','d) Asia']

#create empty dataframe to store results
results = pd.DataFrame(np.zeros([6,4]),columns = ['pantrop','am','af','as'])
#iterate regions
for mm, mask in enumerate([mask_pantrop,mask_america,mask_africa,mask_asia]):
    #get current agb in the region
    ticks = []
    ax = fig.add_subplot(2,2,mm+1)
    print(titles[mm])
    hist = {}
    lvls = ['mean','lower','upper']
    for aa, agb in enumerate([med,med-unc,med+unc]):
        lvl = lvls[aa]
        agb[0].values[agb[0]<0] = 0
        hist[lvl] = ((agb[0]*areas*mask).sum()*1e-13*.48)

    for sc,scen in enumerate(['ssp126','ssp434','ssp245','ssp460','ssp370','ssp585']):
        ticks.append(scenlong[scen])
        ssp = xr.open_dataset('/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/output/AGB_%s.nc' % scen)

        #calculate end of century AGB for all levels
        ssp_mean = (ssp.AGB_mean[-1]*areas*mask).sum()*1e-13*.48
        dmean = ssp_mean - hist['mean']

        results.iloc[sc,mm] = dmean

        ssp_upp = (ssp.AGB_upper[-1]*areas*mask).sum()*1e-13*.48
        dupp = ssp_upp - hist['upper']

        ssp_low = (ssp.AGB_lower[-1]*areas*mask).sum()*1e-13*.48
        dlow = ssp_low - hist['lower']

        ax.bar([sc],[dmean],color=cols[scen])
        ax.vlines([sc],ymin=min(dlow,dupp),ymax=max(dlow,dupp))

        print('%s: %04.1f Pg C (%04.1f Pg C / %04.1f Pg C)' % (scen,ssp_mean,ssp_low,ssp_upp))
        print('%s: %04.1f Pg C (%04.1f Pg C / %04.1f Pg C)' % (scen,dmean,dlow,dupp))

    ax.set_xticks(range(6))
    ax.set_xticklabels(ticks,size='small',rotation = 45)
    ax.text(0.03,0.97,titles[mm],transform = ax.transAxes,weight='bold',va='top',ha='left')
    ax.set_ylabel('Future impact of LULCC on AGB [Pg C]')
    ax.hlines(0,ax.get_xlim()[0],ax.get_xlim()[1])
    ax.set_ylim(-40,20)
fig.show()
fig.savefig('fig3_bars.png',dpi=300,bbox_inches='tight')
