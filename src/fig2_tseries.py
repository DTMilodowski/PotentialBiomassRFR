"""
19/11/2018 - JFE
this scripts plots time series of the evol of AGB according to the LUH2
dataset for the historical period and 6 ssp scenarios
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

#create the figure / ax
fig=plt.figure('tseries');fig.clf()
ax = fig.add_subplot(111)

#load agb with past land use
pot = xr.open_dataset('/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/output/AGB_hist.nc')

ref = pot.ts_mean[(pot.time>=2000) & (pot.time<=2009)].mean()
toplot = (pot.ts_mean-ref)*.48
toplot.plot(ax=ax,color='k',ls='-',lw=2,label='Historical')

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

for sc,scen in enumerate(['ssp126','ssp434','ssp245','ssp460','ssp370','ssp585']):
    pot = xr.open_dataset('/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/output/AGB_%s.nc' % scen)
    toplot = (pot.ts_mean-ref)*.48
    toplot.plot(ax=ax,color=cols[scen],ls='-',lw=2,label=scenlong[scen])

ax.set_ylabel('Impact of LULCC on AGB [Pg C]')
ax.set_xlabel('Year')
ax.legend(loc='lower left')
ax.fill_between([2000,2009],[-45,-45],[45,45],color='silver',edgecolor='silver',zorder=-1)
ax.set_ylim(-45,45)
ax.grid(True,ls=':')
fig.show()
fig.savefig('fig2_tseries.png',dpi=300,bbox_inches='tight')
