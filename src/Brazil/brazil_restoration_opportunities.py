"""
BRAZIL RESTORATION OPPORTUNITIES
--------------------------------------------------------------------------------
Combine restoration potential maps with:
- WRI restoration opportunity maps
- Land cover maps

"""
import numpy as np
import sys
import xarray as xr #xarray to read all types of formats
import glob as glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import useful as useful
sns.set(style="whitegrid")

"""
A quick function to plot error bars onto bar plot
"""
def plot_bar_CIs(lc,uc,ax,jitter=0):
    positions = np.arange(uc.size)+jitter
    for ii,pos in enumerate(positions):
        ax.plot([pos,pos],[lc[ii],uc[ii]],'-',lw=3.8,color='white')
        ax.plot([pos,pos],[lc[ii],uc[ii]],'-',lw=2,color='0.5')
    return 0


country_code = 'BRA'
version = '011'

"""
#===============================================================================
PART A: DEFINE PATHS AND LOAD IN DATA
- Potential biomass maps (from netcdf file)
- Biome boundaries (Mapbiomas)
- WRI opportunity map
#-------------------------------------------------------------------------------
"""
path2data = '/disk/scratch/local.2/PotentialBiomass/processed/%s/' % country_code
path2output = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/output/' # change to either your own storage space or create a directory on racadal scratch ('/disk/scratch/local.2/)
path2mapbiomas = '/scratch/local.2/MAPBIOMAS/'

# load potential biomass models from netdf file
AGBpot_ds = xr.open_dataset('%s%s_%s_AGB_potential_RFR_worldclim_soilgrids_final.nc' %
                                (path2output, country_code,version))
AGBpot     = AGBpot_ds['AGBpot'].values
AGBobs     = AGBpot_ds['AGBobs'].values
AGBseq     = AGBpot-AGBobs
AGBpot_min = AGBpot_ds['AGBpot_min'].values
AGBobs_min = AGBpot_ds['AGBobs_min'].values
AGBseq_min = AGBpot_min-AGBobs_min
AGBpot_max = AGBpot_ds['AGBpot_max'].values
AGBobs_max = AGBpot_ds['AGBobs_max'].values
AGBseq_max = AGBpot_max-AGBobs_max

cell_areas =  useful.get_areas(latorig=AGBpot_ds.coords['lat'].values,
                            lonorig=AGBpot_ds.coords['lon'].values)
cell_areas/=10**4 # m^2 -> ha
# load biome boundaries
# - for each biome, load in the 1km regridded mapbiomas dataset for that biome
#   and create a mask based on the pixels for which there is data
biome_files = glob.glob('%s*1km.tif' % path2mapbiomas)
masks = {}
biome_labels = ['Amazonia','Cerrado','Pampa','Mataatlantica','Caatinga','Pantanal']
for ii, file in enumerate(biome_files):
    mb = xr.open_rasterio(file)[0]
    if ii == 0:
        masks['Brazil'] = mb.values>0
    else:
        masks['Brazil'] += mb.values>0
    masks[biome_labels[ii]] = mb.values>0*np.isfinite(AGBpot)
masks['Brazil'] *=np.isfinite(AGBpot)
# load opportunity map
opportunity = xr.open_rasterio('%sWRI_restoration/WRI_restoration_opportunities_%s.tif' % (path2data, country_code))[0]
opp_class = ['existing forest','wide-scale','mosaic','remote','urban-agriculture']
for cc,opp in enumerate(opp_class):
    masks[opp] = (opportunity.values==cc)*masks['Brazil']

# Load mapbiomas data for 2008
mb2005 = useful.load_mapbiomas('BRA',timestep=20,aggregate=1)

# Load mapbiomas data for latest year (2017)
mb2017 = useful.load_mapbiomas('BRA',timestep=-1,aggregate=1)

"""
#===============================================================================
PART B: National summaries
#-------------------------------------------------------------------------------
"""
# Summarise each of the opportunity classes for Brazil
areas_ha = np.zeros(5)
potC_Mg = np.zeros(5)
seqC_Mg = np.zeros(5)
obsC_Mg = np.zeros(5)

# Arrays for upper and lower limits of uncertainty
potC_Mg_max = np.zeros(5)
seqC_Mg_max = np.zeros(5)
obsC_Mg_max = np.zeros(5)

potC_Mg_min = np.zeros(5)
seqC_Mg_min = np.zeros(5)
obsC_Mg_min = np.zeros(5)

for cc,opp in enumerate(opp_class):
    mask = masks[opp]
    areas_ha[cc] = np.sum(cell_areas[mask])*1.
    potC_Mg[cc] = np.sum(AGBpot[mask]*cell_areas[mask])
    potC_Mg_min[cc] = np.sum(AGBpot_min[mask]*cell_areas[mask])
    potC_Mg_max[cc] = np.sum(AGBpot_max[mask]*cell_areas[mask])
    seqC_Mg[cc] = np.sum(AGBseq[mask]*cell_areas[mask])
    seqC_Mg_min[cc] = np.sum(AGBseq_min[mask]*cell_areas[mask])
    seqC_Mg_max[cc] = np.sum(AGBseq_max[mask]*cell_areas[mask])
    obsC_Mg[cc] = np.sum(AGBobs[mask]*cell_areas[mask])
    obsC_Mg_min[cc] = np.sum(AGBobs_min[mask]*cell_areas[mask])
    obsC_Mg_max[cc] = np.sum(AGBobs_max[mask]*cell_areas[mask])

potC_Mg_ha     = potC_Mg/areas_ha
potC_Mg_ha_min = potC_Mg_min/areas_ha
potC_Mg_ha_max = potC_Mg_max/areas_ha
seqC_Mg_ha     = seqC_Mg/areas_ha
seqC_Mg_ha_min = seqC_Mg_min/areas_ha
seqC_Mg_ha_max = seqC_Mg_max/areas_ha
obsC_Mg_ha     = obsC_Mg/areas_ha
obsC_Mg_ha_min = obsC_Mg_min/areas_ha
obsC_Mg_ha_max = obsC_Mg_max/areas_ha

# opportunity Classes:
# 0   - No opportunity
# 1   - Wide-scale restoration
# 2   - Mosaic restoration
# 3   - Remote restoration
# 4   - Agricultural lands and urban areas
print('\t=====================================================================')
print('\trestoration opportunity areas in 10^6 ha')
print('\t---------------------------------------------------------------------')
print('\tforest\t\twide-scale,\tmosaic,\t\tremote,\t\tagriculture')
print('\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (areas_ha[0]/10.**6,
    areas_ha[1]/10.**6,areas_ha[2]/10.**6,areas_ha[3]/10.**6,areas_ha[4]/10.**6))
print('\t=====================================================================')

print('\t=====================================================================')
print( '\tobserved biomass within each class, in Pg C')
print( '\t---------------------------------------------------------------------')
print( '\tforest\t\twide-scale,\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t\t%.2f' % (obsC_Mg[0]/10.**6,
        obsC_Mg[1]/10.**6,obsC_Mg[2]/10.**6,obsC_Mg[3]/10.**6,obsC_Mg[4]/10.**6))
print( '\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t\t%.2f' % (obsC_Mg_min[0]/10.**6,
        obsC_Mg_min[1]/10.**6,obsC_Mg_min[2]/10.**6,obsC_Mg_min[3]/10.**6,obsC_Mg_min[4]/10.**6))
print( '\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t\t%.2f' % (obsC_Mg_max[0]/10.**6,
        obsC_Mg_max[1]/10.**6,obsC_Mg_max[2]/10.**6,obsC_Mg_max[3]/10.**6,obsC_Mg_max[4]/10.**6))
print( '\t---------------------------------------------------------------------')
print( '\tobserved biomass density within each class, in Pg C / ha')
print( '\t---------------------------------------------------------------------')
print( '\tforest\t\twide-scale,\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (obsC_Mg_ha_min[0],
        obsC_Mg_ha_min[1],obsC_Mg_ha_min[2],obsC_Mg_ha_min[3],obsC_Mg_ha_min[4]))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (obsC_Mg_ha[0],
        obsC_Mg_ha[1],obsC_Mg_ha[2],obsC_Mg_ha[3],obsC_Mg_ha[4]))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (obsC_Mg_ha_max[0],
        obsC_Mg_ha_max[1],obsC_Mg_ha_max[2],obsC_Mg_ha_max[3],obsC_Mg_ha_max[4]))
print( '\t=====================================================================')

print( '\t=====================================================================')
print( '\tpotential biomass within each class, in Pg C')
print( '\t---------------------------------------------------------------------')
print( '\tforest\t\twide-scale,\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t\t%.2f' % (potC_Mg[0]/10.**6,
    potC_Mg[1]/10.**6,potC_Mg[2]/10.**6,potC_Mg[3]/10.**6,potC_Mg[4]/10.**6))
print( '\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t\t%.2f' % (potC_Mg_min[0]/10.**6,
    potC_Mg_min[1]/10.**6,potC_Mg_min[2]/10.**6,potC_Mg_min[3]/10.**6,potC_Mg_min[4]/10.**6))
print( '\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t\t%.2f' % (potC_Mg_max[0]/10.**6,
    potC_Mg_max[1]/10.**6,potC_Mg_max[2]/10.**6,potC_Mg_max[3]/10.**6,potC_Mg_max[4]/10.**6))
print( '\t---------------------------------------------------------------------')
print( '\tpotential biomass density within each class, in Pg C / ha')
print( '\t---------------------------------------------------------------------')
print( '\tforest\t\twide-scale,\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (potC_Mg_ha[0],
    potC_Mg_ha[1],potC_Mg_ha[2],potC_Mg_ha[3],potC_Mg_ha[4]))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (potC_Mg_ha_min[0],
    potC_Mg_ha_min[1],potC_Mg_ha_min[2],potC_Mg_ha_min[3],potC_Mg_ha_min[4]))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (potC_Mg_ha[0],
    potC_Mg_ha_max[1],potC_Mg_ha_max[2],potC_Mg_ha_max[3],potC_Mg_ha_max[4]))
print( '\t=====================================================================')

print( '\t=====================================================================')
print( '\tAGB sequestration potential within each class, in Pg C')
print( '\t---------------------------------------------------------------------')
print( '\tforest\t\twide-scale,\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t\t%.2f' % (seqC_Mg[0]/10.**6,
    seqC_Mg[1]/10.**6,seqC_Mg[2]/10.**6,seqC_Mg[3]/10.**6,seqC_Mg[4]/10.**6))
print( '\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t\t%.2f' % (seqC_Mg_min[0]/10.**6,
    seqC_Mg_min[1]/10.**6,seqC_Mg_min[2]/10.**6,seqC_Mg_min[3]/10.**6,seqC_Mg_min[4]/10.**6))
print( '\t%.2f,\t%.2f,\t%.2f,\t%.2f,\t\t%.2f' % (seqC_Mg_max[0]/10.**6,
    seqC_Mg_max[1]/10.**6,seqC_Mg_max[2]/10.**6,seqC_Mg_max[3]/10.**6,seqC_Mg_max[4]/10.**6))
print( '\t---------------------------------------------------------------------')
print( '\tAGB deficit within each class, in Pg C / ha')
print( '\t---------------------------------------------------------------------')
print( '\tforest\t\twide-scale,\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (seqC_Mg_ha[0],
    seqC_Mg_ha[1],seqC_Mg_ha[2],seqC_Mg_ha[3],seqC_Mg_ha[4]))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (seqC_Mg_ha_min[0],
    seqC_Mg_ha_min[1],seqC_Mg_ha_min[2],seqC_Mg_ha_min[3],seqC_Mg_ha_min[4]))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (seqC_Mg_ha_max[0],
    seqC_Mg_ha_max[1],seqC_Mg_ha_max[2],seqC_Mg_ha_max[3],seqC_Mg_ha_max[4]))

# now plot this information up into some simple charts.
# Three panels:
# - Area
# - AGBobs & AGBpot
# - AGBseq
fig,axes = plt.subplots(nrows=1,ncols=3,sharex='all',figsize=[8,3.4])
sns.barplot(x=opp_class,y=areas_ha,hue=opp_class,palette='Greens_d',dodge=False,
            ax=axes[0])
sns.barplot(x=opp_class,y=potC_Mg,hue=opp_class,palette='Greens_d',dodge=False,
            ax=axes[1],facecolor='white')
sns.barplot(x=opp_class,y=obsC_Mg,hue=opp_class,palette='Greens_d',dodge=False,
            ax=axes[1])
plot_bar_CIs(potC_Mg_min,potC_Mg_max,axes[1],jitter=0.1)
plot_bar_CIs(obsC_Mg_min,obsC_Mg_max,axes[1],jitter=-0.1)
sns.barplot(x=opp_class,y=seqC_Mg,hue=opp_class,palette='Greens_d',dodge=False,
            ax=axes[2])
plot_bar_CIs(seqC_Mg_min,seqC_Mg_max,axes[2])
colours=[]
for patch in axes[0].patches:
    colours.append(patch.get_facecolor())
for ax in axes:
    ax.legend_.remove()
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45, ha='right')
    for ii,patch in enumerate(ax.patches):
        patch.set_edgecolor(colours[ii%len(colours)])
axes[1].set_ylim(bottom=0)
axes[2].set_ylim(axes[1].get_ylim())
# convert areas to 10^6 km
axes[0].set_title('Area of opportunity class')
y_ticks = axes[0].get_yticks()
axes[0].set_yticklabels(['{:3.0f}'.format(i/(10**8)) for i in y_ticks])
axes[0].set_ylabel('Area / 10$^6$ km$^2$')
# convert Mg to 10^9 Mg
axes[1].set_title('Aboveground carbon stock')
axes[1].annotate('potential (open)\nobserved (filled)',xy=(0.95,0.95),
                xycoords='axes fraction',backgroundcolor='white',ha='right',
                va='top',fontsize=10)
y_ticks = axes[1].get_yticks()
axes[1].set_yticklabels(['{:3.0f}'.format(i/(10**9)) for i in y_ticks])
axes[1].set_ylabel('Aboveground carbon / Pg')
axes[2].set_title('Potential carbon defecit')
y_ticks = axes[2].get_yticks()
axes[2].set_yticklabels(['{:3.0f}'.format(i/(10**9)) for i in y_ticks])
axes[2].set_ylabel('Aboveground carbon / Pg')
fig.tight_layout()
fig.savefig('%s%s_%s_national_summary.png' % (path2output,country_code,version))
fig.show()

"""
#===============================================================================
PART C: Biome level summaries
#-------------------------------------------------------------------------------
"""
# Create pandas data frame for ease of plotting with seaborn
# - column variables
#   biome, opportunity_class, area_ha, AGBobs, AGBpot, AGBseq
biome = []
opportunity_class = []
area = []
agbobs = []
agbpot = []
agbseq = []
agbobs_min = []
agbpot_min = []
agbseq_min = []
agbobs_max = []
agbpot_max = []
agbseq_max = []

for bb, bio in enumerate(biome_labels):
    for cc,opp in enumerate(opp_class):
        mask = masks[opp]*masks[bio]
        biome.append(bio)
        opportunity_class.append(opp)
        area.append(np.sum(cell_areas[mask])*1.)
        agbpot.append(np.sum(AGBpot[mask]*cell_areas[mask]))
        agbseq.append(np.sum(AGBseq[mask]*cell_areas[mask]))
        agbobs.append(np.sum(AGBobs[mask]*cell_areas[mask]))
        agbpot_min.append(np.sum(AGBpot_min[mask]*cell_areas[mask]))
        agbseq_min.append(np.sum(AGBseq_min[mask]*cell_areas[mask]))
        agbobs_min.append(np.sum(AGBobs_min[mask]*cell_areas[mask]))
        agbpot_max.append(np.sum(AGBpot_max[mask]*cell_areas[mask]))
        agbseq_max.append(np.sum(AGBseq_max[mask]*cell_areas[mask]))
        agbobs_max.append(np.sum(AGBobs_max[mask]*cell_areas[mask]))

df = pd.DataFrame({'biome':biome,'opportunity class':opportunity_class,
                    'area_ha':area,'AGBobs':agbobs,
                    'AGBobs_min':agbobs_min,'AGBobs_max':agbobs_max,
                    'AGBpot':agbpot,'AGBpot_min':agbpot_min,'AGBpot_max':agbpot_max,
                    'AGBseq':agbseq,'AGBseq_min':agbseq_min,'AGBseq_max':agbseq_max})

n_biomes = len(biome_labels)
biome_display_labels = biome_labels.copy()
biome_display_labels[3] = 'Mata\nAtlantica'

# Create bar plot with restoration potential split by biome
fig,axes = plt.subplots(nrows=n_biomes,ncols=3,sharex='all',sharey='col',
            figsize=[8,12])
for ii,biome in enumerate(biome_labels):
    df_biome=df[df['biome']==biome]
    sns.barplot(x='opportunity class',y='area_ha',hue='opportunity class',
                palette='Greens_d',dodge=False,ax=axes[ii][0],
                data=df_biome)
    sns.barplot(x='opportunity class',y='AGBpot',hue='opportunity class',
                palette='Greens_d',dodge=False,facecolor='white',
                ax=axes[ii][1],data=df_biome)
    sns.barplot(x='opportunity class',y='AGBobs',hue='opportunity class',
                palette='Greens_d',dodge=False,ax=axes[ii][1],
                data=df_biome)
    sns.barplot(x='opportunity class',y='AGBseq',hue='opportunity class',
                palette='Greens_d',dodge=False,ax=axes[ii][2],
                data=df_biome)
    plot_bar_CIs(np.asarray(df_biome['AGBpot_min']),np.asarray(df_biome['AGBpot_max']),axes[ii][1],jitter=0.1)
    plot_bar_CIs(np.asarray(df_biome['AGBobs_min']),np.asarray(df_biome['AGBobs_max']),axes[ii][1],jitter=-0.1)
    plot_bar_CIs(np.asarray(df_biome['AGBseq_min']),np.asarray(df_biome['AGBseq_max']),axes[ii][2])

colours=[]
for patch in axes[0][0].patches:
    colours.append(patch.get_facecolor())
for ii,row in enumerate(axes):
    for jj,ax in enumerate(row):
        ax.legend_.remove()
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45, ha='right')
        ax.set_xlabel(None)
        for ii,patch in enumerate(ax.patches):
            patch.set_edgecolor(colours[jj%len(colours)])
        ax.set_ylim(bottom=0)

axes[0][0].set_title('Area of opportunity\nclass / 10$^6$ km$^2$')
axes[0][1].set_title('Aboveground carbon\nstock / Pg')
axes[0][1].annotate('potential (open)\nobserved (filled)',xy=(0.95,0.90),
                xycoords='axes fraction',backgroundcolor='white',ha='right',
                va='top',fontsize=10)
axes[0][2].set_title('Potential carbon\ndefecit / Pg')

# convert areas to 10^6 km
for ii,ax in enumerate(axes[:,0]):
    y_ticks = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}'.format(i/(10**8)) for i in y_ticks],fontsize=10)
    if np.any((ii==1,ii==4)):
        #ax.set_ylabel('Area / 10$^6$ km$^2$')
        ax.set_ylabel(None)
    else:
        ax.set_ylabel(None)

# convert Mg to 10^9 Mg
for ii,ax in enumerate(axes[:,1]):
    y_ticks = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}'.format(i/(10**9)) for i in y_ticks],fontsize=10)
    if np.any((ii==1,ii==4)):
        #ax.set_ylabel('Aboveground carbon / 10$^9$ Mg')
        ax.set_ylabel(None)
    else:
        ax.set_ylabel(None)

for ii,ax in enumerate(axes[:,2]):
    y_ticks = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}'.format(i/(10**9)) for i in y_ticks],fontsize=10)
    if np.any((ii==1,ii==4)):
        #ax.set_ylabel('Aboveground carbon / 10$^9$ Mg')
        ax.set_ylabel(None)
    else:
        ax.set_ylabel(None)

#fig.tight_layout()
plt.subplots_adjust(wspace = 0.3,hspace=0.2,left=0.2)
for ii,ax in enumerate(axes[:,0]):
    ax.annotate(biome_display_labels[ii],xy=(-.45,0.50),
                xycoords='axes fraction',backgroundcolor='white',ha='center',
                va='center')

fig.savefig('%s%s_%s_biome_summary.png' % (path2output,country_code,version))
fig.show()

"""
#===============================================================================
PART D: Breakdown of potential biomass by landcover type for feasible
restoration areas at time of biomass map.

Note that the Avitabile map is representative of the "2000s". This is somewhat
ambiguous, so I will assume that the 2008 land cover is representative.
#-------------------------------------------------------------------------------
"""
# Subset
# Options: 'Brazil' or one from biome_labels ['Amazonia','Cerrado','Pampa',
#'Mataatlantica','Caatinga','Pantanal']
subset = 'Brazil'

# create landcover masks
lc_masks={}
lc_class = ['Natural Forest','Natural Non-Forest','Plantation','Pasture',
                'Agriculture','Urban']#,'Other']
for cc,lc in enumerate(lc_class):
    lc_masks[lc] = (mb2005==cc+1)

# Create pandas data frame for ease of plotting with seaborn
# - column variables
#   biome, landcover class (2005), area_ha, AGBobs, AGBpot, AGBseq
biome = []
landcover_class = []
area = []
agbobs = []
agbpot = []
agbseq = []
agbobs_min = []
agbpot_min = []
agbseq_min = []
agbobs_max = []
agbpot_max = []
agbseq_max = []

for bb, bio in enumerate(biome_labels):
    for cc,lc in enumerate(lc_class):
        mask = lc_masks[lc]*masks[bio]*np.isfinite(AGBobs)
        biome.append(bio)
        landcover_class.append(lc)
        area.append(np.sum(cell_areas[mask])*1.)
        agbpot.append(np.sum(AGBpot[mask]*cell_areas[mask]))
        agbseq.append(np.sum(AGBseq[mask]*cell_areas[mask]))
        agbobs.append(np.sum(AGBobs[mask]*cell_areas[mask]))
        agbpot_min.append(np.sum(AGBpot_min[mask]*cell_areas[mask]))
        agbseq_min.append(np.sum(AGBseq_min[mask]*cell_areas[mask]))
        agbobs_min.append(np.sum(AGBobs_min[mask]*cell_areas[mask]))
        agbpot_max.append(np.sum(AGBpot_max[mask]*cell_areas[mask]))
        agbseq_max.append(np.sum(AGBseq_max[mask]*cell_areas[mask]))
        agbobs_max.append(np.sum(AGBobs_max[mask]*cell_areas[mask]))

df2005 = pd.DataFrame({'biome':biome,'landcover':landcover_class,
                    'area_ha':area,'AGBobs':agbobs,
                    'AGBobs_min':agbobs_min,'AGBobs_max':agbobs_max,
                    'AGBpot':agbpot,'AGBpot_min':agbpot_min,'AGBpot_max':agbpot_max,
                    'AGBseq':agbseq,'AGBseq_min':agbseq_min,'AGBseq_max':agbseq_max})

# Now plot up summaries according to the subset in question
fig,axes = plt.subplots(nrows=1,ncols=3,sharex='all',figsize=[8,3.4])
if subset in biome_labels:
    df_sub=df[df2005['biome']==subset]
elif subset in ['Brazil','all']:
    df_sub=df2005.groupby('landcover',as_index=False).agg(sum)

sns.barplot(x='landcover',y='area_ha',hue='landcover',
                palette='Greens_d',dodge=False,ax=axes[0],
                data=df_sub)
sns.barplot(x='landcover',y='AGBpot',hue='landcover',
                palette='Greens_d',dodge=False,facecolor='white',
                ax=axes[1],data=df_sub)
sns.barplot(x='landcover',y='AGBobs',hue='landcover',
                palette='Greens_d',dodge=False,ax=axes[1],
                data=df_sub)
sns.barplot(x='landcover',y='AGBseq',hue='landcover',
                palette='Greens_d',dodge=False,ax=axes[2],
                data=df_sub)
plot_bar_CIs(np.asarray(df_sub['AGBpot_min']),np.asarray(df_sub['AGBpot_max']),axes[1],jitter=0.1)
plot_bar_CIs(np.asarray(df_sub['AGBobs_min']),np.asarray(df_sub['AGBobs_max']),axes[1],jitter=-0.1)
plot_bar_CIs(np.asarray(df_sub['AGBseq_min']),np.asarray(df_sub['AGBseq_max']),axes[2])

colours=[]
for patch in axes[0].patches:
    colours.append(patch.get_facecolor())
for ax in axes:
    ax.legend_.remove()
    ax.set_xlabel(None)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45, ha='right')
    for ii,patch in enumerate(ax.patches):
        patch.set_edgecolor(colours[ii%len(colours)])

axes[1].set_ylim(bottom=0)
axes[2].set_ylim(bottom=0)#axes[1].get_ylim())
# convert areas to 10^6 km
axes[0].set_title('Area of landcover class (2005)')
y_ticks = axes[0].get_yticks()
axes[0].set_yticklabels(['{:3.1f}'.format(i/(10**8)) for i in y_ticks])
axes[0].set_ylabel('Area / 10$^6$ km$^2$')
# convert Mg to 10^9 Mg
axes[1].set_title('Aboveground carbon stock')
axes[1].annotate('potential (open)\nobserved (filled)',xy=(0.95,0.95),
                xycoords='axes fraction',backgroundcolor='white',ha='right',
                va='top',fontsize=10)
y_ticks = axes[1].get_yticks()
axes[1].set_yticklabels(['{:3.0f}'.format(i/(10**9)) for i in y_ticks])
axes[1].set_ylabel('Aboveground carbon / 10$^9$ Mg')
axes[2].set_title('Potential carbon defecit')
y_ticks = axes[2].get_yticks()
axes[2].set_yticklabels(['{:3.0f}'.format(i/(10**9)) for i in y_ticks])
axes[2].set_ylabel('Aboveground carbon / 10$^9$ Mg')
fig.tight_layout()
if subset in biome_labels:
    fig.savefig('%s%s_%s_%s_summary_by_landcover.png' % (path2output,country_code,version,subset))
elif subset in ['Brazil','all']:
    fig.savefig('%s%s_%s_national_summary_by_landcover.png' % (path2output,country_code,version))
fig.show()

"""
#===============================================================================
PART E: Breakdown of potential biomass by landcover type for feasible
restoration areas in 2017.

Not applicable for natural land cover, since there is no information on regrowth/
degradation impact on biomass in these areas. For new non-natural land cover
classes, assume mean biomass values, following Chazdon et al., Science, 2014(?),
allowing estimation of restoration potential

#-------------------------------------------------------------------------------
"""
lc2005_masks={}
lc2017_masks={}
#lc_class = ['Natural Forest','Natural Non-Forest','Plantation','Pasture','Agriculture','Urban']#,'Other']
lc_idx = [3,4,5,6]
lc_class = ['Plantation','Pasture','Agriculture','Urban']#,'Other']
for cc,lc in enumerate(lc_class):
    lc2005_masks[lc] = (mb2005==lc_idx[cc])
    lc2017_masks[lc] = (mb2017==lc_idx[cc])

# Update biomass maps
AGBest_2017 = AGBobs.copy()
AGBest_2017_min = AGBobs_min.copy()
AGBest_2017_max = AGBobs_max.copy()
for bb, bio in enumerate(biome_labels):
    for cc,lc in enumerate(lc_class):
        mask = lc2017_masks[lc]*masks[bio]*np.isfinite(AGBobs)
        mask[lc2005_masks[lc]]=False # only update areas that have changed
        dfmask = np.all((df2005['biome']==bio, df2005['landcover']==lc),axis=0)
        #print(bio,lc,float(df2005['AGBobs'][dfmask]/df2005['area_ha'][dfmask]))
        AGBest_2017[mask] = float(df2005['AGBobs'][dfmask]/df2005['area_ha'][dfmask])
        AGBest_2017_min[mask] = float(df2005['AGBobs_min'][dfmask]/df2005['area_ha'][dfmask])
        AGBest_2017_max[mask] = float(df2005['AGBobs_max'][dfmask]/df2005['area_ha'][dfmask])


# Create pandas data frame for ease of plotting with seaborn
# - column variables
#   biome, landcover class (2017), area_ha, AGBobs, AGBpot, AGBseq
biome = []; landcover_class = []; area = []; agbobs = []; agbpot = []; agbseq = []
agbobs_min = []; agbpot_min = []; agbseq_min = []; agbobs_max = []; agbpot_max = []
agbseq_max = []

for bb, bio in enumerate(biome_labels):
    for cc,lc in enumerate(lc_class):
        mask = lc2017_masks[lc]*masks[bio]*np.isfinite(AGBobs)
        dfmask = np.all((df2005['biome']==bio, df2005['landcover']==lc),axis=0)
        biome.append(bio)
        landcover_class.append(lc)
        area.append(np.sum(cell_areas[mask])*1.)
        agbpot.append(np.sum(AGBpot[mask]*cell_areas[mask]))
        agbobs.append(np.sum(AGBest_2017[mask]*cell_areas[mask]))
        agbpot_min.append(np.sum(AGBpot_min[mask]*cell_areas[mask]))
        agbobs_min.append(np.sum(AGBest_2017_min[mask]*cell_areas[mask]))
        agbpot_max.append(np.sum(AGBpot_max[mask]*cell_areas[mask]))
        agbobs_max.append(np.sum(AGBest_2017_max[mask]*cell_areas[mask]))

        agbseq.append(agbpot[-1]-agbobs[-1])
        agbseq_min.append(agbpot_min[-1]-agbobs_min[-1])
        agbseq_max.append(agbpot_max[-1]-agbobs_max[-1])

df2017 = pd.DataFrame({'biome':biome,'landcover':landcover_class,
                    'area_ha':area,'AGBobs':agbobs,
                    'AGBobs_min':agbobs_min,'AGBobs_max':agbobs_max,
                    'AGBpot':agbpot,'AGBpot_min':agbpot_min,'AGBpot_max':agbpot_max,
                    'AGBseq':agbseq,'AGBseq_min':agbseq_min,'AGBseq_max':agbseq_max})

# Now plot up summaries according to the subset in question
fig,axes = plt.subplots(nrows=1,ncols=3,sharex='all',figsize=[8,3.4])
if subset in biome_labels:
    df_sub=df2017[df2017['biome']==subset]
elif subset in ['Brazil','all']:
    df_sub=df2017.groupby('landcover',as_index=False).agg(sum)

sns.barplot(x='landcover',y='area_ha',hue='landcover',
                palette='Greens_d',dodge=False,ax=axes[0],
                data=df_sub)
sns.barplot(x='landcover',y='AGBpot',hue='landcover',
                palette='Greens_d',dodge=False,facecolor='white',
                ax=axes[1],data=df_sub)
sns.barplot(x='landcover',y='AGBobs',hue='landcover',
                palette='Greens_d',dodge=False,ax=axes[1],
                data=df_sub)
sns.barplot(x='landcover',y='AGBseq',hue='landcover',
                palette='Greens_d',dodge=False,ax=axes[2],
                data=df_sub)
plot_bar_CIs(np.asarray(df_sub['AGBpot_min']),np.asarray(df_sub['AGBpot_max']),axes[1],jitter=0.1)
plot_bar_CIs(np.asarray(df_sub['AGBobs_min']),np.asarray(df_sub['AGBobs_max']),axes[1],jitter=-0.1)
plot_bar_CIs(np.asarray(df_sub['AGBseq_min']),np.asarray(df_sub['AGBseq_max']),axes[2])

colours=[]
for patch in axes[0].patches:
    colours.append(patch.get_facecolor())
for ax in axes:
    ax.legend_.remove()
    ax.set_xlabel(None)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45, ha='right')
    for ii,patch in enumerate(ax.patches):
        patch.set_edgecolor(colours[ii%len(colours)])

axes[1].set_ylim(bottom=0)
axes[2].set_ylim(bottom=0)#axes[1].get_ylim())
# convert areas to 10^6 km
axes[0].set_title('Area of landcover class (2017)')
y_ticks = axes[0].get_yticks()
axes[0].set_yticklabels(['{:3.1f}'.format(i/(10.**8)) for i in y_ticks])
axes[0].set_ylabel('Area / 10$^6$ km$^2$')
# convert Mg to 10^9 Mg
axes[1].set_title('Aboveground carbon stock')
axes[1].annotate('potential (open)\nobserved (filled)',xy=(0.95,0.95),
                xycoords='axes fraction',backgroundcolor='white',ha='right',
                va='top',fontsize=10)
y_ticks = axes[1].get_yticks()
axes[1].set_yticklabels(['{:3.0f}'.format(i/(10**9)) for i in y_ticks])
axes[1].set_ylabel('Aboveground carbon / 10$^9$ Mg')
axes[2].set_title('Potential carbon defecit')
y_ticks = axes[2].get_yticks()
axes[2].set_yticklabels(['{:3.0f}'.format(i/(10**9)) for i in y_ticks])
axes[2].set_ylabel('Aboveground carbon / 10$^9$ Mg')
fig.tight_layout()
if subset in biome_labels:
    fig.savefig('%s%s_%s_%s_summary_by_landcover.png' % (path2output,country_code,version,subset))
elif subset in ['Brazil','all']:
    fig.savefig('%s%s_%s_national_summary_by_landcover_2017.png' % (path2output,country_code,version))
fig.show()

"""
#===============================================================================
PART F: Feasibility of restoration potential.

For 2%, 5%, 10%, 20%, 30% of agriculture/pasture, what is the expected carbon
sequestration for (i) random restoration; (ii) optimal restoration, based on
2018 land cover distribution

#-------------------------------------------------------------------------------
"""
AGBseq2017 = AGBpot-AGBest_2017
AGBseq2017_min = AGBpot_min-AGBest_2017_min
AGBseq2017_max = AGBpot_max-AGBest_2017_max
restoration_percentage = np.arange(0,110,5)
n_perc = restoration_percentage.size
agri_seq_opt = np.zeros(n_perc)
agri_seq_ran = np.zeros((n_perc,25))
pasture_seq_opt = np.zeros(n_perc)
pasture_seq_ran = np.zeros((n_perc,25))

AGBseq_agri = AGBseq2017[lc2017_masks['Agriculture']]
AGBseq_pasture = AGBseq2017[lc2017_masks['Pasture']]
AGBseq_agri=AGBseq_agri[np.isfinite(AGBseq_agri)]
AGBseq_pasture=AGBseq_pasture[np.isfinite(AGBseq_pasture)]

agri_class_size = AGBseq_agri.size
pasture_class_size = AGBseq_pasture.size

for ii,perc in enumerate(restoration_percentage):
    n_agri = int(agri_class_size*perc/100.)
    n_pasture = int(pasture_class_size*perc/100.)
    if perc==0:
        agri_seq_opt[ii]=0
        pasture_seq_opt[ii]=0
        agri_seq_ran[ii,:]=0
        pasture_seq_ran[ii,:]=0
    elif perc<100:
        agri_seq_opt[ii]=np.sum(np.sort(AGBseq_agri)[-n_agri:])
        pasture_seq_opt[ii]=np.sum(np.sort(AGBseq_pasture)[-n_pasture:])
        for ss in range(0,25):
            agri_seq_ran[ii,ss]=np.sum(np.random.choice(AGBseq_agri,n_agri,replace=False))
            pasture_seq_ran[ii,ss]=np.sum(np.random.choice(AGBseq_pasture,n_pasture,replace=False))
    else:
        agri_seq_opt[ii]=np.sum(AGBseq_agri)
        pasture_seq_opt[ii]=np.sum(AGBseq_pasture)
        agri_seq_ran[ii,:]=np.sum(AGBseq_agri)
        pasture_seq_ran[ii,:]=np.sum(AGBseq_pasture)

agri_seq_ran=np.mean(agri_seq_ran,axis=1)
pasture_seq_ran=np.mean(pasture_seq_ran,axis=1)

# Plot things up
fig,axes = plt.subplots(nrows=1,ncols=2,sharex='all',sharey='all',figsize=[6,3.4])

axes[0].plot(restoration_percentage,agri_seq_opt,'-',color='black',label='optimal')
axes[0].plot(restoration_percentage,agri_seq_ran,'--',color='black',label='random')
axes[1].plot(restoration_percentage,pasture_seq_opt,'-',color='black',label='optimal')
axes[1].plot(restoration_percentage,pasture_seq_ran,'--',color='black',label='random')

# convert areas to 10^6 km
axes[0].set_ylim(bottom=0)
axes[0].set_xlim(left=0,right=100)
y_ticks = axes[0].get_yticks()
axes[0].set_yticklabels(['{:3.2f}'.format(i/(10.**9)) for i in y_ticks])
axes[0].set_ylabel('Potential increase in\naboveground carbon / Pg')

axes[0].set_title('Agriculture to Natural')
axes[1].set_title('Pasture to Natural')
axes[0].legend()
axes[0].set_xlabel('Percentage of land restored')
axes[1].set_xlabel('Percentage of land restored')
fig.tight_layout()
fig.savefig('%s%s_%s_restoration_scenarios_2017.png' % (path2output,country_code,version))
fig.show()
