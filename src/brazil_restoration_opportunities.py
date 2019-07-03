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

country_code = 'BRA'
version = '008'

"""
#===============================================================================
PART A: DEFINE PATHS AND LOAD IN DATA
- Potential biomass maps (from netcdf file)
- Biome boundaries (Mapbiomas)
- WRI opportunity map
#-------------------------------------------------------------------------------
"""
path2data = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
path2output = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/output/'
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
opp_class = ['forest','wide-scale','mosaic','remote','urban-agriculture']
for cc,opp in enumerate(opp_class):
    masks[opp] = (opportunity.values==cc)*masks['Brazil']

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
print('=====================================================================')
print('\trestoration opportunity areas in 10^6 ha')
print('---------------------------------------------------------------------')
print('\tforest\t\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture')
print('\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (areas_ha[0]/10.**6,
    areas_ha[1]/10.**6,areas_ha[2]/10.**6,areas_ha[3]/10.**6,areas_ha[4]/10.**6))
print('=====================================================================')

print('=====================================================================')
print( '\tobserved biomass within each class, in 10^6 Mg C')
print( '---------------------------------------------------------------------')
print( '\tforest\t\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (obsC_Mg[0]/10.**6,
        obsC_Mg[1]/10.**6,obsC_Mg[2]/10.**6,obsC_Mg[3]/10.**6,obsC_Mg[4]/10.**6))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (obsC_Mg_min[0]/10.**6,
        obsC_Mg_min[1]/10.**6,obsC_Mg_min[2]/10.**6,obsC_Mg_min[3]/10.**6,obsC_Mg_min[4]/10.**6))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (obsC_Mg_max[0]/10.**6,
        obsC_Mg_max[1]/10.**6,obsC_Mg_max[2]/10.**6,obsC_Mg_max[3]/10.**6,obsC_Mg_max[4]/10.**6))
print( '---------------------------------------------------------------------')
print( '\observed biomass density within each class, in 10^6 Mg C / ha')
print( '---------------------------------------------------------------------')
print( '\tforest\t\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (obsC_Mg_ha_min[0],
        obsC_Mg_ha_min[1],obsC_Mg_ha_min[2],obsC_Mg_ha_min[3],obsC_Mg_ha_min[4]))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (obsC_Mg_ha[0],
        obsC_Mg_ha[1],obsC_Mg_ha[2],obsC_Mg_ha[3],obsC_Mg_ha[4]))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (obsC_Mg_ha_max[0],
        obsC_Mg_ha_max[1],obsC_Mg_ha_max[2],obsC_Mg_ha_max[3],obsC_Mg_ha_max[4]))
print( '=====================================================================')

print( '=====================================================================')
print( '\tpotential biomass within each class, in 10^6 Mg C')
print( '---------------------------------------------------------------------')
print( '\tforest\t\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (potC_Mg[0]/10.**6,
    potC_Mg[1]/10.**6,potC_Mg[2]/10.**6,potC_Mg[3]/10.**6,potC_Mg[4]/10.**6))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (potC_Mg_min[0]/10.**6,
    potC_Mg_min[1]/10.**6,potC_Mg_min[2]/10.**6,potC_Mg_min[3]/10.**6,potC_Mg_min[4]/10.**6))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (potC_Mg_max[0]/10.**6,
    potC_Mg_max[1]/10.**6,potC_Mg_max[2]/10.**6,potC_Mg_max[3]/10.**6,potC_Mg_max[4]/10.**6))
print( '---------------------------------------------------------------------')
print( '\tforest\t\tpotential biomass density within each class, in 10^6 Mg C / ha')
print( '---------------------------------------------------------------------')
print( '\tforest\t\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (potC_Mg_ha[0],
    potC_Mg_ha[1],potC_Mg_ha[2],potC_Mg_ha[3],potC_Mg_ha[4]))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (potC_Mg_ha_min[0],
    potC_Mg_ha_min[1],potC_Mg_ha_min[2],potC_Mg_ha_min[3],potC_Mg_ha_min[4]))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (potC_Mg_ha[0],
    potC_Mg_ha_max[1],potC_Mg_ha_max[2],potC_Mg_ha_max[3],potC_Mg_ha_max[4]))
print( '=====================================================================')

print( '=====================================================================')
print( ' AGB sequestration potential within each class, in 10^6 Mg C')
print( '---------------------------------------------------------------------')
print( '\tforest\t\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (seqC_Mg[0]/10.**6,
    seqC_Mg[1]/10.**6,seqC_Mg[2]/10.**6,seqC_Mg[3]/10.**6,seqC_Mg[4]/10.**6))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (seqC_Mg_min[0]/10.**6,
    seqC_Mg_min[1]/10.**6,seqC_Mg_min[2]/10.**6,seqC_Mg_min[3]/10.**6,seqC_Mg_min[4]/10.**6))
print( '\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f,\t\t%.2f' % (seqC_Mg_max[0]/10.**6,
    seqC_Mg_max[1]/10.**6,seqC_Mg_max[2]/10.**6,seqC_Mg_max[3]/10.**6,seqC_Mg_max[4]/10.**6))
print( '---------------------------------------------------------------------')
print( ' AGB deficit within each class, in 10^6 Mg C / ha')
print( '---------------------------------------------------------------------')
print( '\tforest\t\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture')
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
sns.set(style="whitegrid")
fig,axes = plt.subplots(nrows=1,ncols=3,sharex='all',figsize=[8,3.4])
sns.barplot(x=opp_class,y=areas_ha,hue=opp_class,palette='Greens_d',dodge=False,
            ax=axes[0])
sns.barplot(x=opp_class,y=potC_Mg,hue=opp_class,palette='Greens_d',dodge=False,
            ax=axes[1],facecolor='white')
sns.barplot(x=opp_class,y=obsC_Mg,hue=opp_class,palette='Greens_d',dodge=False,
            ax=axes[1])
sns.barplot(x=opp_class,y=seqC_Mg,hue=opp_class,palette='Greens_d',dodge=False,
            ax=axes[2])
colours=[]
for patch in axes[0].patches:
    colours.append(patch.get_facecolor())
for ax in axes:
    ax.legend_.remove()
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45, ha='right')
    for ii,patch in enumerate(ax.patches):
        patch.set_edgecolor(colours[ii%len(colours)])
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
axes[1].set_ylabel('Aboveground carbon / 10$^9$ Mg')
axes[2].set_title('Potential carbon defecit')
y_ticks = axes[2].get_yticks()
axes[2].set_yticklabels(['{:3.0f}'.format(i/(10**9)) for i in y_ticks])
axes[2].set_ylabel('Aboveground carbon / 10$^9$ Mg')
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

for bb, bio in enumerate(biome_labels):
    for cc,opp in enumerate(opp_class):
        mask = masks[opp]*masks[bio]
        biome.append(bio)
        opportunity_class.append(opp)
        area.append(np.sum(cell_areas[mask])*1.)
        agbpot.append(np.sum(AGBpot[mask]*cell_areas[mask]))
        agbseq.append(np.sum(AGBseq[mask]*cell_areas[mask]))
        agbobs.append(np.sum(AGBobs[mask]*cell_areas[mask]))

df = pd.DataFrame({'biome':biome,'opportunity class':opportunity_class,
                    'area_ha':area,'AGBobs':agbobs,'AGBpot':agbpot,
                    'AGBseq':agbseq})


"""
#===============================================================================
PART D: Breakdown of potential biomass by landcover type for feasible
restoration areas - not implemented at present
#-------------------------------------------------------------------------------
"""
n_biomes = len(biome_labels)
biome_display_labels = biome_labels.copy()
biome_display_labels[3] = 'Mata\nAtlantica'

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

#if ii//n_biomes>0:
#    ax.set_ylim(axes[1].get_ylim())

axes[0][0].set_title('Area of opportunity\nclass / 10$^6$ km$^2$')
axes[0][1].set_title('Aboveground carbon\nstock / 10$^9$ Mg')
axes[0][1].annotate('potential (open)\nobserved (filled)',xy=(0.95,0.90),
                xycoords='axes fraction',backgroundcolor='white',ha='right',
                va='top',fontsize=10)
axes[0][2].set_title('Potential carbon\ndefecit / 10$^9$ Mg')

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
