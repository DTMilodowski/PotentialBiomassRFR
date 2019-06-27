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
AGBpot_ds = xr.open_dataset('%s%s_%s_AGB_potential_RFR_worldclim_soilgrids.nc' %
                                (path2output, country_code,version))
AGBpot = AGBpot_ds['AGBpot8'].values
AGBobs = AGBpot_ds['AGBobs'].values
AGBseq = AGBpot-AGBobs

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

"""
# Arrays for upper and lower limits of uncertainty
potC_Mg_max = np.zeros(5)
seqC_Mg_max = np.zeros(5)
obsC_Mg_max = np.zeros(5)

potC_Mg_min = np.zeros(5)
seqC_Mg_min = np.zeros(5)
obsC_Mg_min = np.zeros(5)
"""

for cc,opp in enumerate(opp_class):
    mask = masks[opp]
    areas_ha[cc] = np.sum(cell_areas[mask])*1.
    potC_Mg[cc] = np.sum(AGBpot[mask]*cell_areas[mask])
    seqC_Mg[cc] = np.sum(AGBseq[mask]*cell_areas[mask])
    obsC_Mg[cc] = np.sum(AGBobs[mask]*cell_areas[mask])

potC_Mg_ha = potC_Mg/areas_ha
seqC_Mg_ha = seqC_Mg/areas_ha
obsC_Mg_ha = obsC_Mg/areas_ha

# opportunity Classes:
# 0   - No opportunity
# 1   - Wide-scale restoration
# 2   - Mosaic restoration
# 3   - Remote restoration
# 4   - Agricultural lands and urban areas
print('=====================================================================')
print('\trestoration opportunity areas in ha')
print('---------------------------------------------------------------------')
print('\tforest\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture')
print('\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f' % (areas_ha[0],areas_ha[1],areas_ha[2],areas_ha[3],areas_ha[4]))
print('=====================================================================')

print('=====================================================================')
print( '\tobserved biomass within each class, in 10^6 Mg C')
print( '---------------------------------------------------------------------')
print( '\tforest\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f' % (obsC_Mg[0],obsC_Mg[1],obsC_Mg[2],obsC_Mg[3],obsC_Mg[4]))
print( '---------------------------------------------------------------------')
print( '\observed biomass density within each class, in 10^6 Mg C / ha')
print( '---------------------------------------------------------------------')
print( '\tforest\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f' % (obsC_Mg_ha[0],obsC_Mg_ha[1],obsC_Mg_ha[2],obsC_Mg_ha[3],obsC_Mg_ha[4]))
print( '=====================================================================')

print( '=====================================================================')
print( '\tpotential biomass within each class, in 10^6 Mg C')
print( '---------------------------------------------------------------------')
print( '\tforest\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f' % (potC_Mg[0],potC_Mg[1],potC_Mg[2],potC_Mg[3],potC_Mg[4]))
print( '---------------------------------------------------------------------')
print( '\tforest\tpotential biomass density within each class, in 10^6 Mg C / ha')
print( '---------------------------------------------------------------------')
print( '\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f' % (potC_Mg_ha[0],potC_Mg_ha[1],potC_Mg_ha[2],potC_Mg_ha[3],potC_Mg_ha[4]))
print( '=====================================================================')

print( '=====================================================================')
print( ' AGB sequestration potential within each class, in 10^6 Mg C')
print( '---------------------------------------------------------------------')
print( '\tforest\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f' % (seqC_Mg[0],seqC_Mg[1],seqC_Mg[2],seqC_Mg[3],seqC_Mg[4]))
print( '---------------------------------------------------------------------')
print( ' AGB deficit within each class, in 10^6 Mg C / ha')
print( '---------------------------------------------------------------------')
print( '\tforest\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture')
print( '\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f' % (seqC_Mg_ha[0],seqC_Mg_ha[1],seqC_Mg_ha[2],seqC_Mg_ha[3],seqC_Mg_ha[4]))

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
