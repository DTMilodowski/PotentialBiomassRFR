mask"""
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

country_code = 'BRA'
version = '008'

"""
#===============================================================================
PART A: DEFINE PATHS AND LOAD IN DATA
- Potential biomass maps (from netcdf file)
- Biome map (Mapbiomas)
- WRI opportunity map
#-------------------------------------------------------------------------------
"""
opportunities_file = '/home/dmilodow/DataStore_DTM/FOREST2020/WRI_restoration_opportunities/WRI_restoration/WRI_restoration_opportunities_regrid_tropics.tif'
path2data = '/disk/scratch/local.2/dmilodow/PotentialBiomass/processed/%s/' % country_code
path2output = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/output/'
path2mapbiomas = '/scratch/local.2/MAPBIOMAS/'

# load potential biomass models from netdf file
AGBpot_ds = xr.open_dataset(%s%s_%s_AGB_potential_RFR_worldclim_soilgrids.nc' % (path2output,
                                country_code,version)
AGBpot = AGBpot_ds['AGBpot8'].values
AGBobs = AGBpot_ds['AGBobs'].values
AGBseq = AGBpot-AGBobs
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
    masks[biome_label[ii]] = mb.values>0

# load opportunity map
opportunity = xr.open_rasterio('%sWRIopportunities/WRI_restoration_opportunities_%s.tif' % (path2data, country_code))[0]
opp_class = ['wide-scale','mosaic','remote','agriculture']
for cc,opp in enumerate(opp_class):
    masks[opp] = opportunity.values==cc+1

"""
#===============================================================================
PART B: National summaries
#-------------------------------------------------------------------------------
"""

# Get Brazil national boundaries
# - Summarise each of the opportunity classes
opp_class = ['wide-scale','mosaic','remote','agriculture']
areas_ha = np.zeros(4)
potC_Mg = np.zeros(4)
seqC_Mg = np.zeros(4)
defC_Mg = np.zeros(4)
obsC_Mg = np.zeros(4)

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
# 4   - Agricultural lands
print '====================================================================='
print '\trestoration opportunity areas in ha'
print '---------------------------------------------------------------------'
print '\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture'
print '\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f' % (areas_ha[0],areas_ha[1],areas_ha[2],areas_ha[3])
print '====================================================================='

print '====================================================================='
print '\tobserved biomass within each class, in 10^6 Mg C'
print '---------------------------------------------------------------------'
print '\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture'
print '\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f' % (obsC_Mg[0],obsC_Mg[1],obsC_Mg[2],obsC_Mg[3])
print '---------------------------------------------------------------------'
print '\observed biomass density within each class, in 10^6 Mg C / ha'
print '---------------------------------------------------------------------'
print '\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture'
print '\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f' % (obsC_Mg_ha[0],obsC_Mg_ha[1],obsC_Mg_ha[2],obsC_Mg_ha[3])
print '====================================================================='

print '====================================================================='
print '\tpotential biomass within each class, in 10^6 Mg C'
print '---------------------------------------------------------------------'
print '\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture'
print '\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f' % (potC_Mg[0],potC_Mg[1],potC_Mg[2],potC_Mg[3])
print '---------------------------------------------------------------------'
print '\tpotential biomass density within each class, in 10^6 Mg C / ha'
print '---------------------------------------------------------------------'
print '\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture'
print '\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f' % (potC_Mg_ha[0],potC_Mg_ha[1],potC_Mg_ha[2],potC_Mg_ha[3])
print '====================================================================='

print '====================================================================='
print ' AGB sequestration potential within each class, in 10^6 Mg C'
print '---------------------------------------------------------------------'
print '\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture'
print '\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f' % (seqC_Mg[0],seqC_Mg[1],seqC_Mg[2],seqC_Mg[3])
print '---------------------------------------------------------------------'
print ' AGB deficit within each class, in 10^6 Mg C / ha'
print '---------------------------------------------------------------------'
print '\twide-scale,\t\tmosaic,\t\tremote,\t\tagriculture'
print '\t%.0f,\t\t%.0f,\t\t%.0f,\t\t%.0f' % (seqC_Mg_ha[0],seqC_Mg_ha[1],seqC_Mg_ha[2],seqC_Mg_ha[3])

"""
#===============================================================================
PART C: Biome level summaries
#-------------------------------------------------------------------------------
"""


"""
#===============================================================================
PART D: Breakdown of potential biomass by landcover type for feasible
restoration areas
#-------------------------------------------------------------------------------
"""
