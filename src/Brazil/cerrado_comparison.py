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
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('../')
import useful as useful
sns.set()#style="whitegrid")

"""
A quick function to plot error bars onto bar plot
"""
def plot_bar_CIs(lc,uc,ax,jitter=0,positions=[]):
    if len(positions)==0:
        positions = np.arange(uc.size,dtype='float')
    else:
        positions=np.asarray(positions)
    positions+=jitter
    for ii,pos in enumerate(positions):
        ax.plot([pos,pos],[lc[ii],uc[ii]],'-',lw=3.8,color='white')
        ax.plot([pos,pos],[lc[ii],uc[ii]],'-',lw=2,color='0.5')
    return 0


country_code = 'BRA'
version = '013'

"""
#===============================================================================
PART A: DEFINE PATHS AND LOAD IN DATA
- Potential biomass maps (from netcdf file)
- Biome boundaries (Mapbiomas)
- Globbiomass
- Field data
#-------------------------------------------------------------------------------
"""
path2data = '/disk/scratch/local.2/PotentialBiomass/processed/%s/' % country_code
path2output = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/output/' # change to either your own storage space or create a directory on racadal scratch ('/disk/scratch/local.2/)
path2mapbiomas = '/scratch/local.2/MAPBIOMAS/'

# load potential biomass models from netdf file
AGBpot_ds = xr.open_dataset('%s%s_%s_AGB_potential_RFR_avitabile_worldclim_soilgrids_final.nc' %
                                (path2output, country_code,version))
AGBpot     = AGBpot_ds['AGBpot'].values
AGBobs     = AGBpot_ds['AGBobs'].values
AGBpot_min = AGBpot_ds['AGBpot_min'].values
AGBobs_min = AGBpot_ds['AGBobs_min'].values
AGBpot_min[AGBpot_min<0]=0
AGBobs_min[AGBobs_min<0]=0
AGBpot_max = AGBpot_ds['AGBpot_max'].values
AGBobs_max = AGBpot_ds['AGBobs_max'].values

lat = AGBpot_ds.coords['lat'].values
lon = AGBpot_ds.coords['lon'].values
cell_areas =  useful.get_areas(latorig=lat,lonorig=lon)
cell_areas/=10**4 # m^2 -> ha

# load biome boundaries
# - for each biome, load in the 1km regridded mapbiomas dataset for that biome
#   and create a mask based on the pixels for which there is data
biome_files = glob.glob('%s/Collection4/*1km.tif' % path2mapbiomas)
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

# Load mapbiomas data for 2008
mb2005 = useful.load_mapbiomas('BRA',timestep=20,aggregate=3)

# Load in Globbiomass
gb = xr.open_rasterio('/disk/scratch/local.2/PotentialBiomass/processed/BRA/agb/globbiomass/BRA_globbiomass_agb_1km.tif')[0]
gb_unc = xr.open_rasterio('/disk/scratch/local.2/PotentialBiomass/processed/BRA/agb/globbiomass/BRA_globbiomass_agb_err_1km.tif')[0]

# load in plot data
roitman_site_level_data = pd.read_csv('/home/dmilodow/DataStore_DTM/FOREST2020/Cerrado/FieldData/Roitman_etal_PNAS_2018/TableS3_aggregated_site_level_data.csv',
                                        usecols=np.arange(0,13),encoding='ISO-8859-1').dropna()
roitman_site_level_data.rename(columns={'Biomass (ton ha-1)':'AGB','BiomassInferior  95% CIL (ton ha-1)':'AGB_95_lower',
                    'BiomassSuperior 95% CIL (ton ha-1)':'AGB_95_upper',
                    'Longitude (decimal)':'lon','Latitude (decimal)':'lat'}, inplace=True)

n_field_sites = roitman_site_level_data.shape[0]

df = pd.DataFrame({'Site':roitman_site_level_data['Site Number'],
'lat':roitman_site_level_data.lat.astype('float'),
'lon':roitman_site_level_data.lon.astype('float'),
'AGB_roitman':roitman_site_level_data.AGB.astype('float'),
'AGB_roitman_lower':roitman_site_level_data.AGB_95_lower.astype('float'),
'AGB_roitman_upper':roitman_site_level_data.AGB_95_upper.astype('float'),
'AGB_avitabile':np.zeros(n_field_sites),
'AGB_avitabile_lower':np.zeros(n_field_sites),
'AGB_avitabile_upper':np.zeros(n_field_sites),
'AGBpot':np.zeros(n_field_sites),
'AGBpot_lower':np.zeros(n_field_sites),
'AGBpot_upper':np.zeros(n_field_sites),
'AGB_globbiomass':np.zeros(n_field_sites),
'AGB_globbiomass_lower':np.zeros(n_field_sites),
'AGB_globbiomass_upper':np.zeros(n_field_sites)})

for row in range(n_field_sites):
    # get sample locations from maps
    col_idx = np.argmin(np.abs(lon-float(roitman_site_level_data.lon[row])))
    row_idx = np.argmin(np.abs(lat-float(roitman_site_level_data.lat[row])))
    df.AGB_avitabile[row] = AGBobs[row_idx,col_idx]
    df.AGB_avitabile_upper[row] = AGBobs_max[row_idx,col_idx]
    df.AGB_avitabile_lower[row] = AGBobs_min[row_idx,col_idx]
    df.AGBpot[row] = AGBpot[row_idx,col_idx]
    df.AGBpot_upper[row] = AGBpot_max[row_idx,col_idx]
    df.AGBpot_lower[row] = AGBpot_min[row_idx,col_idx]
    df.AGB_globbiomass[row] = gb.values[row_idx,col_idx]
    df.AGB_globbiomass_upper[row] = gb.values[row_idx,col_idx]+gb_unc.values[row_idx,col_idx]
    df.AGB_globbiomass_lower[row] = np.max((gb.values[row_idx,col_idx]-gb_unc.values[row_idx,col_idx],0))



"""
#===============================================================================
PART B: COMPARISON OF PLOT VS REMOTE SENSING (Avitabile/Globbiomass) & POTENTIAL
AGB (based on Avitabile)
- Direct comparison in scatter plot
- Distribution comparisons
#-------------------------------------------------------------------------------
"""
sns.set()
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(8,10),sharex=True,sharey=True)
titles = ['a) Field (Roitman et al.) vs.\nAvitabile et al.','b) Field (Roitman et al.) vs.\nGlobbiomass']

# Plot comparisons against field data
remote_AGB = [df.AGB_avitabile.copy(),df.AGB_globbiomass.copy()]
remote_AGB_l = [df.AGB_avitabile_lower.copy(),df.AGB_globbiomass_lower.copy()]
remote_AGB_u = [df.AGB_avitabile_upper.copy(),df.AGB_globbiomass_upper.copy()]
for ii,ax in enumerate(axes):
    mask=np.isfinite(df.AGB_roitman*remote_AGB[ii])
    _,_,r,p,_ = stats.linregress(df.AGB_roitman[mask],remote_AGB[ii][mask])
    rmse = np.sqrt(np.mean((df.AGB_roitman[mask]-remote_AGB[ii][mask])**2))
    label = 'R$^2$ = %.02f\nRMSE = %.02f' % (r**2,rmse)

    ax.set_title(titles[ii])
    ax.plot([0,80],[0,80],':',color='0.5')
    ax.errorbar(df.AGB_roitman,remote_AGB[ii],
                    xerr=(df.AGB_roitman-df.AGB_roitman_lower,df.AGB_roitman_upper-df.AGB_roitman),
                    yerr=(remote_AGB[ii]-remote_AGB_l[ii], remote_AGB_u[ii]-remote_AGB[ii]),
                    marker='',linestyle='',color='#0088aa',linewidth=0.5)
    ax.plot(df.AGB_roitman,remote_AGB[ii],'o',color='black')
    ax.annotate(label, xy=(0.95,0.95), xycoords='axes fraction', fontsize=10,
                backgroundcolor='none', ha='right', va='top')
    ax.set_xlabel('AGB (field) [Mg ha $^{-1}$]')
    ax.set_aspect('equal')
    ax.set_xlim(left=0,right=80)
    ax.set_ylim(bottom=0,top=230)

axes[0].set_ylabel('AGB (remote) [Mg ha $^{-1}$]')

fig.show()



from sklearn.neighbors import KernelDensity


AGBdist={}
grid = np.linspace(-49.5,249.5,300)
kde = KernelDensity(bandwidth=2,kernel='gaussian')
kde.fit(df.AGB_roitman[np.isfinite(df.AGB_roitman)][:,None])
AGBdist['Roitman et al'] = np.exp(kde.score_samples(grid[:,None]))
kde.fit(df.AGB_avitabile[np.isfinite(df.AGB_avitabile)][:,None])
AGBdist['Avitabile et al plots'] = np.exp(kde.score_samples(grid[:,None]))
kde.fit(df.AGB_globbiomass[np.isfinite(df.AGB_globbiomass)][:,None])
AGBdist['Globbiomass plots'] = np.exp(kde.score_samples(grid[:,None]))
kde.fit(df.AGBpot[np.isfinite(df.AGBpot)][:,None])
AGBdist['Potential plots'] = np.exp(kde.score_samples(grid[:,None]))
# Savanna formation
kde.fit(AGBobs[masks['Cerrado']*np.isfinite(AGBobs)*(mb2005==2)][:,None])
AGBdist['Avitabile et al'] = np.exp(kde.score_samples(grid[:,None]))
kde.fit(gb.values[masks['Cerrado']*np.isfinite(gb.values)*(mb2005==2)][:,None])
AGBdist['Globbiomass'] = np.exp(kde.score_samples(grid[:,None]))
kde.fit(AGBpot[masks['Cerrado']*np.isfinite(AGBpot)*(mb2005==2)][:,None])
AGBdist['Potential'] = np.exp(kde.score_samples(grid[:,None]))
# Forest formation
kde.fit(AGBobs[masks['Cerrado']*np.isfinite(AGBobs)*(mb2005==1)][:,None])
AGBdist['Avitabile et al FF'] = np.exp(kde.score_samples(grid[:,None]))
kde.fit(gb.values[masks['Cerrado']*np.isfinite(gb.values)*(mb2005==1)][:,None])
AGBdist['Globbiomass FF'] = np.exp(kde.score_samples(grid[:,None]))
kde.fit(AGBpot[masks['Cerrado']*np.isfinite(AGBpot)*(mb2005==1)][:,None])
AGBdist['Potential FF'] = np.exp(kde.score_samples(grid[:,None]))

# apply reflective lower boundary at 0
n_neg = np.sum(grid<0)
for kk in AGBdist.keys():
    AGBdist[kk][n_neg:2*n_neg]+=AGBdist[kk][:n_neg][::-1]
    AGBdist[kk][:n_neg]=np.nan

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,6))
ax.plot(grid,AGBdist['Roitman et al'],'-',color='black',linewidth=2,label='Roitman et al (field)')
ax.plot(grid,AGBdist['Avitabile et al FF'],'-',color='red',linewidth=2,label='Avitabile et al (FF)')
ax.plot(grid,AGBdist['Globbiomass FF'],'-',color='blue',linewidth=2,label='Globbiomass (FF)')
ax.plot(grid,AGBdist['Potential FF'],'-',color='orange',linewidth=2,label='Potential (FF)')
ax.plot(grid,AGBdist['Avitabile et al'],':',color='red',linewidth=2,label='Avitabile et al')
ax.plot(grid,AGBdist['Globbiomass'],':',color='blue',linewidth=2,label='Globbiomass')
ax.plot(grid,AGBdist['Potential'],':',color='orange',linewidth=2,label='Potential')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.set_xlabel('AGB [Mg ha$^{-1}$]')
ax.set_ylabel('Frequency density')
ax.legend(loc = 'upper right')
fig.tight_layout()
fig.show()
#fig.savefig('../figures/manuscript/figS8_agb_distributions.png',bbox_inches='tight')



"""
#===============================================================================
PART C: BREAKDOWN OF C STOCKS IN CERRADO BY LAND COVER
#-------------------------------------------------------------------------------
"""
lc_class = ['Forest formation','Savanna formation','Mangrove','Plantation','Wetland','Grassland',
'Pasture','Agriculture','Mosaic agri-pasture','Urban','Other']

colours = np.asarray(['#1f4423','#32cd32','#687537','#935132','#45c2a5','#b8af4f',
'#ffd966','#e974ed','#ffefc3', '#af2a2a','#d5d5e5'])
lc_val = [[3],[4],[5],[9],[11],[12],[15],[19,20],[21],[24],[23,29,30,32,13]]
lc_masks={}
for cc,lc in enumerate(lc_class):
    lc_masks[lc] = np.zeros(mb2005_full.shape,dtype='bool')
    for val in lc_val[cc]:
        lc_masks[lc][mb2005_full==val]=True

        landcover_class = []
        area = []
        agbobs = []
        agbpot = []
        agbobs_min = []
        agbpot_min = []
        agbobs_max = []
        agbpot_max = []

        for cc,lc in enumerate(lc_class):
            mask = lc_masks[lc]*masks['Cerrado']*np.isfinite(AGBobs)
            landcover_class.append(lc)
            area.append(np.sum(cell_areas[mask])*1.)
            agbpot.append(np.sum(AGBpot[mask]*cell_areas[mask]))
            agbobs.append(np.sum(AGBobs[mask]*cell_areas[mask]))
            agbpot_min.append(np.sum(AGBpot_min[mask]*cell_areas[mask]))
            agbobs_min.append(np.sum(AGBobs_min[mask]*cell_areas[mask]))
            agbpot_max.append(np.sum(AGBpot_max[mask]*cell_areas[mask]))
            agbobs_max.append(np.sum(AGBobs_max[mask]*cell_areas[mask]))

            df = pd.DataFrame({'landcover':landcover_class,'area_ha':area,
            'AGBobs':agbobs,'AGBobs_min':agbobs_min,'AGBobs_max':agbobs_max,
            'AGBpot':agbpot,'AGBpot_min':agbpot_min,'AGBpot_max':agbpot_max,})

            # Plot observed vs. potential AGB
            fig,axis = plt.subplots(nrows=1,ncols=1,figsize=[4,3.4])

            # Create discrete color ramp
            lc_palette = sns.color_palette(colours).as_hex()

            for var in df.keys():
                if 'AGB' in var:
                    df[var]=df[var].div(10**9)

                    sns.barplot(x='landcover',y='AGBpot',hue='landcover',palette=lc_palette,dodge=False,
                    facecolor='white',ax=axis,order=lc_class,data=df)
                    sns.barplot(x='landcover',y='AGBobs',hue='landcover',palette=lc_palette,dodge=False,
                    ax=axis,order=lc_class,data=df)

                    plot_bar_CIs(np.asarray(df['AGBpot_min']),np.asarray(df['AGBpot_max']),axis,jitter=0.1,positions=positions)
                    plot_bar_CIs(np.asarray(df['AGBobs_min']),np.asarray(df['AGBobs_max']),axis,jitter=-0.1,positions=positions)

                    colours=[]
                    for patch in axes[1].patches:
                        colours.append(patch.get_facecolor())
                        for ax in axes:
                            ax.legend_.remove()
                            ax.set_xlabel(None)
                            ax.set_xticklabels(ax.get_xticklabels(),rotation=45, ha='right')
                            for ii,patch in enumerate(ax.patches):
                                patch.set_edgecolor(colours[ii%len(colours)])

                                axis.set_ylim(bottom=0)
                                axis.set_title('Aboveground carbon stock')
                                axis.annotate('potential (open)\nobserved (filled)',xy=(0.95,0.95),
                                xycoords='axes fraction',backgroundcolor='white',ha='right',
                                va='top',fontsize=10)
                                axis.set_ylabel('Aboveground carbon / Pg')
                                fig.tight_layout()
                                fig.savefig('%s%s_%s_cerrado_summary_by_landcover_detail.png' % (path2output,country_code,version))
                                fig.show()
