"""

regrid data sets
- this routine clips and regrids the data layers to be fed into the RFR analysis to estimate potential biomass
- it predominately makes use of gdal
- the 30 arcsec worldclim2 bio data are used as the template onto which other data are regridded
- regridding modes are as follows:
  - AGB (Avitabile) - nearest neighbour
  - ESA CCI landcover - mode
  - Primary forest maps (Morgano et al.) - optional
  - Intact forest landscapes (Popatov et al.) - optional
  - SOILGRIDS - nearest neighbour
- the extent is specified based on the bounding box (lat long limits)
- if required a mask is generated to clip the output to the specified national boundaries

- the regridding routines utilise the following packages:
  - GDAL
  This is run from the command line, rather than with the python bindings

14/09/2018
David T. Milodowski

"""

# This script just calls GDAL and NCO to clip existing datasets to a specified bounding box
import os
import glob

# Bounding Box
W = -76.
E = -34.
N = 10.
S = -34.

# create some directories to host the processed data_io
prefix = 'BRA'
outdir = '/disk/scratch/local.2/PotentialBiomass/processed/'
os.system('mkdir %s' % outdir)
os.system('mkdir %s%s' % (outdir,prefix))
os.system('mkdir %s%s/wc2' % (outdir,prefix))
os.system('mkdir %s%s/agb' % (outdir,prefix))
os.system('mkdir %s%s/esacci' % (outdir,prefix))
os.system('mkdir %s%s/soilgrids' % (outdir,prefix))
os.system('mkdir %s%s/forestcover' % (outdir,prefix))
os.system('mkdir %s%s/mapbiomas' % (outdir,prefix))
os.system('mkdir %s%s/WRI_restoration' % (outdir,prefix))
os.system('mkdir %s%s/IUCN_protected_areas' % (outdir,prefix))


# Start with the worldclim2 data. This should be straightforward using gdal as are just clipping
# to the target extent
wc2dir = '/disk/scratch/local.2/worldclim2/'
wc2files = glob.glob('%swc2*tif' % wc2dir);wc2files.sort()
wc2subset = []
wc2vars = []
for ff,fname in enumerate(wc2files):
    variable = fname.split('_')[-1].split('.')[0]
    outfname = fname.split('/')[-1][:-4]
    #if int(variable) in ['01','04','05','06','12','13','14','15']:
    if int(variable) in range(1,20):
        wc2vars.append(variable)
        wc2subset.append(fname)
        os.system("gdalwarp -overwrite -dstnodata -9999 -te %f %f %f %f %s %s%s/wc2/%s_%s.tif" % (W,S,E,N,wc2files[ff],outdir,prefix,outfname,prefix))

# Soilgrids next. Soilgrids are downloaded as geotiffs. Resolution is identical to worldclim, so regridding process is the same
sgdir = '/disk/scratch/local.2/soilgrids/1km/'
sgfiles = glob.glob('%s*_1km_ll.tif' % sgdir);sgfiles.sort()
sgsubset = []
sgvars = []
for ff,fname in enumerate(sgfiles):
    variable = fname.split('/')[-1][:-11]
    outfname = fname.split('/')[-1][:-4]
    sgvars.append(variable)
    sgsubset.append(fname)
    os.system("gdalwarp -overwrite -te %f %f %f %f %s %s%s/soilgrids/%s_%s.tif" % (W,S,E,N,sgfiles[ff],outdir,prefix,outfname,prefix))

# Aboveground Biomass - Avitabile map - 1 km resolution so same gdal example should be sufficient
agbfiles= ['/exports/csce/datastore/geos/groups/gcel/AGB/avitabile/Avitabile_AGB_Map/Avitabile_AGB_Map.tif','/home/dmilodow/DataStore_GCEL/AGB/avitabile/Avitabile_AGB_Uncertainty/Avitabile_AGB_Uncertainty.tif']
agbvars = ['Avitabile_AGB','Avitabile_AGB_Uncertainty']
os.system("gdalwarp -overwrite -dstnodata -9999 -te %f %f %f %f %s %s%s/agb/%s_%s_1km.tif" % (W,S,E,N,agbfiles[0],outdir,prefix,agbvars[0],prefix))
os.system("gdalwarp -overwrite -dstnodata -9999 -te %f %f %f %f %s %s%s/agb/%s_%s_1km.tif" % (W,S,E,N,agbfiles[1],outdir,prefix,agbvars[1],prefix))

# ESA CCI landcover - the original resolution of these rasters is higher than the reference data, so this needs to be regridded. The mode landcover class will be used.
# Note that the ESA CCI landcover data are originally in netcdf format.
lcdir = '/home/dmilodow/DataStore_GCEL/ESA_CCI_landcover/'
lcfiles = glob.glob('%s*.nc' % lcdir);lcfiles.sort()
for ff,fname in enumerate(lcfiles):
    outfname = fname.split('/')[-1][:22]+fname.split('/')[-1][27:42]+"-1km-mode-lccs-class"
    os.system("gdalwarp -overwrite -te %f %f %f %f -tr 0.008333333333333 -0.008333333333333 -r mode -of GTIFF NETCDF:%s:lccs_class %s%s/esacci/%s-%s.tif" % (W,S,E,N,lcfiles[ff],outdir,prefix,outfname,prefix))
    outfname = fname.split('/')[-1][:22]+fname.split('/')[-1][27:42]+"-1km-mode-change-count"
    os.system("gdalwarp -overwrite -te %f %f %f %f -tr 0.008333333333333 -0.008333333333333 -r mode -of GTIFF NETCDF:%s:change_count %s%s/esacci/%s-%s.tif" % (W,S,E,N,lcfiles[ff],outdir,prefix,outfname,prefix))

# INtact forest landscapes (Potapov et al.)
#ifl_file = '/disk/scratch/local.2/landcover_maps/IFL_2013/IFL_2013.tif'
#os.system("gdalwarp -overwrite -te %f %f %f %f -tr 0.008333333333333 -0.008333333333333 -r mode -of GTIFF %s %s%s/forestcover/IFL_2013_%s.tif" % (W,S,E,N,ifl_file,outdir,prefix,prefix))
hfl_file = '/disk/scratch/local.2/global_hinterland_forests/source_files/SAM_2013hinterland_25vcf.tif'
#os.system("gdal_translate -tr 0.008333333333333 -0.008333333333333 -r mode -of GTIFF %s HFL_temp.tif" % (hfl_file))
os.system("gdalwarp -overwrite -te %f %f %f %f -tr 0.008333333333333 -0.008333333333333 -r mode -of GTIFF %s %s%s/forestcover/HFL_2013_%s.tif" % (W,S,E,N,hfl_file,outdir,prefix,prefix))

# MAPBIOMAS
mbdir = '/disk/scratch/local.2/MAPBIOMAS/'
mbfiles = glob.glob('%s*_1km.tif' % mbdir);mbfiles.sort()
for ff,fname in enumerate(mbfiles):
    outfname =  mbfiles[ff].split('/')[-1].split('.')[0]
    print(fname)
    os.system("gdalwarp -overwrite -te %f %f %f %f -tr 0.008333333333333 -0.008333333333333 -r mode -of GTIFF %s %s%s_1km.tif" % (W,S,E,N,fname,mbdir,outfname))

os.system("gdal_merge.py -init 0 -n 0 -a_nodata 0 -ul_lr %f %f %f %f -ps 0.008333333333333 -0.008333333333333 -o %s%s/mapbiomas/BRAZIL_mapbiomas_1km.tif %s*_1km.tif" % (W,N,E,S,outdir,prefix,mbdir))

# WRI restoration opportunities
wri_file = '/disk/scratch/local.2/WRI_restoration_opportunities/WRI_restoration/WRI_restoration_opportunities.tif'
os.system("gdalwarp -overwrite -te %f %f %f %f -tr 0.008333333333333 -0.008333333333333 -r mode -of GTIFF %s %s%s/WRI_restoration/WRI_restoration_opportunities_%s.tif" % (W,S,E,N,wri_file,outdir,prefix,prefix))

# IUCN Protected Areas
pa_file = '/disk/scratch/local.2/IUCN/WDPA_Jul2019_land.shp'
os.system('gdal_rasterize -burn 1 -te %f %f %f %f -tr 0.008333333333333 -0.008333333333333 -of GTIFF %s %s%s/IUCN_protected_areas/IUCN_protected_areas_%s.tif' % (W,S,E,N,pa_file,outdir,prefix,prefix))
