"""
simple_figures
--------------------------------------------------------------------------------
GENERATE FIGURES REQUIRED FOR METHODS DESCRIPTION
This is a set of scripts to produce simple figures from potential biomass work

21/03/2019 - D. T. Milodowski
--------------------------------------------------------------------------------
"""
import numpy as np                  # standard package for scientific computing
import xarray as xr                 # xarray geospatial package
import matplotlib.pyplot as plt     # plotting package
import seaborn as sns               # another useful plotting package
sns.set()                           # set some nice default plotting options
from matplotlib.colors import ListedColormap
# Plot simple map figure
def plot_xarray(xarr, figure_name,figsize_x=8,figsize_y=6,vmin=None,vmax=None,cmap='viridis',add_colorbar=False,
                extend ='neither',cbar_kwargs={},show=False,title=""):
    if vmin is None:
        vmin = np.nanmin(xarr)
    if vmax is None:
        vmax =np.nanmax(xarr)
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(figsize_x,figsize_y))
    if add_colorbar:
        xarr.plot(ax=axis, vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=add_colorbar,
                    extend=extend, cbar_kwargs=cbar_kwargs)
    else:
        xarr.plot(ax=axis, vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=add_colorbar)
    axis.set_aspect("equal")
    axis.set_title(title,fontsize=16)
    fig.savefig(figure_name)
    if show:
        fig.show()
    return fig,axis

# function to make ESA CCI colormap
#  1. Agriculture
#  2. Forest
#  3. Grassland
# 4. Wetland
# 5. Settlement
# 6. Shrub
# 7. Lichens/Mosses
# 8. Sparse
# 9. Bare
# 10. Water
# 11. Ice
def esa_cci_colormap(landcover, level=0):
    if level==0:
        id =    np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        labels = np.asarray(['agriculture','forest','grassland','wetland','urban',   'shrub','lichens','sparse','bare','water','ice'])
        colours = np.asarray(['#ffff99', '#33a02c', '#b2df8a', '#6a3d9a', "#000000", '#ff7f00','#cab2d6','#fdbf6f','#e31a1c','#1f78b4','#a6cee3'])

        id_temp,idx_landcover,idx_id = np.intersect1d(landcover,id,return_indices=True)
        id = id[idx_id]
        labels=labels[idx_id]
        colours=colours[idx_id]
        lc_cmap = ListedColormap(sns.color_palette(colours).as_hex())

    return lc_cmap,labels


# Aggregate classes
# 0 = simplest (11 categories); 1 = level one subclasses; 2 = full set of subclasses
def aggregate_classes(landcover,level = 0):
    if level==0:
        # Agri
        landcover[np.all((landcover>=10,landcover<50),axis=0)]=1
        # Forest
        landcover[np.all((landcover>=50,landcover<110),axis=0)]=2
        landcover[np.any((landcover==160,landcover==170),axis=0)]=2
        # Grass
        landcover[np.any((landcover==110,landcover==130),axis=0)]=3
        # Wetland
        landcover[landcover==180]=4
        # Settlement
        landcover[landcover==190]=5
        # Shrub
        landcover[np.all((landcover>=120,landcover<130),axis=0)]=6
        # Lichens
        landcover[landcover==140]=7
        # Sparse
        landcover[np.all((landcover>=150,landcover<160),axis=0)]=8
        # Bare
        landcover[np.all((landcover>=200,landcover<210),axis=0)]=9
        # Water
        landcover[landcover==210]=10
        # Ice
        landcover[landcover==220]=11

    if level==1:
        landcover=landcover/10

    return landcover
