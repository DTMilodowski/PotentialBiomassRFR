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

# Plot simple map figure
def plot_xarray(xarr, figure_name,figsize_x=8,figsize_y=6,vmin=None,vmax=None,cmap='viridis',add_colorbar=False,
                extend ='neither',cbar_kwargs={},show=False):
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
    fig.savefig(figure_name)
    if show:
        fig.show()
    return fig
