import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torchmetrics
import einops
import seaborn as sns
import torch
import geopandas as gpd
import xarray as xr

def contrast_stretch(x, lower_percent=1, upper_percent=99, lower=None, upper=None):
    """
    Perform contrast stretching on an image x.
    :param x: numpy array, the image x to be stretched.
    :param lower_percent: lower percentile for stretch.
    :param upper_percent: upper percentile for stretch.
    :return: stretched image x.
    """
    if x.ndim > 2:
        # do it per band (band assumed to be dim 0)
        return np.stack([contrast_stretch(b, lower_percent, upper_percent) for b in x])
    if not lower:
        lower = np.nanpercentile(x, lower_percent)
    if not upper:
        upper = np.nanpercentile(x, upper_percent)
    stretched_band = np.clip((x - lower) / (upper - lower), 0, 1)

    return stretched_band

def contrast_stretch_xr(x, lower_percent=1, upper_percent=99, lower=None, upper=None):
    """
    Perform contrast stretching on an xarray.DataArray.
    :param x: xarray.DataArray, the image data to be stretched.
    :param lower_percent: lower percentile for stretch (default is 1%).
    :param upper_percent: upper percentile for stretch (default is 99%).
    :param lower: explicit lower limit for stretch. Overrides lower_percent if provided.
    :param upper: explicit upper limit for stretch. Overrides upper_percent if provided.
    :return: xarray.DataArray with contrast-stretched values.
    """
    def stretch_band(band):
        nonlocal lower, upper
        if lower is None:
            lower = np.nanpercentile(band, lower_percent)
        if upper is None:
            upper = np.nanpercentile(band, upper_percent)
        return np.clip((band - lower) / (upper - lower), 0, 1)

    if 'variable' in x.sizes and x.sizes['variable'] > 1:
        stretched = xr.concat([stretch_band(x.isel(variable=band)) for band in range(x.sizes['variable'])], dim='variable')
    else:
        stretched = stretch_band(x)

    return stretched
