# %%
import multiprocessing as mp
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
from joblib import Parallel, delayed
from pyproj import CRS
from tqdm import tqdm

from utils import crop_to_region, load_subregion_and_reproject


def climate_standardization(ds, nr_days=15, return_z=True, return_mean_std=False, n_jobs=10):
    """Compute z = (x - mean) / std for each day in the dataset through the surrounding days before and after nr_days."""
    manager = mp.Manager()
    ds_z = manager.list()
    if return_mean_std:
        ds_mean = manager.list()
        ds_std = manager.list()

    def climate_stand_day(i):
        start = 365 if i == nr_days else (i-nr_days)%365
        end = (i+nr_days)%365

        if start < end:
            days = list(range(start, end+1))
        else:
            days = list(range(start, 366))
            period2 = list(range(1, end+1))
            days.extend(period2)
        # compute z from day i
        ds_f = ds.sel(time=ds.time.dt.dayofyear.isin(days)) # select surrounding days
        mean = ds_f.mean(skipna=True, dim='time')
        std = ds_f.std(skipna=True, ddof=1, dim='time')
        if return_z:
            z = (ds_f - mean) / std
            # only add day i
            ds_z.append(z.sel(time=z.time.dt.dayofyear.isin(i))) # select only day i
        if return_mean_std:
            ds_mean.append(mean)
            ds_std.append(std)

    Parallel(n_jobs=n_jobs)(delayed(climate_stand_day)(d) for d in range(1,366))

    out = ()
    if return_z:
        out += (xr.concat(ds_z, 'time').sortby('time'),)
    if return_mean_std:
        out += (xr.concat(ds_mean, 'time').sortby('time'), xr.concat(ds_std, 'time').sortby('time'))
    return out

def replace_non_successive_ones(arr_in):
    # assumes [time, ] and replaces all not successive 1s with count < 3
    if arr_in.shape[0] < 3:
        raise ValueError('at least three time steps should be provided')
    if len(arr_in.shape) == 1:
        arr_pad = np.pad(arr_in, [(2,2)], constant_values=0)
    elif len(arr_in.shape) == 2:
        arr_pad = np.pad(arr_in, [(2,2), (0,0)], constant_values=0)
    elif len(arr_in.shape) == 3:
        arr_pad = np.pad(arr_in, [(2,2), (0,0), (0,0)], constant_values=0)
    else:
        raise ValueError('higher input dimensions than 3 not supported')

    # compute 3s for successive by adding the array to itself at different positions
    arr_added = ((arr_pad[1:] + arr_pad[:-1])[1:] + arr_pad[:-2])[:-2]
    # find the 3s
    idxs = np.argwhere(arr_added == 3)
    # replace the two indexes before with 3s (because they are 1 and 2)
    for i in range(1,3):
        idx = np.hstack((np.expand_dims(idxs[...,0] - i, axis=1), idxs[...,1:]))
        if len(arr_in.shape) == 1:
            c = (idx[:,0])
        elif len(arr_in.shape) == 2:
            c = (idx[:,0],idx[:,1])
        elif len(arr_in.shape) == 3:
            c = (idx[:,0],idx[:,1],idx[:,2])
        # overwrite to 3
        arr_added[c] = 3
    # replace everything which is not at 3 -> all 3s are successive 1s
    arr_in = xr.where(arr_added < 3, 0, 1)
    return arr_in

def compute_hcw_index(ds, keys, quantiles, start_date=None, end_date=None, at_least_three_days=False):
    """Compute the HCW index for the dataset ds with the keys for the min and max temperature and
    the quantiles.

    If start_date and end_date are provided, only the time between these dates is considered. If
    at_least_three_days is True, only events which are longer than 3 days are considered.
    """
    ds_min = ds[keys[0]]
    ds_max = ds[keys[1]]
    if isinstance(quantiles, list):
        quantiles = np.array(quantiles)
    #sort the quantiles list for > .5 reversed and for < .5 forward
    #(needs to be in correct ordr for overwriting correctly (xr.where) later)
    quantiles = np.concatenate([sorted(quantiles[quantiles > .5], reverse=True),
                                sorted(quantiles[quantiles < .5], reverse=False)])

    z_min = climate_standardization(ds_min)
    z_max = climate_standardization(ds_max)
    # select the time of interest
    quant_ar = xr.where(ds_min.isnull(), ds_min, 0) # create an array with zeros but keep nans
    quant_ar.name = 'hcw'
    if start_date and end_date:
        quant_ar = quant_ar.sel(time=slice(start_date, end_date))
        z_min = z_min.sel(time=slice(start_date, end_date))
        z_max = z_max.sel(time=slice(start_date, end_date))
    for q in quantiles:
        z = stats.norm.ppf(q)
        # only when tmin + tmax are in exceptional range (add the numbers instead of '&' to prevent downloading)
        if z < 0:
            quan = xr.where((z_min < z) & (z_max < z), 1, 0)
        else:
            quan = xr.where(((z_min > z) & (z_max > z)), 1, 0)
        # only events which are more than 3 days
        if at_least_three_days:
            quan = replace_non_successive_ones(quan)
        # overwrite only values which are zero
        quan = quan * q
        quant_ar = xr.where((quant_ar == 0), quan, quant_ar)
    return quant_ar

def read_dataset_file_daymet(year, region, var, fn_pat, fo, daymet_crs, goal_crs, chunks='auto'):
    """Read the corresponding dataset files for the year and the variable and crop it to the
    region."""
    ds = xr.open_dataset(fo + fn_pat.format(var, year), engine='h5netcdf', chunks=chunks)
    del ds['lat']
    del ds['lon']
    ds = ds[var]

    k = list(ds.attrs.keys())
    for key in k:
        del ds.attrs[key]

    if 'lambert_conformal_conic' in ds.coords:
        ds = ds.drop_vars('lambert_conformal_conic')

    ds = ds.rio.set_spatial_dims("x", "y")
    ds = ds.rio.write_crs(daymet_crs)

    ds = load_subregion_and_reproject(ds, region, daymet_crs, goal_crs, 1000, all_touched=False)

    return ds

def load_daymet_patch(region, fo, bands=['prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp'], time_slice=('1980-01-01', '2022-12-31'), crs=None, nr_jobs=10):
    """Load the daymet patch for the region and the bands and the time slice.

    If crs is None, the original crs is used.
    """
    crs_daymet = CRS.from_proj4('+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs')
    fn_pat = 'daymet_v4_daily_na_{}_{}.nc'
    start_year, end_year = int(time_slice[0].split('-')[0]), int(time_slice[1].split('-')[0])

    # load the data
    patches = []
    for band in bands:
        patch_years = Parallel(nr_jobs, backend='threading')(delayed(read_dataset_file_daymet)(year, region, band, fn_pat, fo, crs_daymet, crs) for year in range(start_year, end_year+1))
        patch_year = xr.concat(patch_years, dim='time')
        patches.append(patch_year)
    patch = xr.merge(patches)
    patch = patch.compute()

    return patch

def create_hcw_frequencies(fo_sv, data_folder, overwrite=False):
    from pc_utils import load_county_shp_USA
    os.makedirs(fo_sv, exist_ok=True)
    dates = ['2008-01-01', '2022-12-31']
    df_us_county = load_county_shp_USA()
    quantiles = [.01, .02, .05, .1, .9, .95, .98, .99]

    def process_data(i):
        geoid = df_us_county.loc[[i]]['GEOID'].values[0]
        fn = f'hcw_freq_{geoid}_{dates[0][:4]}_{dates[-1][:4]}.nc'
        if not os.path.exists(fo_sv + fn) or overwrite:
            daymet = load_daymet_patch(df_us_county.loc[[i]], data_folder, bands=['tmax', 'tmin'], time_slice=dates)
            hcw = compute_hcw_index(daymet, ['tmin', 'tmax'], quantiles, start_date=dates[0], end_date=dates[1], at_least_three_days=True)
            # optional save hcw hcw.to_netcdf(fo_sv + f'hcw_{geoid}_{dates[0][:4]}_{dates[-1][:4]}.nc')
            res = None
            for k in hcw.time:
                h = hcw.sel(time=k)
                binary_bands_small = [xr.where((h <= j) & (h > 0), 1, 0).astype('uint32') for j in quantiles[:len(quantiles)//2]]
                binary_bands_big = [xr.where(h >= j, 1, 0).astype('uint32') for j in quantiles[len(quantiles)//2:]]
                binary_bands_small.extend(binary_bands_big)
                if res is None:
                    res = xr.concat(binary_bands_small, dim='band')
                else:
                    res += xr.concat(binary_bands_small, dim='band')
            res.coords['band'] = np.array(quantiles)
            res = res.to_dataset(name='freq')
            res.to_netcdf(fo_sv + fn)

    Parallel(n_jobs=10)(delayed(process_data)(i) for i in tqdm(df_us_county.index))

# %%
if __name__ == '__main__':
    import rootutils
    home_folder = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    create_hcw_frequencies(os.path.join(home_folder, 'data/daymet/hcw/freq/'), os.path.join(home_folder, 'data/daymet/'))
