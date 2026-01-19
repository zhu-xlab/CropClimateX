import numpy as np
import pandas as pd
import xarray as xr
from functools import reduce
from operator import or_
import math
import pyproj
import matplotlib.pyplot as plt
from matplotlib import rcParamsDefault

def increase_plt_text_size(scale):

    # Reset all rcParams to Matplotlib defaults
    plt.rcParams.update(rcParamsDefault)
    # Named font sizes mapped to approximate numeric points
    size_map = {
        'xx-small': 6, 'x-small': 8, 'small': 10, 'medium': 12,
        'large': 14, 'x-large': 16, 'xx-large': 18
    }

    # Get current font sizes
    params = [
        'font.size', 'axes.titlesize', 'axes.labelsize',
        'xtick.labelsize', 'ytick.labelsize',
        'legend.fontsize', 'figure.titlesize'
    ]

    for p in params:
        v = plt.rcParams[p]
        # Convert named sizes to numeric, scale, then back to numeric value
        if isinstance(v, str):
            v = size_map.get(v, 12)  # fallback if unknown
        plt.rcParams[p] = v * scale

def rm_attrs(ds):
    """Remove all attributes from a dataset and its variables."""
    ds.attrs.clear()

    if isinstance(ds, xr.Dataset):
        for var in ds.data_vars:
            ds[var].attrs.clear()
            for attr in list(ds[var].attrs.keys()):
                del ds[var].attrs[attr]
    return ds

def rm_coords(ds, to_keep):
    """Remove all coords from a dataset except the ones in to_keep."""
    for coord in ds.coords:
        if coord not in to_keep:
            ds = ds.drop(coord)
    return ds

def purge_value(ds, name):
    """Remove the name value from coords, attrs and data_vars."""
    if name in ds.coords:
        ds = ds.drop_vars(name)
    if name in ds.attrs:
        del ds.attrs[name]
    for var in ds.data_vars:
        if name in ds[var].attrs:
            del ds[var].attrs[name]
    return ds

def check_bits_set(x, start_n, end_n, all_set=True):
    if math.isnan(x): # take nan as bit not set
        return False
    if isinstance(x, float):
        x = int(x)
    mask = (1 << (end_n - start_n + 1)) - 1
    masked_bits = (x >> start_n) & mask
    if all_set: # all bits are set
        return masked_bits == mask
    else: # one bit is set
        return masked_bits != 0 and (masked_bits & (masked_bits - 1)) == 0

def histogram_matching(da, ref_da, bands=[]):
    """Match the histogram of the dataset to the reference dataset and ignore nan's.
    Since we have not only RGB bands, process each band separatly.
    only the bands in the list are processed, if empty all bands are processed.
    see https://github.com/mapbox/rio-hist/blob/master/docs/notebooks/Histogram_Matching_Numpy.ipynb"""
    if len(bands) == 0:
        bands = [b for b in da.data_vars if b in ref_da.data_vars]
    
    # loop through channels
    for band in bands:
        if band not in ref_da.data_vars or band not in da.data_vars:
            warnings.warn(f"Band {band} not in datasets. Skipping.")
            continue
        # get flattened arrays
        arr = da[band].values.ravel()
        ref_arr = ref_da[band].values.ravel()
        nan_mask = np.isnan(arr)

        # get counts
        s_values, bin_idx, s_counts = np.unique(arr[~np.isnan(arr)], return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(ref_arr[~np.isnan(ref_arr)], return_counts=True)

        # get the normalized cumulative sum of the counts
        s_quantiles = np.nancumsum(s_counts).astype(np.float64) / np.count_nonzero(~np.isnan(arr))
        t_quantiles = np.nancumsum(t_counts).astype(np.float64) / np.count_nonzero(~np.isnan(ref_arr))

        # interpolate linearly to the matching quantiles
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        result_arr = np.full(arr.shape, np.nan)  # create dummy array
        # Fill in the interpolated values for non-NaN positions
        result_arr[~nan_mask] = interp_t_values[bin_idx]
        # replace the values in dataset
        da[band] = (da[band].dims, result_arr.reshape(da[band].shape))

    return da

def select_image_by_cloud_cover(ds, band, cloud_cover_thresh=1., invalid_values=[]):
    """Select the first image with cloud cover below threshold and lowest cloud cover in set.
    invalid_values are treated like cloud cover.
    Band is assumed to be binary mask with 1 for cloud and 0 for no cloud."""
    # remove images with cloud cover / invalid_values above threshold
    bands = [b for b in ds.data_vars if b != band]
    invalid_mask = reduce(or_, [(ds[b].isin(invalid_values) | ds[b].isnull()) for b in bands]) # alternative for iterating over the arrays

    ds[band] = (ds[band] | invalid_mask)
    cloud_cover = ds[band].sum(dim=[dim for dim in ds.sizes if dim != 'time']) / np.prod([ds.sizes[dim] for dim in ds.sizes if dim != 'time'])
    ds.coords['cloud_cover'] = cloud_cover
    ds = ds.sel(time=(cloud_cover <= cloud_cover_thresh))
    if len(ds.time) <= 0:
        raise ValueError('No images with cloud cover below threshold found.')

    # rank each time step by cloud cover of the image
    ds = ds.sortby('cloud_cover')
    # select the best image
    da = ds.isel(time=0)
    # rm time coordinate
    if 'time' in da.coords:
        del da['time']
    if 'cloud_cover' in da.coords:
        del da['cloud_cover']

    return da

def select_pixels_by_first_valid_value(ds, invalid_values=[], sort_by_amount=False, proxy_band='first'):
    """Select the first valid value in time dimension for each pixel in the dataset."""
    # sort dataset by number of valid values in time dimension
    if sort_by_amount:
        if proxy_band == "first":
            proxy_band = [b for b in ds.data_vars if b not in ["spatial_ref"]][0]
        valid_counts = (ds[proxy_band].isin(invalid_values) | ds[proxy_band].isnull()).sum(dim=('x', 'y'))
        ds = ds.sortby(valid_counts, ascending=True)
    da = ds.isel(time=0).copy()
    # save the coords to restore them later
    original_coords = {coord: da.coords[coord] for coord in da.coords}
    for i in range(1,len(ds.time)):
        da = xr.where((da[proxy_band].isin(invalid_values) | da[proxy_band].isnull()), ds.isel(time=i), da)

    # restore coordinates
    da = da.assign_coords({coord: original_coords[coord] for coord in original_coords})
    
    return da

def select_pixels_by_mask(ds, band, thres=1.):
    cloud_cover = ds[band].mean(dim=[dim for dim in ds.dims if dim != 'time'])
    ds.coords['cloud_cover'] = cloud_cover
    ds = ds.sel(time=(cloud_cover <= thres))
    if len(ds.time) <= 0:
        raise ValueError('No images with mask below threshold found.')
    
    # rank each time step by cloud cover of the image
    ds = ds.sortby('cloud_cover')
    # select the first cloud free pixels
    da = ds.isel(time=0).copy()
    for i in range(1,len(ds.time)):
        da = xr.where(da[band], ds.isel(time=i), da)

    # rm time coordinate if it got added by the resample method
    if 'time' in da.coords:
        del da['time']
    if 'cloud_cover' in da.coords:
        del da['cloud_cover']

    return da

def select_pixels_by_cloud_cover(ds, band, proxy_band='first', invalid_values=[], thres=1., adjust_for_conditions=[]):
    """Rank images by cloud cover, remove images with cloud cover above threshold.
    And select the first cloud free pixels sorted by the overall cloud cover.
    Band is assumed to be binary mask with 1 for cloud and 0 for no cloud.
    proxy_band is used to detemine the invalid values.
    invalid_values are treated like np.nan"""
    if proxy_band == "first":
        proxy_band = [b for b in ds.data_vars if b not in [band, "spatial_ref"]][0]
    cloud_cover = ds[band].mean(dim=[dim for dim in ds.dims if dim != 'time'])
    invalid_cover = xr.where(ds[proxy_band].isin(invalid_values) | ds[proxy_band].isnull(), 1, 0).mean(dim=[dim for dim in ds.dims if dim != 'time'])

    # in case it is not bool yet
    if not ds[band].dtype == bool:
        ds[band] = ds[band].astype(bool)
    # remove images with cloud cover above threshold
    total_cover = xr.where(ds[band] | ds[proxy_band].isin(invalid_values) | ds[proxy_band].isnull(), 1, 0).mean(dim=[dim for dim in ds.dims if dim != 'time'])
    ds = ds.assign_coords(cloud_cover = cloud_cover, invalid_cover=invalid_cover, total_cover=total_cover)
    ds = ds.sel(time=(total_cover <= thres))
    if len(ds.time) <= 0:
        raise ValueError('No images with cloud cover below threshold found.')

    # rank each time step by cloud cover of the image (also use id to ensure same order for diff runs)
    sort_vals = ['invalid_cover','cloud_cover']
    if 'id' in ds.coords:
        sort_vals.append('id')
    ds = ds.sortby(sort_vals)
    # select the first cloud free pixels
    da = ds.isel(time=0).copy().compute()
    if len(adjust_for_conditions) > 0:
        # match all histograms to the first image
        for i in range(1,len(ds.time)):
            # create arrays without invalid values
            # histogram_matching does only work with nan's
            ds_arr = ds.isel(time=i).copy().where(
                ~ds[proxy_band].isel(time=i).isnull() \
                & ~ds[proxy_band].isel(time=i).isin(invalid_values)
            )
            da_arr = da.where(
                ~da[proxy_band].isnull() \
                & ~da[proxy_band].isin(invalid_values)
            )
            ds[{'time': i}] = histogram_matching(ds_arr, da_arr, adjust_for_conditions)
    # save coordinate (sometimes it is lost in where)
    original_coords = {coord: da.coords[coord] for coord in da.coords}
    for i in range(1,len(ds.time)): # start with 1 because 0 is already selected
        da = xr.where(da[band] & ~ds[band].isel(time=i), ds.isel(time=i), da)

    # replace invalid values with cloud pixels
    for i in range(len(ds.time)): # start with 0 in case pixels were overwritten by the first loop
        da = xr.where((da[proxy_band].isin(invalid_values) | da[proxy_band].isnull()) &\
             ~(ds[proxy_band].isel(time=i).isin(invalid_values) | ds[proxy_band].isel(time=i).isnull()),\
                 ds.isel(time=i), da)

    # restore coordinates (sometimes they are mixed up in where)
    da = da.assign_coords({coord: original_coords[coord] for coord in original_coords})
    # rm time coordinate gets added by the resample method
    for coord in ['time', 'cloud_cover', 'invalid_cover', 'total_cover']:
        if coord in da.coords:
            del da[coord]

    return da

def mosaic_per_year(ds, interval='16D', func=select_pixels_by_first_valid_value, include_first_from_next_year=False, **kwargs):
    """Create a mosaic of the dataset for the given interval, starting in each year with the first
    date of the year."""
    data_ys = []
    start_y = ds.time.dt.year.min().values
    end_y = ds.time.dt.year.max().values
    ds = ds.sortby('time') # make sure the dataset is sorted by time
    for y in range(start_y, end_y+1): # start in each year with 01-01 to have clear format
        start_date = pd.to_datetime(f'{y}-01-01')
        if include_first_from_next_year:
            end_date = pd.to_datetime(f'{y+1}-01-01') + pd.Timedelta(interval) # set end date in new year
        else:
            end_date = pd.to_datetime(f'{y}-12-31')
        ds_y = ds.sel(time=slice(f'{y}-01-01', end_date.strftime('%Y-%m-%d')))
        if len(ds_y.time) > 0:
            ds_y_r = ds_y.resample(time=interval, skipna=True, label='left', origin=start_date)\
                .apply(func, **kwargs)
            if include_first_from_next_year:
                # select only values within the year, otherwise more values in the next year are included
                ds_y_r = ds_y_r.sel(time=slice(start_date, pd.to_datetime(f'{y}-12-31')))
            data_ys.append(ds_y_r)
    new_ds = xr.concat(data_ys, dim='time')
    new_ds = new_ds.sortby('time')
    return new_ds

def mosaic(ds, interval='16D', func=select_pixels_by_first_valid_value, **kwargs):
    """Create a mosaic of the dataset for the given interval, assumes ds has a time dimension and other dims are equally shaped."""
    ds = ds.sortby('time') # make sure the dataset is sorted by time
    time_data = []
    start_date = ds.time.min().values.astype(f'datetime64[{interval[-1]}]') # take the smallest date (without time)
    while True:
        end_date = start_date + pd.Timedelta(interval) - pd.Timedelta(1, 'ns')
        ds_sel = ds.sel(time=slice(start_date, end_date))
        if len(ds_sel.time) > 0: # when there is data process it
            ds_r = ds_sel.resample(time=interval, skipna=True, label='left', origin=start_date)\
                .apply(func, **kwargs)
            time_data.append(ds_r)
        if end_date > ds.time.max().values:
            break
        start_date = end_date + pd.Timedelta(1, 'ns')
    new_ds = xr.concat(time_data, dim='time', coords='all')
    new_ds = new_ds.sortby('time')
    return new_ds

def crop_to_region(data, region, use_bounding_box, margin=0, **kwargs):
    """Crop the data to the region.

    If use_bounding_box is True, the region will be used as a bounding box, otherwise the exact
    shape will be used. margin can be a number or tuple of 2 numbers to add a margin for the
    clipping to the bounding box. If margin is True, the margin will be twice the size of the
    region.
    """
    if use_bounding_box:
        if margin:
            if isinstance(margin, (int,float)):
                margin = (margin, margin)
            elif isinstance(margin, bool): # if margin is True, use twice the size of the region
                margin = (region.bounds.maxx-region.bounds.minx, region.bounds.maxy-region.bounds.miny)
            elif not isinstance(margin, (list, tuple)) or len(margin) != 2:
                raise ValueError('margin should be a float, int or tuple of length 2')
            return data.rio.clip_box(region.bounds.minx-margin[0], region.bounds.miny-margin[1], region.bounds.maxx+margin[0], region.bounds.maxy+margin[1])
        return data.rio.clip_box(region.bounds.minx, region.bounds.miny, region.bounds.maxx, region.bounds.maxy)
    else:
        return data.rio.clip(region.geometry, **kwargs)

def load_subregion_and_reproject(ds, region, org_crs, goal_crs, org_resolution, **kwargs):
    """Load a subregion by first clipping the data to the region in the original crs and then
    reprojecting it to the goal crs and clipping again for a more computation friendly .nc loading
    region is assumed to be a geopandas dataframe with a single geometry."""
    if region is not None:
        # first crop in org crs with margin, then again with region in goal crs
        region_reproj = region.to_crs(org_crs)
        ds = crop_to_region(ds, region_reproj, True, margin=True)

    if goal_crs is not None:
        ds = ds.rio.reproject(goal_crs, resolution=org_resolution)
        ds.rio.write_crs(goal_crs, inplace=True)

    if region is not None and goal_crs is not None:
        ds = crop_to_region(ds, region, False, margin=False, **kwargs)

    return ds

def split_date_range(start_end_dates, split):
    if split == 'monthly':
        start_date = pd.to_datetime(start_end_dates[0][:-3])
        end_date = pd.to_datetime(start_end_dates[1][:-3])
        if end_date - start_date > pd.Timedelta(days=31):
            monthly_dates_start = list(pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m-%d'))
            monthly_dates_end = list(pd.date_range(start=start_date, end=end_date, freq='ME', inclusive='right').strftime('%Y-%m-%d'))
            monthly_dates_end.append(start_end_dates[1])
            monthly_dates_start[0] = start_end_dates[0]
            out = list(zip(monthly_dates_start, monthly_dates_end))
        else:
            out = [start_end_dates]
    elif split == 'yearly':
        start_date = pd.to_datetime(start_end_dates[0][:-6])
        end_date = pd.to_datetime(start_end_dates[1][:-6])
        if end_date - start_date > pd.Timedelta(days=365):
            yearly_dates_start = list(pd.date_range(start=start_date, end=end_date, freq='YS').strftime('%Y-%m-%d'))
            yearly_dates_end = list(pd.date_range(start=start_date, end=end_date, freq='YE', inclusive='right').strftime('%Y-%m-%d'))
            yearly_dates_end.append(start_end_dates[1])
            yearly_dates_start[0] = start_end_dates[0]
            out = list(zip(yearly_dates_start, yearly_dates_end))
        else:
            out = [start_end_dates]
    elif split == 'weekly':
        start_date = pd.to_datetime(start_end_dates[0])
        end_date = pd.to_datetime(start_end_dates[1])
        if end_date - start_date > pd.Timedelta(days=7):
            weekly_dates_start = list(pd.date_range(start=start_date, end=end_date, freq='W-MON', inclusive='both').strftime('%Y-%m-%d'))
            weekly_dates_end = list((pd.date_range(start=start_date, end=end_date, freq='W-SUN', inclusive='both')).strftime('%Y-%m-%d'))
            weekly_dates_end.append(start_end_dates[1])
            if not weekly_dates_start[0] == start_end_dates[0]:
                weekly_dates_start = [start_end_dates[0]] + weekly_dates_start
            out = list(zip(weekly_dates_start, weekly_dates_end))
        else:
            out = [start_end_dates]
    else:
        raise ValueError(f"split {split} not supported")
    return out

def split_date_range_by_freq(start_date, end_date, freq):
    # Generate the full range of dates from start to end date
    all_dates = pd.date_range(start=start_date, end=end_date).tolist()

    # Split the dates by the specified frequency
    split_ranges = []
    for i in range(0, len(all_dates), freq):
        range_start = all_dates[i].strftime('%Y-%m-%d')
        range_end = all_dates[min(i + freq - 1, len(all_dates) - 1)].strftime('%Y-%m-%d')
        split_ranges.append([range_start, range_end])
    
    return split_ranges

def shp_to_utm_crs(shp):
    """convert the shape from WGS84 to UTM crs."""
    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(*shp.geometry.bounds.values[0]),
    )

    # Save the CRS
    epsg = utm_crs_list[0].code
    utm_crs = pyproj.CRS.from_epsg(epsg)
    shp = shp.to_crs(utm_crs)
    return shp

def number_to_range(input_num, divider=10):
    """transform the input number to a range of the divider."""
    if isinstance(input_num, str):
        input_num = int(input_num)
    start = (input_num // divider) * divider
    end = start + divider - 1
    return f"{start}-{end}"
