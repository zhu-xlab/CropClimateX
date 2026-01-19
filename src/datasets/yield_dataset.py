from typing import Any, Dict, Optional, Tuple
from torch.utils.data import Dataset
import xarray as xr
import geopandas as gpd
import pandas as pd
import os
import numpy as np
from src.datasets.utils import resample_yearly, bands_freq, gap_filling_yearly
import zarr
import itertools
import glob
import torch
import math
from datetime import datetime

def load_yield_labels(data_dir, years, crops, geoids):
    # filter the labels
    labels = pd.read_csv(os.path.join(data_dir, 'yield/master_yield_df.csv'), dtype={'yield': str, 'geoid': str, 'year': int, 'crop': str})
    if years and len(years) > 0:
        labels = labels[labels['year'].isin(years)]
    if crops and len(crops) > 0:
        labels = labels[labels['crop'].isin(crops)]
    if geoids and len(geoids) > 0:
        labels = labels[labels['geoid'].isin(geoids)]
    labels['yield'] = labels['yield'].astype(float)

    if len(labels.index) == 0:
        raise ValueError(f"No data found for the provided configuration, please check the config.")
    return labels

def str_to_day_of_year(date_str):
    date = datetime.strptime(date_str, "%m-%d")
    return date.timetuple().tm_yday

class YieldDataset(Dataset):
    def __init__(self, data_dir: str, x_transform=None, y_transform=None, bands:dict=None,
                crops:list=None, time_range:tuple=('01-01', '12-31'),years:list=None,geoids:list=None,
                resampling:dict={}, filter_by_crops=False, use_county_mask=True, spatial_splitting:int=0,
                cdl_30m=True, return_meta_data=False, return_label_data=False, seed=42, handle_nans='zeros',
                ignore_label_registry=False, gap_filling=False) -> None:
        self.data_dir = data_dir
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.handle_nans = handle_nans

        if isinstance(crops, str):
            crops = [crops]
        crops = [crop.lower() for crop in crops]
        self.crops = crops
        classes = {'corn': 1, 'cotton': 2,'soybeans': 5,'winter_wheat': 24,'oats': 28}
        self.crop_ids = [classes[c] for c in crops]
        self.time_range = time_range
        
        if years is None:
            years = list(range(2018, 2023))
        if isinstance(years, int):
            years = list(years)
        self.years = years
        
        if not isinstance(bands, dict):
            bands = dict(bands)
        for m in bands:
            if not isinstance(bands[m], (list, tuple)):
                bands[m] = list(bands[m])
        self.bands = bands
        
        if isinstance(geoids, str):
            geoids = list(geoids)
        self.geoids = geoids
        
        self.resampling = resampling
        self.gap_filling = gap_filling
        self.filter_by_crops = filter_by_crops
        self.use_county_mask = use_county_mask
        self.spatial_splitting = math.sqrt(spatial_splitting)
        self.cdl_30m = cdl_30m
        self.return_meta_data = return_meta_data
        self.return_label_data = return_label_data

        if not ignore_label_registry:
            self.labels = load_yield_labels(data_dir, years, crops, geoids)
        else:
            # create empty labels with all entries
            self.labels = pd.DataFrame({'geoid': geoids, 'year': [years]*len(geoids), 'yield': [0]*len(geoids)})
            self.labels = self.labels.explode('year')

    def __len__(self) -> int:
        return len(self.labels.index)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        geoid = self.labels.iloc[idx]['geoid']
        y = np.array(self.labels.iloc[idx]['yield'])
        year = self.labels.iloc[idx]['year']
                
        # prep temporal selection
        start, end = str_to_day_of_year(self.time_range[0]), str_to_day_of_year(self.time_range[1]) 
        if start < end:
            days = [list(range(start, end+1))]
            year = [year]
        else:
            # list of days for two years
            days = [list(range(start, 366)), list(range(1, end+1))]
            year = [year-1, year]

        # load cdl and county mask
        if self.use_county_mask or self.filter_by_crops:
            cdl_str = 'cdl' if self.cdl_30m else 'cdl_500m'
            fns = glob.glob(os.path.join(self.data_dir, f'{cdl_str}/{cdl_str}_{geoid}_*.zarr'))
            if len(fns) == 0:
                raise ValueError(f"No cdl data found for geoid {geoid}")
            if self.cdl_30m:
                cdl_str += '_30m'
            groups = [list(zarr.open(fn)) for fn in fns]
            masks = {}
            for j,fn in enumerate(fns): # loop through files
                for i in groups[j]: # loop through patches
                    cdl = xr.open_zarr(fn, group=i, consolidated=True)
                    cdl = cdl.sel(time=(cdl['time.year'].isin(year)))
                    cdl = cdl.drop_vars('time')

                    res_dict = self.resampling['cdl'] if 'cdl' in self.resampling else self.resampling['default'] if self.resampling else None
                    co_mask = resample_yearly(cdl['co_mask'], res_dict).values
                    crop_mask = resample_yearly(cdl[cdl_str].isin(self.crop_ids), res_dict).values
                    masks[i] = {'co_mask': co_mask, 'crop_mask': crop_mask}

        dss = []
        master = {} # one entry for each patch
        nr_bands = sum([len(self.bands[m]) for m in self.bands])
        x = None # output tensor
        band_nr = 0 # running index for bands
        for m in self.bands: # loop through sensors
            fns = [os.path.join(self.data_dir, f'{m}/{m}_{geoid}_{i}.zarr') for i in ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59']]
            # filter out non-existing files
            fns = [fn for fn in fns if os.path.exists(fn)]
            if len(fns) == 0:
                raise ValueError(f"No data found for geoid {geoid} and modality {m}")
            try:
                groups = [list(zarr.open(fn)) for fn in fns]
                nr_pts = int(groups[-1][-1]) + 1
            except:
                print('error loading: ',fns, groups)
            for j,fn in enumerate(fns): # loop through files
                for i in groups[j]: # loop through patches
                    # load the sample
                    ds = xr.open_zarr(fn, group=i)
                    # select bands
                    ds = ds[self.bands[m]]
                    # select time
                    if 'time' in ds.dims:
                        if 'cdl' in self.bands[m][0]: # take only year when it is cdl, since it is yearly
                            ds = ds.sel(time=ds['time.year'].isin(year[-1]))
                        else:
                            ds = ds.sel(time=((ds['time.year'].isin(year[-1]) & (ds.time.dt.dayofyear.isin(days[-1])) | (ds['time.year'].isin(year[0]) & (ds.time.dt.dayofyear.isin(days[0]))))))
                    # fill gaps
                    if self.gap_filling and m in bands_freq:
                        ds = gap_filling_yearly(ds, bands_freq[m], f"{year[0]}-{self.time_range[0]}", f"{year[-1]}-{self.time_range[-1]}")
                    # process to diff spatial/temporal resolution
                    res_dict = None
                    if m in self.resampling:
                        res_dict = self.resampling[m]
                    elif 'default' in self.resampling:
                        res_dict = self.resampling['default']
                    ds = resample_yearly(ds, res_dict, master[i] if i in master else None)
                    # rm data outside county
                    if self.use_county_mask:
                        ds = ds.where(masks[i]['co_mask'], np.nan)
                    # select by crop mask
                    if self.filter_by_crops:
                        ds = ds.where(masks[i]['crop_mask'], np.nan)
                    # split one patch into multiple patches
                    if self.spatial_splitting > 0:
                        len_split = int(ds.sizes['x']/self.spatial_splitting)
                        steps = np.arange(0, ds.sizes['x'], len_split)
                        if x is None:
                            x = np.empty([int(nr_pts*self.spatial_splitting**2), nr_bands, len(ds.time), len_split, len_split])
                        for xs, ys in itertools.product(steps, steps):
                            x[int(i),band_nr:band_nr+len(self.bands[m]),...] = ds.isel(x=slice(xs, xs+len_split), y=slice(ys, ys+len_split)).to_array().values
                    else:
                        if x is None: # shape : [p, c, t, h, w]
                            x = np.empty([nr_pts, nr_bands, len(ds.time), ds.sizes['x'], ds.sizes['y']])
                        x[int(i),band_nr:band_nr+len(self.bands[m]),...] = ds.to_array()
                    if i not in master:
                        master[i] = ds
                    ds.close()
            band_nr += len(self.bands[m])

        if x.ndim == 5:
            x = np.swapaxes(x, 1, 2) # reorder to [p, t, c, h, w]

        # handle nans
        if self.handle_nans == 'zeros':
            x = np.where(np.isnan(x), 0, x)
            y = np.where(np.isnan(y), 0, y)

        # create a tensor
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if self.x_transform:
            x = self.x_transform(x)
        if self.y_transform:
            y = self.y_transform(y)
        out = {'x': x, 'y': y}
        if self.return_meta_data:
            out['meta'] = {'time':[master[m].time.values.astype('int64') for m in master], 'geoid':geoid}
            if self.spatial_splitting > 0:
                nr_splits = int(self.spatial_splitting)**2
                out['meta']['split_part'] = list(range(nr_splits))*len(master.keys())
                out['meta']['pid'] = [p for p in list(master.keys()) for _ in range(nr_splits)]
            else:
                out['meta']['pid'] = list(master.keys())
                out['meta']['split_part'] = [-1]*len(master.keys())
        if self.return_label_data:
            out['label'] = {'geoid': geoid, 'split_part': -1, 'time': f"{self.labels.iloc[idx]['year']}"}

        return out
