# %%
import glob
import os
import warnings
import logging
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from numcodecs import Blosc
import rootutils
import utils_cdl_nlcd as u_cdl
import utils_daymet as u_dymt
import utils_sentinel2 as u_sen2
import utils_soil as u_soil
import utils_usdm as u_usdm
import utils as u
from utils import split_date_range
import re
import shutil
import time
import subprocess
import random
import socket
import rasterio
from rasterio.enums import Resampling

from joblib import Parallel, delayed
from tqdm import tqdm

from utils import check_bits_set, mosaic, select_pixels_by_cloud_cover, number_to_range, select_pixels_by_first_valid_value
# %%
nr_groups = 10 # how many patches will be grouped into one zarr file
dates = ('2018-01-01', '2022-12-31')
home_folder = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sv_folder = os.path.join(home_folder, 'data/shared/raw/')
num_workers = os.cpu_count()
copy_to_local=True if not 'storage' in socket.gethostname() else False
# %%
def check_exists(ln, geoid, name, dates=None, overwrite=False):
    sv_fns = [os.path.join(home_folder, sv_path, name, f'{name}_{geoid}_{i}-{i+nr_groups-1}.zarr') for i in range(0, ln, nr_groups)]
    if overwrite or any(map(lambda path: not os.path.exists(path), sv_fns)):
        for sv_fn in sv_fns:
            if os.path.exists(sv_fn):
                shutil.rmtree(sv_fn)
                time.sleep(10) # give it some time to delete
        return False
    return True

def sv_patches(gdf, name, data=None, transform=None, dates=None, encoding=None):
    """if data is none the transform needs to load the data, if data is provided the transform is used to transform the data."""
    assert data or transform, 'Either data or transform function for loading should be provided.'
    # clip to shps + ensure the right crs
    def write_to_zarr(i, name, data=None, transform=None, dates=None, encoding=None):
        import rioxarray as rxr # import again since it is not initialized in new processes
        geoid, idx = gdf.loc[i]['GEOID_PID'].split('_')
        idx = int(idx)
        idx_str = number_to_range(idx, nr_groups)
        file_name = f'{name}_{geoid}_{idx_str}.zarr'
        sv_fn = os.path.join(home_folder, sv_path, name, file_name)
        if os.path.exists(os.path.join(sv_fn, f'{idx}')):
            # don't write if already exists
            return
        os.makedirs(os.path.dirname(sv_fn), exist_ok=True)
        
        try:
            if transform and data:
                data = transform(data)
            elif data is None:
                data = transform(gdf.loc[[idx]])
            
            # rm source attr
            if 'source' in data.attrs:
                del data.attrs['source']
            # remove and add crs info since it makes problems otherwise
            for var in data.data_vars:
                if 'grid_mapping' in data[var].attrs:
                    del data[var].attrs['grid_mapping']
            data.rio.write_crs(gdf.crs, inplace=True)
            data.attrs['crs'] = gdf.crs.to_string()
            
            # when there are pixels missing (because outside of file) fill them
            # data = data.rio.pad_box(*gdf.loc[[idx]].total_bounds, constant_values=np.nan)
            # clip in case to remove additional pixel from command before + if county data
            data_new = data.rio.clip(gdf.loc[[idx]].geometry)
            if data_new.sizes['x'] != data_new.sizes['y'] or data_new.sizes['y'] % 2:
                # the size is wrong (only happened for 37083_2)
                data_new = data.rio.clip(gdf.loc[[idx]].geometry, all_touched=True)
                if data_new.sizes['x'] != data_new.sizes['y'] or data_new.sizes['y'] % 2:
                    warnings.warn(f"Size of the patch {geoid, idx} does not match, starting to adjust automatically: {data_new.sizes}")
                # rm single pixels from each side
                while data_new.sizes['x'] % 2:
                    data_new = data_new.isel(x=slice(1, None))
                while data_new.sizes['y'] % 2:
                    data_new = data_new.isel(y=slice(1, None))
            data = data_new
            
            # define compression
            compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
            enc = {var: {"compressor": compressor} for var in data.data_vars}
            if 'time' in data.sizes:
                data = data.chunk({'time': 4, 'x': 200, 'y': 200})
            else:
                data = data.chunk({'x': 200, 'y': 200})
            
            if encoding:
                for var in data.data_vars:
                    if var in encoding:
                        encoding[var].update(enc[var])
                    else:
                        encoding[var] = enc[var]
            else:
                encoding = enc

            data.to_zarr(sv_fn, mode='a', group=f'{idx}', encoding=encoding, write_empty_chunks=False, consolidated=True)
        except Exception as e:
            print(f'{geoid}_{idx} failed:',e)
            if os.path.exists(f'{sv_fn}/{idx}'):
                shutil.rmtree(f'{sv_fn}/{idx}')
    
    [write_to_zarr(i, name, data, transform, dates, encoding) for i in gdf.index]

def load_netcdf_distributed_files(path, region=None, dim='time', num_workers=os.cpu_count(), copy_to_local=copy_to_local):
    """alternative to xr.open_mfdataset to load distributed files, since open_mfdataset does take a long time."""
    def load_clip(path, region):
        try:
            ds = xr.open_dataset(os.path.join(dir, path), engine='h5netcdf', chunks='auto')
        except Exception as e:
            print(f'Error while loading (this file will not be used for creating, please fix the file and try again) {path}: {e}')
            # raise
            return
        if region is not None:
            ds = ds.rio.write_crs(region.crs)
            ds = ds.rio.clip(region.geometry, all_touched=True)
        ds = ds.load() # necessary when copying and deleting to server
        ds.close()
        return ds
    
    dir = os.path.dirname(path)
    pattern = re.compile(os.path.basename(path))
    fns = os.listdir(dir)
    fns = [file for file in fns if pattern.match(file)]

    if copy_to_local:
        # copy files to local
        rnd = random.randint(0, 99999999)
        while True:
            mnt_dir = os.path.join(home_folder.parents[1], f"mnt_{rnd}")
            if not os.path.exists(mnt_dir):
                os.makedirs(mnt_dir)
                break
            rnd += 1
        ip = os.environ['IP']
        username = os.environ['USERNAME']
        keyfile = os.environ['KEYFILE']
        files = [f"{username}@{ip}:/mnt/datastorage/home/{username}/uscc/data/shared/raw/{dir.split('/')[-2]}/{dir.split('/')[-1]}/{f}" for f in fns]
        subprocess.run(["rsync", "-u", "-e", f"ssh -i {keyfile}", *files, f"{mnt_dir}/"])
        # write new file names
        fns = glob.glob(f"{mnt_dir}/*.nc")

    ds_l = Parallel(n_jobs=num_workers, backend='threading')(delayed(load_clip)(fn, region=region) for fn in fns)
    
    if len(ds_l) < 1:
        raise ValueError(f"No files found for {path}")
    if len(ds_l) > 1:
        # check if x and y are the same
        if not indices_are_identical(ds_l, ['x', 'y']):
            warnings.warn("coordinates are not the same, automatically adjusting them.")
            ds_l = [ds.rio.write_crs(ds.rio.crs) for ds in ds_l]
            ds_l = [ds_l[0]] + [ds.rio.reproject_match(ds_l[0]) for ds in ds_l[1:]]
        ds_l = [d for d in ds_l if d is not None]
        if not all(['spatial_ref' in ds.coords for ds in ds_l]):
            ds_l = [d.rio.write_crs(d.rio.crs) for d in ds_l]
        ds = xr.concat(ds_l, dim=dim, coords="minimal")
    else:
        ds = ds_l[0]
    ds = ds.sortby(dim)

    if copy_to_local:
        # clean up - rm files
        shutil.rmtree(f"{mnt_dir}")

    return ds

def indices_are_identical(datasets: list, indices: list) -> bool:
    """Check if the indices from the xarray datasets are equal."""
    if len(datasets) == 1:
        return True
    
    # get all indices
    if indices:
        coords = indices
    else:
        coords = list(set(key for ds in datasets for key in list(ds.indexes.keys())))
    
    # check if all indices exist in all datasets
    if not all(all(coord in ds.indexes for ds in datasets) for coord in coords):
        return False

    # check if all indices are equal
    if not all(all(datasets[0].indexes[coord].equals(ds.indexes[coord]) for ds in datasets) for coord in coords):
        return False

    return True

def prepare_patches(gdf, geoid, shp_co, data_path, sv_path, dates, version_name,
                    modality=['modis', 'modis_lai', 'modis_lst', 'daymet', 'usdm', 'dem', 'soil', 'cdl', 'nlcd', 'sen2', 'sen1', 'landsat'], overwrite=False):
    nr_files = len(gdf.index)
    
    year_start = int(dates[0].split('-')[0])
    year_end = int(dates[1].split('-')[0])
    # product: org scale + offset -> type; new transform -> type
    # modis: 0.0001 -> int16; kept
    name = 'modis'
    load_name = version_name + '_' + name
    if name in modality:
        # red, nir08, blue, green, swir16
        bands = [f'sur_refl_b0{i}' for i in range(1,5)] + ['sur_refl_b06']
        encoding={d: {'dtype': 'int16', 'scale_factor': 0.0001, '_FillValue': np.iinfo(np.int16).min} for d in bands}
        def transform(row):
            years = range(year_start, year_end+1)
            year_file_str = '|'.join([str(y) for y in years])
            ds = load_netcdf_distributed_files(os.path.join(data_path, rf'{load_name}/{geoid}/{load_name}_{row["GEOID_PID"].iloc[0]}_({year_file_str}|{year_end+1}-01-).*.nc$'))
            for b in bands:
                # reverse the scaling
                ds[b] = ds[b] * encoding[b]['scale_factor']
                # remove uncorrect values (there are fill values (-2147483648) in the data, where it was not processed)
                ds[b] = ds[b].where(ds[b] >= -100*encoding[b]['scale_factor'])
            # binarize QA -> if bit 0 or 2 is set (cloud or shadow) -> True, see https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MOD09A1
            ds['c_mask'] = xr.apply_ufunc(lambda x: check_bits_set(x, 0, 0) | check_bits_set(x, 2, 2), ds['StateQA'], vectorize=True, dask='parallelized', output_dtypes=[bool])
            ds = ds.drop_vars(['StateQA','sur_refl_b05','sur_refl_b07'])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = fill_missing_dates(ds, '8D')
            return ds
        sv_patches(gdf, name, transform=transform, encoding=encoding)

    # modis lai/fpar: 0.1/0.01 -> uint8; kept
    name = 'modis_lai'
    load_name = version_name + '_' + name
    if name in modality:
        bands = ['Lai_500m', 'Fpar_500m']
        encoding={d: {'dtype': 'uint8', 'scale_factor': 0.1, '_FillValue': np.iinfo(np.uint8).max} for d in ['Lai_500m']}
        encoding.update({d: {'dtype': 'uint8', 'scale_factor': 0.01, '_FillValue': np.iinfo(np.uint8).max} for d in ['Fpar_500m']})
        def transform(row):
            years = range(year_start, year_end+1)
            year_file_str = '|'.join([str(y) for y in years])
            ds = load_netcdf_distributed_files(os.path.join(data_path, rf'{load_name}/{geoid}/{load_name}_{row["GEOID_PID"].iloc[0]}_({year_file_str}|{year_end+1}-01-).*.nc$'))
            for k in bands:
                # remove fill values 249-255, see: https://lpdaac.usgs.gov/documents/2/mod15_user_guide.pdf
                ds[k] = ds[k].where(ds[k] < 249)
                # apply scale factor
                ds[k] = ds[k] * encoding[k]['scale_factor']
            # binarize QA -> if bit 3 xor 4 is set (clouds) -> True https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MYD15A2H
            ds['c_mask'] = xr.apply_ufunc(lambda x: check_bits_set(x, 3, 3) ^ check_bits_set(x, 4, 4), ds['FparLai_QC'], vectorize=True, dask='parallelized', output_dtypes=[bool])
            # if bit 5 and 6 are set (clouds, shadows) -> True
            ds['c_mask_fpar_extra'] = xr.apply_ufunc(lambda x: check_bits_set(x, 5, 6, all_set=False), ds['FparExtra_QC'], vectorize=True, dask='parallelized', output_dtypes=[bool])
            ds = ds.drop_vars(['FparLai_QC', 'FparExtra_QC', 'FparStdDev_500m', 'LaiStdDev_500m'])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = fill_missing_dates(ds, '8D')
            return ds
        sv_patches(gdf, 'modis_lai_fpar', transform=transform, encoding=encoding)

    # modis lst: 0.02 -> uint32 (in Kelvin); offset -200 -> uint8 (drop the numbers after comma)
    name = 'modis_lst'
    load_name = version_name + '_' + name
    if name in modality:
        encoding={d: {'dtype': 'uint8', 'add_offset': 200, '_FillValue': np.iinfo(np.uint8).min} for d in ['LST_Day_1km', 'LST_Night_1km']}
        def transform(row):
            years = range(year_start, year_end+1)
            year_file_str = '|'.join([str(y) for y in years])
            ds = load_netcdf_distributed_files(os.path.join(data_path, rf'{load_name}/{geoid}/{load_name}_{row["GEOID_PID"].iloc[0]}_({year_file_str}|{year_end+1}-01-).*.nc$'))
            ds['LST_Day_1km'] = ds['LST_Day_1km'] * 0.02
            ds['LST_Night_1km'] = ds['LST_Night_1km'] * 0.02
            # binarize the QC -> if bit 0 is set (cloud) -> True, see https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MOD11A2
            ds['c_mask_day'] = xr.apply_ufunc(lambda x: check_bits_set(x, 0, 0), ds['QC_Day'], vectorize=True, dask='parallelized', output_dtypes=[bool])
            ds['c_mask_night'] = xr.apply_ufunc(lambda x: check_bits_set(x, 0, 0), ds['QC_Night'], vectorize=True, dask='parallelized', output_dtypes=[bool])
            ds = ds.drop_vars(['QC_Day', 'QC_Night'])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = fill_missing_dates(ds, '8D')
            return ds
        sv_patches(gdf, 'modis_'+name, transform=transform, encoding=encoding)

    # daymet
        # tmax, tmin: in celsius -> float32; to kelvin (+ 273,15 and -200) = + 73,15 -> uint8 (drop the numbers after comma)
        # prcp: mm/day (between 0 - ~2000) -> float32; scale by 100 -> uint16
        # vp Pa (between and 10000) -> float32; -> uint16 (loose precision after comma)
        # srad W/m^2 (between 0 and 2000) -> float32; scale by 100 -> uint16
        # swe kg/m^2 (between 0 and 15000) -> float32; -> uint16 (loose precision after comma)
    # hcw numbers between 0 and 1 -> float; scale by 100 -> uint8
    name = 'daymet'
    if name in modality and not check_exists(nr_files, geoid, name, overwrite=overwrite):
        encoding={d: {'dtype': 'uint8', 'add_offset': 200, '_FillValue': np.iinfo(np.uint8).min} for d in ['tmin', 'tmax']}
        encoding.update({d: {'dtype': 'uint16', 'scale_factor': 0.01, '_FillValue': np.iinfo(np.uint16).max} for d in ['prcp', 'srad']})
        encoding.update({d: {'dtype': 'uint16', '_FillValue': np.iinfo(np.uint16).max} for d in ['vp']})
        encoding.update({d: {'dtype': 'uint8', 'scale_factor': 0.01, '_FillValue': np.iinfo(np.uint8).max} for d in ['hcw']})
        # load all data for patches
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore')
            shp = gpd.GeoDataFrame(geometry=[gdf.unary_union], crs=gdf.crs)
            daymet = u_dymt.load_daymet_patch(shp,
                                              os.path.join(home_folder, 'data/daymet/'), bands=['tmax', 'tmin'],
                                              crs=gdf.crs, nr_jobs=2, time_slice=('1980-01-01', dates[1]))
            daymet_rest = u_dymt.load_daymet_patch(shp,
                                              os.path.join(home_folder, 'data/daymet/'), bands=['prcp', 'vp', 'srad'],
                                              crs=gdf.crs, nr_jobs=2, time_slice=(dates[0], dates[1]))
            hcw_index = u_dymt.compute_hcw_index(daymet, ['tmin', 'tmax'], quantiles=[.01,.05,.1,.9,.95,.99], start_date=dates[0], end_date=dates[1], at_least_three_days=True)
            daymet = daymet.sel(time=slice(dates[0], dates[1])) # now select the right time slice for daymet
            daymet['prcp'] = daymet_rest['prcp']
            daymet['vp'] = daymet_rest['vp']
            daymet['srad'] = daymet_rest['srad']
        # change the format
        daymet['tmin'] += 273.15
        daymet['tmax'] += 273.15
        daymet['hcw'] = hcw_index
        sv_patches(gdf, name, data=daymet, encoding=encoding)
    
    name = "daymet_clim"
    if name in modality and not check_exists(nr_files, geoid, name, overwrite=overwrite):
        # encoding={d: {'dtype': 'float32'} for d in ['tmin_mean', 'tmax_mean', 'prcp_mean', 'srad_mean', 'vp_mean', 'tmin_std', 'tmax_std', 'prcp_std', 'srad_std', 'vp_std']}
        encoding={d: {'dtype': 'uint16', 'add_offset': 200, 'scale_factor': 0.01, '_FillValue': np.iinfo(np.uint16).min} for d in ['tmin_mean', 'tmax_mean']}
        encoding.update({d: {'dtype': 'uint16', 'scale_factor': 0.01, '_FillValue': np.iinfo(np.uint16).max} for d in ['prcp_mean', 'srad_mean']})
        encoding.update({d: {'dtype': 'uint16', 'scale_factor': 0.1, '_FillValue': np.iinfo(np.uint16).max} for d in ['vp_mean']})
        encoding.update({d: {'dtype': 'uint16', 'scale_factor': 0.01, '_FillValue': np.iinfo(np.uint16).max} for d in ['tmin_std', 'tmax_std', 'prcp_std', 'srad_std', 'vp_std']})

        shp = gpd.GeoDataFrame(geometry=[gdf.unary_union], crs=gdf.crs)
        # load all data for patches
        daymet_clim = u_dymt.load_daymet_patch(shp, os.path.join(home_folder, 'data/daymet/'), bands=['tmin', 'tmax', 'vp', 'prcp', 'srad'], crs=gdf.crs, nr_jobs=2, time_slice=('1980-01-01', '2017-12-31'))
        daymet_clim['tmin'] += 273.15 # to kelvin
        daymet_clim['tmax'] += 273.15
        daymet_clim_mean, daymet_clim_std = u_dymt.climate_standardization(daymet_clim, return_z=False, return_mean_std=True)
        # rename vars
        for var in daymet_clim_mean.data_vars:
            daymet_clim_mean = daymet_clim_mean.rename({var: f"{var}_mean"})
            daymet_clim_std = daymet_clim_std.rename({var: f"{var}_std"})
        # merge them
        for var in daymet_clim_std.data_vars:
            daymet_clim_mean[var] = daymet_clim_std[var]
        sv_patches(gdf, name, data=daymet_clim_mean, encoding=encoding)

    # usdm numbers between 0 and 5; -> unit8
    name = 'usdm'
    if name in modality:
        def transform(row):
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore')
                
                daymet = u_dymt.load_daymet_patch(row, os.path.join(home_folder, 'data/daymet/'), bands=['tmax'], time_slice=('2022-01-01', '2022-01-01'), crs=row.crs)
                usdm = u_usdm.load_usdm_patch(row, os.path.join(home_folder, 'data/USDM/'), time_slice=dates, ref_ds=daymet)
                usdm['usdm'] = usdm['usdm'].astype('uint8')
            return usdm
        sv_patches(gdf, name, transform=transform)

    # cdl
    name = 'cdl'
    if name in modality:
        def transform(row):
            classes = {'corn': 1, 'cotton': 2,'soybean': 5,'winter_wheat': 24,'oats': 28}
            path = os.path.join(home_folder, 'data/shared/cropland/')
            ds = u_cdl.load_cdl_patch(row, path, classes=classes, confidence=.3, time_slice=dates, crs=row.crs)
            
            # create county mask
            if 'time' in ds.sizes:
                da_mask = ds[list(ds.data_vars)[0]].isel(time=0)
                if 'time' in ds.sizes:
                    da_mask.drop('time')
            else:
                da_mask = ds[list(ds.data_vars)[0]]
            da_mask = da_mask.astype('float')
            mask = da_mask.rio.clip(shp_co.geometry, all_touched=True, drop=False)
            ds['co_mask'] = mask.notnull()
            ds['co_mask'] = ds['co_mask'].astype(bool)
            
            # create the 500m version
            def transform2(row):
                # load the corresponding modis data
                pid = row["GEOID_PID"].iloc[0].split("_")[1]
                modis = xr.open_zarr(os.path.join(sv_path, f'modis/modis_{row["GEOID_PID"].iloc[0].split("_")[0]}_{number_to_range(int(pid), nr_groups)}.zarr'), group=f'{pid}')
                modis = modis.isel(time=0)
                ds_l = []
                for i in range(len(ds.time)):
                    ds2 = u_cdl.create_coarse_cdl_map(modis, ds['cdl_30m'].isel(time=i), threshold=40).astype('int8')
                    ds2 = ds2.assign_coords(time=ds.time[i])
                    ds_l.append(ds2)
                ds2 = xr.concat(ds_l, dim='time')
                ds2 = ds2.to_dataset(name='cdl_500m')
                # county mask for 500m
                da_mask = ds2[list(ds2.data_vars)[0]].isel(time=0)
                da_mask = da_mask.astype('float')
                mask2 = da_mask.rio.clip(shp_co.geometry, all_touched=True, drop=False)
                ds2['co_mask'] = mask2.notnull()
                ds2['co_mask'] = ds2['co_mask'].astype(bool)

                ds2.attrs = ds.attrs
                return ds2
            sv_patches(row, 'cdl_500m', transform=transform2)
            return ds
        sv_patches(gdf, name, transform=transform)

    # nlcd
    name = 'nlcd'
    if name in modality:
        def transform(row):
            path = os.path.join(home_folder, 'data/shared/landcover/usa_nlcd/')
            classes = {'Open water': 11,'Perennial ice/snow': 12,'Developed, open space': 21,'Developed, low intensity': 22,'Developed, medium intensity': 23,
                    'Developed high intensity': 24,'Barren land (rock/sand/clay)': 31,'Deciduous forest': 41,'Evergreen forest': 42,'Mixed forest': 43,
                    'Dwarf scrub': 51,'Shrub/scrub': 52,'Grassland/herbaceous': 71,'Sedge/herbaceous': 72,'Lichens': 73,'Moss': 74,'Pasture/hay': 81,
                    'Cultivated crops': 82,'Woody wetlands': 90,'Emergent herbaceous wetlands': 95}
            ds = u_cdl.load_nlcd_patch(row, path, classes=classes, time_slice=dates, crs=row.crs)
            return ds
        sv_patches(gdf, name, transform=transform)

    # DEM: left as float32 since they are only a single layer -> not much memory
    # aspect between 0 - 360
    # slope between 0 - 90
    # hillshade between 0 - 255
    name = 'dem'
    load_name = version_name + '_' + name
    if name in modality:
        def transform(row):
            fn = os.path.join(home_folder, 'data/shared/usgs_dem/USGS_Seamless_DEM_1.vrt')
            ds = u_soil.load_USGS_3DEP_patch(row, fn)
            return ds
        sv_patches(gdf, name, transform=transform)

    # soil
    name = 'soil'
    load_name = version_name + '_' + name
    if name in modality:
        def transform(row):
            fn = os.path.join(home_folder, 'data/shared/soil/soil_grid/soil_grid.tif')
            return u_soil.load_soil_grid_patch(row, fn)
        sv_patches(gdf, name, transform=transform)

    # landsat: 0.0000275 -0.2 -> uint16; kept
    name = 'landsat'
    load_name = version_name + '_' + name
    if name in modality:
        bands = ['B02', 'B03', 'B04', 'B05', 'B06']
        # 0 is no_data see page 12: https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1619_Landsat8-9-Collection2-Level2-Science-Product-Guide-v5.pdf
        encoding={d: {'dtype': 'uint16', 'scale_factor': 0.0000275, 'add_offset': -.2, '_FillValue': np.iinfo(np.uint16).min} for d in bands}
        def transform(row):
            years = range(year_start, year_end+1)
            year_file_str = '|'.join([str(y) for y in years])
            ds = load_netcdf_distributed_files(os.path.join(data_path, rf'{load_name}/{geoid}/{load_name}_{row["GEOID_PID"].iloc[0]}_.*.nc$'))
            # binarize QA -> if bit 1-4, is set (cloud, shadow) -> True
            # https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1619_Landsat8-9-Collection2-Level2-Science-Product-Guide-v5.pdf
            ds['c_mask'] = xr.apply_ufunc(lambda x: check_bits_set(x, 1, 4, all_set=False), ds['BQA'], vectorize=True, dask='parallelized', output_dtypes=[bool])
            # select the pixels with the least amount of clouds
            for b in bands:
                ds[b] = (ds[b] * encoding[b]['scale_factor']) + encoding[b]['add_offset']
            for v in ['BQA']:
                if v in ds.data_vars:
                    ds = ds.drop_vars(v)
            # do some temporal alignment to the same dates
            ds = fill_missing_dates(ds, '16D', temp_align=True)
            return ds
        sv_patches(gdf, name+'8', transform=transform, encoding=encoding)

    # sen2 0.0001 -> float32; -> use uint16
    name = 'sen2'
    load_name = version_name + '_' + name
    if name in modality:
        # blue, green, red, rededge, nir, swir16
        bands = ["B02","B03","B04","B06","B08","B11"]
        # 0 is no_data, hence the fill value is set to the minimum: page 384: https://sentinel.esa.int/documents/d/sentinel/s2-pdgs-cs-di-psd-v15-0
        encoding={d: {'dtype': 'uint16', 'scale_factor': 0.0001, '_FillValue': np.iinfo(np.uint16).min} for d in bands}
        def transform(row):
            ds = load_netcdf_distributed_files(os.path.join(data_path, rf'{load_name}/{geoid}/{load_name}_{row["GEOID_PID"].iloc[0]}_.*.nc$'))
            ds = u_sen2.align_processing_version(ds, str(row['GEOID_PID'].values[0]))
            # binarize QA -> if numbers (cloud, shadow) -> True https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm-overview
            ds['c_mask'] = xr.apply_ufunc(lambda x: x in [0,1,3,8,9,10], ds['SCL'], vectorize=True, dask='parallelized', output_dtypes=[bool])
            if 'spatial_ref' in ds.data_vars:
                del ds['spatial_ref']
            ds = mosaic(ds, interval='1D', func=select_pixels_by_cloud_cover, band='c_mask', invalid_values=[255], proxy_band='SCL', adjust_for_conditions=bands)
            for b in bands:
                ds[b] = ds[b] * encoding[b]['scale_factor']
            for var in ['SCL', 'B01', 'B05', 'B07', 'B10', 'B12', 'B8A', 'B09']:
                if var in ds.data_vars:
                    ds = ds.drop_vars(var)
            # do some temporal alignment to the same dates
            ds = fill_missing_dates(ds, '15D', temp_align=True)
            # drop tile ids since they are now mosaics
            ds = ds.drop_vars(['id'])
            return ds
        sv_patches(gdf, name, transform=transform, encoding=encoding)

    name = "sen1"
    load_name = version_name + '_' + name
    if name in modality:
        bands = ["vv", "vh"]
        db_min, db_max = -50, 20
        # nodata: -32768
        encoding={d: {'dtype': 'uint16', 'scale_factor': (np.abs(db_min) + db_max) / 65535, 'add_offset': db_min, '_FillValue': np.iinfo(np.uint16).min} for d in bands}
        drop_less_than_ts = 5
        def transform(row):
            ds = load_netcdf_distributed_files(os.path.join(data_path, rf'{load_name}/{geoid}/{load_name}_{row["GEOID_PID"].iloc[0]}_.*.nc$'))
            if 'spatial_ref' in ds.data_vars:
                del ds['spatial_ref']
            # split into a and d
            ds_a = ds.where(ds['sat:orbit_state'] == 'ascending', drop=True)
            ds_d = ds.where(ds['sat:orbit_state'] == 'descending', drop=True)
            # 2. orbits to 12 day resolution
            if len(ds_a.time) > 0:
                ds_a = mosaic(ds_a, interval='12D', func=select_pixels_by_first_valid_value, invalid_values=[0, np.iinfo(np.uint16).max, np.iinfo(np.int16).min, np.nan], sort_by_amount=True)
                ds_a = rm_time_step(ds_a, threshold=.7, invalid_values=[0, np.iinfo(np.uint16).max, np.iinfo(np.int16).min, np.nan], band='vv')
                if len(ds_a.time) > 0:
                    valid_years = (
                        ds_a["vv"]
                        .groupby("time.year")
                        .count()
                        .where(lambda x: x >= drop_less_than_ts, drop=True)
                        .year
                    )
                else:
                    valid_years = []
                ds_a = ds_a.where(ds_a["time.year"].isin(valid_years), drop=True)
            if len(ds_d.time) > 0:
                ds_d = mosaic(ds_d, interval='12D', func=select_pixels_by_first_valid_value, invalid_values=[0, np.iinfo(np.uint16).max, np.iinfo(np.int16).min, np.nan], sort_by_amount=True)
                ds_d = rm_time_step(ds_d, threshold=.7, invalid_values=[0, np.iinfo(np.uint16).max, np.iinfo(np.int16).min, np.nan], band='vv')
                if len(ds_d.time) > 0:
                    valid_years = (
                        ds_d["vv"]
                        .groupby("time.year")
                        .count()
                        .where(lambda x: x >= drop_less_than_ts, drop=True)
                        .year
                    )
                else:
                    valid_years = []
                ds_d = ds_d.where(ds_d["time.year"].isin(valid_years), drop=True)
            # concat and order by time again
            ds = xr.concat([ds_a, ds_d], dim='time').sortby('time')
            if len(ds.time) < 1:
                raise ValueError("No valid time steps found.")
            # resample to 20m
            ds = ds.rio.write_crs(ds.rio.crs)
            ds = ds.rio.reproject(ds.rio.crs, resolution=20, resampling=Resampling.bilinear)
            # drop added fillvalue from reproject
            for b in bands:
                ds[b].attrs.pop('_FillValue', None)
            # convert to dB
            ds = ds.map(db_scale)
            # clip to min and max
            ds = ds.clip(db_min, db_max)
            # apply no scaling -> will be automatically applied when saving

            ds = ds.drop_vars([v for v in ds.coords if v not in ['time', 'x', 'y', 'sat:orbit_state']])
            return ds
        sv_patches(gdf, name, transform=transform, encoding=encoding)

def rm_time_step(ds, threshold, invalid_values, band):
    mask = ds[band].isnull() | ds[band].isin(invalid_values)
    time_step_rm = ((mask.sum(dim=('x', 'y')) / (ds.sizes['x']*ds.sizes['y'])) > threshold).compute()
    if time_step_rm.any():
        ds = ds.where(~time_step_rm, drop=True)
    return ds

def fill_missing_dates(ds, freq, temp_align=False):
    l = []
    for y in range(2018, 2023):
        date_range = pd.date_range(f'{y}-01-01', f'{y}-12-31', freq=freq)
        date_range = date_range.append(pd.DatetimeIndex([pd.Timestamp(f'{y+1}-01-01')]))
        l_n = []
        for i in range(len(date_range)-1):
            da = ds.sel(time=slice(date_range[i].strftime("%Y-%m-%d"), (date_range[i+1] - pd.Timedelta('1D')).strftime("%Y-%m-%d")))
            if len(da.time) > 1:
                raise RuntimeError('More than one date found')
            if len(da.time) == 0:
                # insert nan's for missing dates
                da = xr.full_like(ds.isel(time=0), np.nan)
                da["time"] = [date_range[i]]
            if temp_align:
                da["time"] = [date_range[i]]
            l_n.append(da)
        ds_n = xr.concat(l_n, dim='time')
        l.append(ds_n)
    return xr.concat(l, dim='time')

def db_scale(x):
    # convert linear to dB
    with np.errstate(divide='ignore', invalid='ignore'):
        x = 10 * np.where(x <= 0, np.nan, np.log10(x))
    return x

geoids = gdf['GEOID'].tolist()
fns = [os.path.join(home_folder, f'CropClimateX/patches/final/patches_{geoid}.geojson') for geoid in geoids]
fn_co = os.path.join(home_folder, 'data/shared/geometry/county/tl_2018_us_county_mainland.shp')
gdf_cos = gpd.read_file(fn_co)
data_path = os.path.join(home_folder, 'data/shared/raw/')
sv_path = os.path.join(home_folder, f'CropClimateX/')
num_workers=int(os.cpu_count()//2)
modality=None
import argparse
p = argparse.ArgumentParser()
p.add_argument('-m','--modality', nargs='*')
args = p.parse_args()
if args.modality:
    modality = args.modality
print(modality)
Parallel(n_jobs=num_workers)(delayed(prepare_patches)(gpd.read_file(file), os.path.basename(file).split('.')[0].split('_')[1], gdf_cos, data_path, sv_path, dates, version_name, modality=modality, overwrite=False) for file in tqdm(fns, desc=','.join(modality)))
