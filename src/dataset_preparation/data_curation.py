# %%
import glob
import os
import warnings
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import xarray as xr
import rioxarray as rxr
import rootutils
import utils_sentinel2 as u_sen2
import utils_soil as u_soil
import utils_download as u_dl
import utils as u
from utils import split_date_range, shp_to_utm_crs
import re
from joblib import Parallel, delayed
from tqdm import tqdm
# %% vars
home_folder = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sv_folder = os.path.join(home_folder, 'data/shared/raw/')
curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# %%
def file_included_exists(fn, dates):
    out = False
    if dates is None:
        return out
    date1 = pd.to_datetime(dates[1], format="%Y-%m-%d")
    date0 = pd.to_datetime(dates[0], format="%Y-%m-%d")
    # if date1 - date0 <= pd.Timedelta(days=366): # needs to check if yearly file exists
    year = re.findall(r'\d{4}', fn)[-1]
    start_date = '01-01'
    end_date = '12-31'
    if year == dates[0][:4]:
        start_date = dates[0][5:]
    if year == dates[1][:4]:
        end_date = dates[1][5:]
    yearly_fn = re.sub(r'\d{4}-\d{2}-\d{2}', f'{year}-{end_date}', fn, count=0)
    yearly_fn = re.sub(r'\d{4}-\d{2}-\d{2}', f'{year}-{start_date}', yearly_fn, count=1)
    out = os.path.exists(yearly_fn) or out
    if date1 - date0 <= pd.Timedelta(days=31): # needs to check if monthly file exists
        # since dates could be strechted over two months -> check if for both month files exist
        start_date = date0.strftime('%Y-%m') + '-01'
        end_date = (date0 + MonthEnd(0)).strftime('%Y-%m-%d')
        monthly_fn = re.sub(r'\d{4}-\d{2}-\d{2}', end_date, fn, count=0)
        monthly_fn = re.sub(r'\d{4}-\d{2}-\d{2}', start_date, monthly_fn, count=1)
        month_1_exists = os.path.exists(monthly_fn)
        start_date = date1.strftime('%Y-%m') + '-01'
        end_date = (date1 + MonthEnd(0)).strftime('%Y-%m-%d')
        monthly_fn = re.sub(r'\d{4}-\d{2}-\d{2}', end_date, fn, count=0)
        monthly_fn = re.sub(r'\d{4}-\d{2}-\d{2}', start_date, monthly_fn, count=1)
        month_2_exists = os.path.exists(monthly_fn)
        out = (month_1_exists and month_2_exists) or out
    return out

def download_entity(dict, geoid, name=None, dates=None, overwrite=False, transform=None, start_end_dates=None):
    suffix = ''
    if dates:
        suffix = f'_{dates[0]}_{dates[1]}'
    fn = os.path.join(sv_folder, name, geoid.split('_')[0], f'{name}_{geoid}{suffix}.nc')
    if (os.path.exists(fn) or file_included_exists(fn, start_end_dates)) and not overwrite:
        return

    # download the data
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore',category=RuntimeWarning)
            data = u_dl.create_cube(**dict, tmp_folder=sv_folder)
            if transform:
                data = transform(data)
            # save the data
            if not name:
                name = data.name
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            data.to_netcdf(fn, engine='h5netcdf')
    except Exception as e:
        if os.path.exists(fn):
            os.remove(fn)
        # raise
        err_fn = f'dc_error_{name}_{curr_time}.txt'
        engine = 'odc'
        if 'download_engine' in dict:
            engine = dict['download_engine']
        with open(err_fn, 'a') as file:
            file.write(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}:{engine},{geoid},{dates},{e}\n")
        warnings.warn(f"Could not download {name} for {geoid} with dates {dates}: {e}")

def download_patch(i, shp, dates, name, modality=['modis', 'dem', 'gnatsgo', 'sen2', 'landsat'], splitting='monthly', download_engine='odc'):
    geoid = shp['GEOID_PID'].values[0]
    display_position = (i%num_workers)+1

    yearly_dates = split_date_range(dates, 'yearly')
    monthly_dates = split_date_range(dates, splitting)
    landsat_dates = yearly_dates
    sentinel_dates = yearly_dates

    if 'modis' in modality:
        for d in tqdm(yearly_dates, position=display_position, leave=False, desc=f'modis_data.w{display_position}-{geoid}'):
            # https://planetarycomputer.microsoft.com/dataset/modis-09A1-061
            dic = dict(shp=shp, collection='modis-09A1-061', start_date=d[0], end_date=d[1],
                        bands=[f'sur_refl_b0{i}' for i in range(1,8)] + ["sur_refl_qc_500m", "sur_refl_state_500m"],
                        resolution=500, filter={'start_datetime':{'gte':d[0]}, "platform": {"eq": 'terra'}})
            dic = dict(shp=shp, collection="MODIS/061/MOD09A1", start_date=d[0], end_date=d[1], bands=[f'sur_refl_b0{i}' for i in range(1,8)] + ["StateQA"],
                        resolution=500, download_engine=download_engine)
            download_entity(dic, geoid, name+'_modis', d)
    if 'modis_lst' in modality:
        # https://planetarycomputer.microsoft.com/dataset/modis-11A2-061
        dic = dict(shp=shp, collection='modis-11A2-061', start_date=d[0], end_date=d[1],
                bands=["LST_Day_1km", "QC_Day", "LST_Night_1km", "QC_Night"],
                    resolution=1000, filter={'start_datetime':{'gte':d[0]}, "platform": {"eq": 'terra'}}, download_engine=download_engine)
        download_entity(dic, geoid, name, d)
    if 'modis_lai' in modality:
        # https://planetarycomputer.microsoft.com/dataset/modis-15A2H-061
        dic = dict(shp=shp, collection='modis-15A2H-061', start_date=d[0], end_date=d[1],
                bands=["Lai_500m", "Fpar_500m", "LaiStdDev_500m", "FparStdDev_500m", "FparLai_QC", "FparExtra_QC"],
                    resolution=500, filter={'start_datetime':{'gte':d[0]}, "platform": {"eq": 'terra,aqua'}}, download_engine=download_engine)
        download_entity(dic, geoid, name, d)

    if 'gnatsgo' in modality:
        # https://planetarycomputer.microsoft.com/dataset/gnatsgo-rasters
        dic = dict(shp=shp, collection='gnatsgo-rasters', bands=["mukey", "aws0_150", "soc0_150", "rootznemc"], resolution=30, dtype='float32', download_engine=download_engine)
        chorizon_cols = ["cokey", "hzdepb_r", "claytotal_r", "sandtotal_r", "silttotal_r", "ph1to1h2o_r", "cec7_r"]
        component_cols = ["mukey", "cokey", "comppct_r"]
        chorizon_table = u_soil.prepare_soil_table(os.path.join(home_folder, 'data/shared/soil/tables/chorizon_gnastgo_official.csv'), chorizon_cols)
        component_table = u_soil.prepare_soil_table(os.path.join(home_folder, 'data/shared/soil/tables/component_gnatsgo_official.csv'), component_cols)
        download_entity(dic, geoid, name+'_gnatsgo', transform=lambda x: u_soil.create_soil_raster(x, chorizon_table, component_table))

    if 'gee' in download_engine:
        landsat_col = 'LANDSAT/LC08/C02/T1_L2' # tier 2 missing: "LANDSAT/LC08/C02/T2_L2"
        landsat_bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'QA_PIXEL']
        sentinel_col = 'COPERNICUS/S2_SR'
        sen_bands = ["B2","B3","B4","B6","B8", "B11","SCL"]
        harmonize_names=True
    elif 'sentinel_hub' in download_engine:
        landsat_col = 'landsat-8-l2'
        landsat_bands = ["B02","B03","B04","B05","B06","BQA"]
        sentinel_col = 'sentinel-2-l2a'
        sen_bands = ["B02","B03","B04","B06","B08", "B11","SCL","CLP"]
        harmonize_names=False
    else:
        landsat_col = 'landsat-c2-l2'
        landsat_bands = ["blue","green","red","nir08","swir16", "qa_pixel"]
        sentinel_col = 'sentinel-2-l2a'
        sen_bands = ["B02","B03","B04","B06","B08", "B11","SCL"]
        harmonize_names=False

    download_kwargs = {}
    transforms = None
    filter = None
    
    if 'landsat' in modality:
        if 'sentinel_hub' in download_engine:
            download_kwargs.update({"time_period":16})
        for d in tqdm(landsat_dates, position=display_position, leave=False, desc=f'landsat_data.w{display_position}-{geoid}'):
            dic = dict(shp=shp, collection=landsat_col, bands=landsat_bands,
                        resolution=30, start_date=d[0], end_date=d[1], filter={"platform": {"eq": 'landsat-8'}},
                        crop=False, download_engine=download_engine, harmonize_names=harmonize_names, download_kwargs=download_kwargs)
            download_entity(dic, geoid, name+'_landsat', d, start_end_dates=(landsat_dates[0][0],landsat_dates[-1][1]))
    if 'sen2' in modality:
        if 's3' in download_engine:
            filter = {"processingLevel": {"eq":"S2MSI2A"}}
            download_kwargs.update({"time_range":15, "stats_band": "SCL", "use_s3fs":True})
            from utils import select_pixels_by_cloud_cover
            download_kwargs.update({"mosaic_args": {'mask_u_func': lambda x: x in [0,1,3,8,9,10], 'interval':'1D',
                                    'func':select_pixels_by_cloud_cover, 'invalid_values':[255]}})
        elif 'sentinel_hub' in download_engine:
            transforms = None
            download_kwargs = dict(time_period=15, overwrite=False)
        elif 'pc' in download_engine:
            def transforms(ds): # not needed for sentinel hub
                ds = ds.astype('uint16') # save as diff data type
                ds = u_sen2.harmonize_sentinel2_dataset_to_old(ds)
                return ds

        for d in tqdm(sentinel_dates, position=display_position, leave=False, desc=f'sentinel_data.w{display_position}-{geoid}'):
            dic = dict(shp=shp, collection=sentinel_col, start_date=d[0], end_date=d[1],
                        bands=sen_bands, resolution=20, crop=False, download_engine=download_engine,
                        harmonize_names=harmonize_names, filter=filter, download_kwargs=download_kwargs)
            download_entity(dic, geoid, name+'_sen2', d, transform=transforms, start_end_dates=(sentinel_dates[0][0],sentinel_dates[-1][1]))
    if 'sen1' in modality:
        # pc
        collection ='sentinel-1-rtc'
        download_kwargs.update(dict(save_metadata=['id', 'sat:absolute_orbit', 'sat:relative_orbit', 's1:slice_number', 's1:total_slices', 'sar:looks_range', 'sar:looks_azimuth', 'sar:looks_equivalent_number', 'sat:orbit_state', 'sar:instrument_mode']))
        for d in tqdm(sentinel_dates, position=display_position, leave=False, desc=f'sentinel1_data.w{display_position}-{geoid}'):
            dic = dict(shp=shp, collection=collection, start_date=d[0], end_date=d[1],
                        bands=['vv', 'vh'], resolution=10, crop=False, download_engine=download_engine,
                        download_kwargs=download_kwargs,
                        )
            download_entity(dic, geoid, name+'_sen1', d, start_end_dates=(sentinel_dates[0][0],sentinel_dates[-1][1]))
# %%
def start_download(i, id, dates, engine='odc', buffer=False, skip_if_geoid_exists=False):
    """i: int nr of download
    id: geoid + pid"""
    if skip_if_geoid_exists:
        for m in modality:
            geoid = id.split('_')[0]
            fn2 = os.path.join(sv_folder, f'{name}_{m}', geoid, f'{name}_{m}_{id}*.nc')
            if len(glob.glob(fn2)) > 0:
                return
    geoid = id.split('_')[0]
    shps = gpd.read_file(os.path.join(home_folder, f'CropClimateX/patches/final/patches_{geoid}.geojson'))
    shp = shps[shps['GEOID_PID'] == id].copy()

    if buffer:
        res = 0
        if 'sen2' in modality:
            res = 20
        elif 'landsat' in modality:
            res = 30
        elif 'sen1' in modality:
            res = 10
        # since in utm crs, add just 8 pixels the resolution
        shp['geometry'] = shp['geometry'].buffer(res*4, resolution=res, join_style='mitre')
    download_patch(i, shp, dates, name, modality=modality, download_engine=engine)

# %%
modality=None
num_workers = os.cpu_count()//4
import argparse
p = argparse.ArgumentParser()
p.add_argument('-m','--modality', nargs='*')
p.add_argument('-w','--num_workers', type=int)
args = p.parse_args()
if args.modality:
    modality = args.modality
if args.num_workers:
    num_workers = args.num_workers
print(modality, ', workers:', num_workers)
# %% 
dates = ('2018-01-01', '2022-12-31')

if 'sen2'  in modality or 'sen1' in modality:
    gdf = gpd.read_file(os.path.join(home_folder, f'CropClimateX/tiny_county_list.geojson'))
else:
    gdf = gpd.read_file(os.path.join(home_folder, f'CropClimateX/county_list.geojson'))
# gdf = gdf.sample(frac=1) # randomize the df
# gdf = gdf[gdf['GEOID'].isin(['18115'])]
ids = []
for idx in gdf.index:
    geoid = gdf.loc[[idx]]['GEOID'].values[0]
    shps = gpd.read_file(os.path.join(home_folder, f'CropClimateX/patches/final/patches_{geoid}.geojson'))
    ids.extend(shps['GEOID_PID'].values)
print(len(ids))
# %% download with cdse
Parallel(n_jobs=num_workers)(delayed(start_download)(i, id, dates, engine='cdse_s3', buffer=True) for i, id in tqdm(enumerate(ids), total=len(ids), position=0, leave=True, desc=f'patches {modality}'))
# %% or download with sentinelhub/...
# Parallel(n_jobs=num_workers)(delayed(start_download)(i, id, dates, engine='sentinel_hub', buffer=True) for i, id in tqdm(enumerate(ids), total=len(ids), position=0, leave=True, desc=f'patches {modality}'))
