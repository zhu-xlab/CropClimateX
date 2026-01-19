import os
import shutil
import requests
import urllib
import re
import xarray as xr
import pandas as pd
import rasterio
import rioxarray as rxr
from tqdm import tqdm
from joblib import Parallel, delayed
import itertools
import warnings
import json
import re
import hashlib
import sentinelhub as sh
import numpy as np
from utils import rm_attrs, rm_coords, split_date_range_by_freq
import geopandas as gpd
from shapely.geometry import Polygon, box
import terragon

gee_to_pc_bands_sen2 = {
    'B1': 'B01',
    'B2': 'B02',
    'B3': 'B03',
    'B4': 'B04',
    'B5': 'B05',
    'B6': 'B06',
    'B7': 'B07',
    'B8': 'B08',
    'B8A': 'B8A',
    'B9': 'B9',
    'B11': 'B11',
    'B12': 'B12',
    'SCL': 'SCL'
}
gee_to_pc_bands_landsat = {
    'SR_B1': '',
    'SR_B2': 'blue',
    'SR_B3': 'green',
    'SR_B4': 'red',
    'SR_B5': 'nir08',
    'SR_B6': 'swir16',
    'SR_B7': 'swir22',
    'ST_B10': 'lwir11',
    'ST_QA': 'qa',
    'QA_PIXEL': 'qa_pixel'
}

def create_cube(shp, collection, resolution, start_date=None, end_date=None, bands=None, crop=True, download_engine='odc',
                filter=None, dtype=None, harmonize_names=False, tmp_folder='tmp/', clip_kwargs:dict={}, download_kwargs:dict={}):
    """Download a cube defined by shp bounds, assumes to use the same crs as shp.

    Crop to shp if crop is True.
    """
    if not isinstance(bands, (list,tuple)) and bands is not None:
        bands = [bands]

    if 'geedim' == download_engine:
        data = download_gee(shp, collection, resolution, start_date, end_date, bands, os.path.join(tmp_folder, 'tmp', 'gee', collection.replace('/', '-')), **download_kwargs)
    elif 'odc' == download_engine or 'stackstac' == download_engine:
        data = download_pc(shp, collection, resolution, start_date, end_date, bands, download_engine, filter, dtype, tmp_folder, **download_kwargs)
    elif 'sentinel_hub' == download_engine:
        data = download_sentinelhub(shp, collection, resolution, start_date, end_date, bands, os.path.join(tmp_folder, 'tmp', 'sentinel_hub', collection), filter, **download_kwargs)
    elif 'cdse_s3' == download_engine:
        if collection == 'sentinel-2-l2a':
            collection = 'SENTINEL-2'
        data = download_cdse_s3(shp, collection, resolution, start_date, end_date, bands, filter, os.path.join(tmp_folder, 'tmp', download_engine, collection), **download_kwargs)
    elif 'pc' == download_engine:
        data = download_pc(shp, collection, resolution, start_date, end_date, bands, 'pc', filter, tmp_folder, **download_kwargs)
    elif 'cdse' == download_engine:
        data = download_cdse(shp, collection, resolution, start_date, end_date, bands, filter, tmp_folder, **download_kwargs)
    else:
        raise ValueError(f"Download engine {download_engine} not supported.")
    
    if harmonize_names: # rename gee band names to pc names or vice versa
        filtered_dict = None
        if 'sentinel-2-l2a' in collection or 'COPERNICUS/S2_SR' in collection:
            filtered_dict = {key: value for key, value in gee_to_pc_bands_sen2.items() if key in data.data_vars}
        elif 'landsat' in collection or 'LANDSAT/LC08/C02/T1_L2' in collection:
            filtered_dict = {key: value for key, value in gee_to_pc_bands_landsat.items() if key in data.data_vars}
        else:
            warnings.warn('Unknown collection, no renaming or transforming done.')
        if filtered_dict:
            if not 'gee' in download_engine:
                filtered_dict = {value: key for key, value in filtered_dict.items()}
            data = data.rename_vars(filtered_dict)

    l = ['x', 'y', 'latitude', 'longitude', 'band', 'id', 'time', 'processorVersion'] + download_kwargs.get('save_metadata', [])
    data = rm_coords(data, l)
    data = rm_attrs(data)

    import rioxarray  # reinit rioxarray in case it is not initialized in the threads
    data.rio.write_crs(shp.crs, inplace=True)
    data.attrs['crs'] = shp.crs.to_string()
    data.attrs['source'] = download_engine
    if crop:
        data = data.rio.clip(shp.geometry, shp.crs, **clip_kwargs)
    if dtype:
        data = data.astype(dtype)

    return data

def download_pc(shp, collection, resolution, start_date, end_date, bands, download_engine, filter, tmp_folder, **kwargs):
    tg = terragon.init('pc')
    ds = tg.create(shp=shp, collection=collection, bands=bands, start_date=start_date, end_date=end_date, resolution=resolution,
    filter=filter, download_folder=tmp_folder, **kwargs)
    return ds

def download_cdse(shp, collection, resolution, start_date, end_date, bands, filter, tmp_folder, **kwargs):
    credentials = {'aws_access_key_id': os.environ.get("S3_ACCESS_KEY"), 'aws_secret_access_key': os.environ.get("S3_SECRET_KEY")}
    tg = terragon.init('cdse', credentials=credentials, base_url="https://stac.dataspace.copernicus.eu/v1")
    ds = tg.create(shp=shp, collection=collection, bands=bands, start_date=start_date, end_date=end_date, resolution=resolution,
    filter=filter, download_folder=tmp_folder, **kwargs)
    return ds

def download_gee(shp, collection, resolution, start_date, end_date, bands, tmp_folder, num_workers=10):
    tg = terragon.init('gee')
    ds = tg.create(shp=shp, collection=collection, bands=bands, start_date=start_date, end_date=end_date,
                   resolution=resolution, num_workers=num_workers, download_folder=tmp_folder, rm_tmp_files=False)
    return ds

def rm_temp_files(fns):
    for fn in fns:
        try:
            os.remove(fn)
        except Exception as e:
            print(f"Failed to remove file in temporary download folder {fn}: {e}")

def sentinelhub_cloud_stats(shp, collection, start_date, end_date, sh_config, tmp_folder):
    """get the clouds stats for the given time period"""
    
    fn = os.path.join(tmp_folder, 'stats_request', f"sh_stats_{collection}_{start_date}_{end_date}_{shp['GEOID_PID'].values[0]}.json")
    if not os.path.exists(fn):
        if collection == 'sentinel-2-l2a':
            resolution = 160
            data_collection = sh.DataCollection.SENTINEL2_L2A
            evalscript=f"""
            //VERSION=3
            function setup() {{
            return {{
                input: [{{
                bands: [
                    "CLP",
                    "dataMask"
                ]
                }}],
                mosaicking: "SIMPLE",
                output: [
                {{
                    id: "data",
                    bands: 1
                }},
                {{
                    id: "dataMask",
                    bands: 1
                }}]
            }}
            }}
            function evaluatePixel(samples) {{
                return {{
                    data: [samples.CLP / 255],
                    dataMask: [samples.dataMask]
                    }}
            }}
            """
        elif collection == 'landsat-8-l2':
            resolution = 30
            data_collection = sh.DataCollection.LANDSAT_OT_L2
            evalscript="""
            //VERSION=3
            function setup() {
            return {
                input: [{
                bands: [
                    "BQA",
                    "dataMask"
                ]
                }],
                mosaicking: "SIMPLE",
                output: [
                {
                    id: "data",
                    bands: 1
                },
                {
                    id: "dataMask",
                    bands: 1
                }]
            }
            }
            function evaluatePixel(samples) {
                var qa = decodeL8C2Qa(samples.BQA)
                var cloud = qa.cloud | qa.cloudShadow | qa.cirrus
                return {
                    data: [cloud],
                    dataMask: [samples.dataMask]
                    }
            }
            """
        else:
            raise ValueError(f"Collection {collection} not supported for sentinelhub.")

        bbox = sh.BBox(bbox=list(shp.total_bounds), crs=sh.CRS(shp.crs.to_epsg()))

        aggregation = sh.SentinelHubStatistical.aggregation(
            evalscript=evalscript,
            time_interval=(start_date, end_date),
            aggregation_interval='P1D',
            resolution=(resolution,resolution),
            )

        request = sh.SentinelHubStatistical(
            aggregation=aggregation,
            input_data=[sh.SentinelHubStatistical.input_data(
                data_collection=data_collection,
            )],
            bbox=bbox,
            config=sh_config
        )
        js = request.get_data()

        if js[0]['status'] != 'OK':
            raise ValueError(f"Error in request: {js[0]['status']}")
        if len(js[0]['data']) == 0:
            warnings.warn(f"Stats API - No data found for {start_date} - {end_date}")

        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, 'w') as f:
            json.dump(js, f)
    else:
        with open(fn, 'r') as f:
            js = json.load(f)

    return js

def calc_sh_cloud_val(cloudy, no_data, num_pixels):
    if isinstance(cloudy, str):
        if cloudy == 'NaN' and no_data == num_pixels:
            cloudy = 1 # no data -> does not want to choose this
    # calculate the percentage of cloudy (already percentage) + no_data pixels
    return cloudy *100 + (no_data/(num_pixels+1e-6)*100)

def sentinelhub_best_days(shp, collection, start_date, end_date, time_period, sh_config, tmp_folder, overwrite=False):
    """returns best days within a time_period period from sentinelhub stats api"""
    fn = os.path.join(tmp_folder, f"dates_{collection}_{start_date}_{end_date}_{shp['GEOID_PID'].values[0]}.json")
    if not os.path.exists(fn) or overwrite:
        js = sentinelhub_cloud_stats(shp, collection, start_date, end_date, sh_config, tmp_folder)

        date_ranges = split_date_range_by_freq(start_date, end_date, time_period)

        # get a timezone to make dates comparable
        tz = pd.to_datetime(js[0]['data'][0]['interval']['from']).tz
        # extract list of best dates
        dates = {}
        for date_range in date_ranges:
            date_start = pd.to_datetime(date_range[0]).tz_localize(tz)
            date_end = pd.to_datetime(date_range[1]).tz_localize(tz)
            # get the dates between the range
            js_filtered = [day for day in js[0]['data'] if pd.to_datetime(day['interval']['from']) >= date_start and pd.to_datetime(day['interval']['from']) <= date_end]
            # merge noData and cloudy pixels
            vals = [calc_sh_cloud_val(day['outputs']['data']['bands']['B0']['stats']['mean'], day['outputs']['data']['bands']['B0']['stats']['noDataCount'], day['outputs']['data']['bands']['B0']['stats']['sampleCount']) for day in js_filtered]
            if len(vals) == 0:
                continue
            id = np.argmin(vals)
            # get date
            date = pd.to_datetime(js_filtered[id]['interval']['from']).date().strftime('%Y-%m-%d')
            dates[date] = vals[id]

        # save to file
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, 'w') as f:
            json.dump(dates, f)
    else:
        with open(fn, 'r') as f:
            dates = json.load(f)
    return dates

def sentinelhub_best_series(shp, collection, start_date, end_date, time_period, sh_config, tmp_folder, overwrite=False):
    """returns a series of dates with the best cloud coverage, if there are multiple."""
    # get sh stats
    stats = sentinelhub_cloud_stats(shp, collection, start_date, end_date, sh_config, tmp_folder)
    # see how many series are there
    sequences = []
    # check if there is more than one sequence
    for i in range(len(stats[0]['data'])):
        d = pd.to_datetime(stats[0]['data'][i]['interval']['from']).tz_convert(None)
        found = False
        for j in range(len(sequences)):
            if d - pd.to_datetime(sequences[j][-1]) in [pd.Timedelta(days=time_period*i) for i in range(1,8)]:
                sequences[j].append(d)
                found = True
                break
        if not found:
            sequences.append([d])
    # transform to string
    sequences = [[d.strftime('%Y-%m-%d') for d in seq] for seq in sequences]
    # switch to artificial sequences, in case of missing dates
    for i in range(len(sequences)):
        # create the first day in the sequence, in case it is missing
        date_range = pd.date_range(end=start_date, start=sequences[i][0], freq=f'-{time_period}D', inclusive='right')
        if len(date_range) > 0:
            seq_first_date = date_range[-1]
        else:
            seq_first_date = sequences[i][0]
        sequences[i] = [d1 for d1,d2 in split_date_range_by_freq(seq_first_date, end_date, time_period)]

    results = []
    for seq in sequences:
        seq_stats = []
        for date in seq:
            # extract from stats
            js_filtered = [day for day in stats[0]['data'] if pd.to_datetime(day['interval']['from']).tz_convert(None).strftime('%Y-%m-%d') == date]
            if len(js_filtered) == 0:
                seq_stats.append(100) # full no data
                continue
            elif len(js_filtered) > 1:
                raise ValueError("More than one date found.")
            # merge noData and cloudy pixels
            vals = [calc_sh_cloud_val(js_filtered[0]['outputs']['data']['bands']['B0']['stats']['mean'],
                                    js_filtered[0]['outputs']['data']['bands']['B0']['stats']['noDataCount'],
                                    js_filtered[0]['outputs']['data']['bands']['B0']['stats']['sampleCount'])]
            seq_stats.append(vals[0])
        # calculate the mean of the stats to compare the sequences
        res = np.mean(seq_stats)
        results.append(res)

    # select the best dates and return them
    idx = 0
    if len(results) > 1:
        idx = np.argmin(results)

    return sequences[idx]

def download_sentinelhub_dates(shp, collection, resolution, dates, bands, tmp_folder, sh_config):
    if 'landsat-8-l2' == collection:
        evalscript=f"""
            //VERSION=3

            function setup() {{
                return {{
                    mosaicking: Mosaicking.SIMPLE,
                    input: [{{
                        bands: {bands},
                        units: {["DN" if b == 'BQA' else "REFLECTANCE" for b in bands]}
                    }}],
                    output: {{
                        bands: {len(bands)},
                        sampleType: "UINT16"
                    }}
                }};
            }}

            function evaluatePixel(sample) {{
                return [{', '.join([f'(sample.{band} + 0.2) / 0.0000275' if band != 'BQA' else f'sample.{band}' for band in bands])}];
            }}
            """
        data_collection=sh.DataCollection.LANDSAT_OT_L2
        other_args = {}

    elif 'sentinel-2-l2a' == collection:
        evalscript = f"""
            //VERSION=3

            function setup() {{
                return {{
                    input: [{{
                        bands: {bands},
                        units: "DN"
                    }}],
                    mosaicking: "SIMPLE",
                    output: {{
                        bands: {len(bands)},
                        sampleType: "UINT16"
                    }}
                }};
            }}

            function evaluatePixel(sample) {{
                return [{', '.join([f'sample.{band}' for band in bands])}];
            }}
            """

        data_collection=sh.DataCollection.SENTINEL2_L2A
        other_args = {"processing" : {'harmonizeValues':"true"}}
    else:
        raise ValueError(f"Collection {collection} not supported for sentinelhub.")

    # download tif files
    tifs = []
    for date in dates:
        end_date = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        time_interval = (date, end_date)

        bbox = sh.BBox(bbox=list(shp.total_bounds), crs=sh.CRS(shp.crs.to_epsg()))
        size = sh.bbox_to_dimensions(bbox, resolution=resolution)

        out_folder = os.path.join(tmp_folder, 'sentinelhub', f'{shp["GEOID_PID"].values[0]}_{date}_{end_date}')

        request = sh.SentinelHubRequest(
            data_folder=out_folder,
            evalscript=evalscript,
            input_data=[
                sh.SentinelHubRequest.input_data(
                    data_collection=data_collection,
                    time_interval=time_interval,
                    mosaicking_order='leastCC',
                    # downsampling='BICUBIC',
                    other_args = other_args
                )
            ],
            responses=[
                sh.SentinelHubRequest.output_response('default', sh.MimeType.TIFF),
                # sh.SentinelHubRequest.output_response('userdata', sh.MimeType.JSON)
            ],
            bbox=bbox,
            size=size,
            config=sh_config
        )

        request.save_data()
        # read tif file
        fn = os.path.join(out_folder, request.get_filename_list()[0])
        da = rxr.open_rasterio(fn)
        da = da.expand_dims(time=pd.to_datetime([date]))
        tifs.append(da)

    # merge data into one xarray dataset
    data = xr.concat(tifs, dim='time')
    data = data.sortby('time')
    # to dataset use bands as variables
    data = data.to_dataset(dim='band')
    data = data.rename({i: b for i, b in zip(data.data_vars, bands)})
    return data

def download_ogs_sentinelhub(shp, collection, resolution, dates, bands, tmp_folder, overwrite=False):
    # compute nr pixels based on resolution
    bbox = sh.BBox(bbox=list(shp.total_bounds), crs=shp.crs)
    nr_pixels = sh.bbox_to_dimensions(bbox, resolution=resolution)
    bbox = list(shp.total_bounds) # take list to generate string

    def construct_url(args):
        conf_id = os.environ.get('OGS_SH_INSTANCE_ID')
        url = f"https://sh.dataspace.copernicus.eu/ogc/wcs/{conf_id}?"
        for k,v in args.items():
            url += f"{k.upper()}={v}&"
        return url
    
    tifs = []
    for date in dates:
        start_date = date
        end_date = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        file_name = f'{shp["GEOID_PID"].values[0]}_{date}_{end_date}.tif'

        if os.path.exists(os.path.join(tmp_folder, 'ogs', shp["GEOID_PID"].values[0], file_name)) and not overwrite:
            continue

        # download items https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/OGC.html
        args = dict(service='WFS',
                    version='1.0.0',
                    request='GetCoverage',
                    format='GeoTIFF',
                    coverage='COSTUM', # script id already created on the dashboard
                    bbox=f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                    maxcc=100,
                    priority='leastCC',
                    time=f"{start_date}/{end_date}",
                    crs=f"EPSG:{shp.crs.to_epsg()}",
                    response_crs=f"EPSG:{shp.crs.to_epsg()}",
                    width=nr_pixels[0],
                    height=nr_pixels[1],
                    downsampling='BICUBIC',
                    resolution=resolution)

        fn = os.path.join(tmp_folder, file_name)
        url = construct_url(args)
        urllib.request.urlretrieve(url, fn)
        da = rxr.open_rasterio(fn)
        da = da.expand_dims(time=pd.to_datetime([date]))
        tifs.append(da)

    # merge data into one xarray dataset
    data = xr.concat(tifs, dim='time')
    data = data.sortby('time')
    # to dataset use bands as variables
    data = data.to_dataset(dim='band')
    data = data.rename({i: b for i, b in zip(data.data_vars, bands)})

    return data

def download_sentinelhub(shp, collection, resolution, start_date, end_date, bands, tmp_folder, filter=None, time_period=None, overwrite=False, use_ogs=False):
    sh_config = sh.SHConfig()
    sh_config.sh_client_id = os.environ.get('SH_CLIENT_ID')
    sh_config.sh_client_secret = os.environ.get('SH_CLIENT_SECRET')
    sh_config.sh_base_url = 'https://services-uswest2.sentinel-hub.com'

    if collection == 'landsat-8-l2':
        # download
        dates = sentinelhub_best_series(shp, collection, start_date, end_date, time_period, sh_config, tmp_folder, overwrite=overwrite)
        data = download_sentinelhub_dates(shp, collection, resolution, dates, bands, tmp_folder, sh_config)
    elif collection == 'sentinel-2-l2a':
        if time_period is None:
            raise ValueError("Time period needs to be specified for sentinel-2-l2a.")
        dates = sentinelhub_best_days(shp, collection, start_date, end_date, time_period, sh_config, tmp_folder, overwrite=overwrite)
        if use_ogs:
            data = download_ogs_sentinelhub(shp, collection, resolution, dates.keys(), bands, tmp_folder, overwrite=overwrite)
        else:
            data = download_sentinelhub_dates(shp, collection, resolution, dates.keys(), bands, tmp_folder, sh_config)
    else:
        raise NotImplementedError(f"Collection {collection} not supported for sentinelhub.")
    return data

def filter_doubled_processing_baseline_by_name(items):
    """use the name of the item to filter the processing baseline,
    see https://sentinels.copernicus.eu/en/web/sentinel/user-guides/sentinel-2-msi/naming-convention
    """
    filtered_items = {}

    # re like "N0301" or "N0500"
    pattern = re.compile(r'(.*_N)(\d{4})(_.*)')

    for item in items:
        match = pattern.match(item['id'])
        if match:
            base_id = match.group(1) + match.group(3)
            proc_date = pd.to_datetime(base_id.split("_")[-1].split(".")[0])
            base_id = "_".join(base_id.split("_")[:-1])
            num = int(match.group(2))
            # If the base_id is not in the dictionary 
            # or if the current num is larger (not when it is 99.99, which seems to be a fill value)
            # or if the num is the same but the processing date is newer
            # update the entry
            if base_id not in filtered_items or (filtered_items[base_id]['num'] < num < 9999 or filtered_items[base_id]['num'] == 9999 != num)\
                or (proc_date > filtered_items[base_id]['proc_date'] and num == filtered_items[base_id]['num']):
                filtered_items[base_id] = {'item': item, 'num': num, 'proc_date': proc_date}
        else:
            raise ValueError(f"Could not extract processing baseline from {item['id']}.")

    # Extract the filtered list
    result = [entry['item'] for entry in filtered_items.values()]

    return result

def download_cdse_s3(shp, collection, resolution, start_date, end_date, bands, filter, tmp_folder, use_s3fs=False, stats_band=None, time_range=None, mosaic_args=None):
    # adds id to the time, this was change in terragon0.2.0:
    # # skip items which were not found
    # time_data, times, ids = map(list, zip(*[(ds, item["properties"]["datetime"], item["id"]) for ds, item in zip(time_data, items) if ds is not None]))

    # time_data = self._align_coords(time_data, shp, resampling)

    # # add time coords (would have been removed by reproject_match in _align_coords)
    # time_data = [ds.assign_coords(time=("time", pd.to_datetime([time]).tz_convert(None))) for ds, time in zip(time_data, times)]
    # time_data = [ds.assign_coords(id=("time", [id])) for ds, id in zip(time_data, ids)]

    credentials = {'aws_access_key_id': os.environ.get("S3_ACCESS_KEY"), 'aws_secret_access_key': os.environ.get("S3_SECRET_KEY")}
    tg = terragon.init('cdse', credentials=credentials)
    if time_range is None:
        items = tg.search(shp=shp, collection=collection, bands=bands, start_date=start_date, end_date=end_date, resolution=resolution, filter=filter, num_workers=2)
        items = filter_doubled_processing_baseline_by_name(items)
        data = tg.download(items)
    else: # use stats for time range
        # get the dates
        dates = s3_filter_by_stats(shp, collection, resolution, start_date, end_date, stats_band, filter, time_range, tmp_folder, use_s3fs, mosaic_args, tg)
        # accumulate the data
        items = []
        for date in dates:
            start_date = date
            end_date = start_date + 'T23:59:59.999'
            its = tg.search(shp=shp, collection=collection, bands=bands, start_date=start_date, end_date=end_date, resolution=resolution, filter=filter)
            its = filter_doubled_processing_baseline_by_name(its)
            items.extend(its)
        data = tg.download(items)
    return data

def s3_filter_by_stats(shp, collection, resolution, start_date, end_date, band, filter, time_range, tmp_folder, use_s3fs, mosaic_args:dict, tg):
    """create the stats of a band for a time range
    mosaic_args: dict with the following keys: mask_u_func (to create mask from band), and parameter for mosaic: interval, func, invalid_values, band"""
    from utils import mosaic
    date_ranges = split_date_range_by_freq(start_date, end_date, time_range)
    dict_mosaic = mosaic_args.copy()
    u_func = dict_mosaic.pop('mask_u_func')
    # file handling
    fn_dates = os.path.join(tmp_folder, f"{collection}_{time_range}_{shp['GEOID_PID'].values[0]}_{start_date}_{end_date}.json")
    time_steps = {}
    if not os.path.exists(fn_dates):
        for date_range in date_ranges:
            s_date = date_range[0]
            e_date = date_range[1]
            items = tg.search(shp=shp, collection=collection, bands=[band], start_date=s_date, end_date=e_date, resolution=resolution, filter=filter, num_workers=2)
            if items is None:
                warnings.warn(f"No items found for {s_date} - {e_date}.")
                continue
            items = filter_doubled_processing_baseline_by_name(items)
            ds = tg.download(items)
            # build day mosaic with best pixels
            ds['mask'] = xr.apply_ufunc(u_func, ds[band], vectorize=True, dask='parallelized', output_dtypes=[bool])
            ds = mosaic(ds, band='mask', **dict_mosaic)
            # get the stats
            c_stats = ds['mask'].mean(dim=['x', 'y']).compute()
            nan_stats = (ds[band].isnull() | ds[band].isin(dict_mosaic['invalid_values'])).mean(dim=['x', 'y']).compute()
            stats = c_stats + nan_stats
            # create a dict with time step and stats
            dict_stats = {date: stat for date, stat in zip(ds.time.dt.strftime('%Y-%m-%d').values, stats.values)}
            time_steps.update({f'{s_date}/{e_date}': dict_stats})
        # save all time steps
        os.makedirs(os.path.dirname(fn_dates), exist_ok=True)
        with open(fn_dates, 'w') as f:
            json.dump(time_steps, f)
    else:
        with open(fn_dates, 'r') as f:
            time_steps = json.load(f)

    # select min for each date range
    dates = [min(date_range, key=date_range.get) for date_range in time_steps.values() ]

    return dates
