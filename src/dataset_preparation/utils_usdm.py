# %%
import glob
import os
import warnings
import zipfile

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from pyproj import CRS
from rasterio.features import rasterize
from tqdm import tqdm

# ignore deprication warning from Shapley (done by rasterio)
warnings.simplefilter("ignore", category=DeprecationWarning)

def download_dataset_USDM(start_year, end_year, fo='data/USDM/'):
    # download USDM index
    os.makedirs(fo, exist_ok=True)
    for i in range(start_year, end_year+1):
        filename = '{}_USDM_M.zip'.format(i)
        print('download: ', filename)
        url = 'https://droughtmonitor.unl.edu/data/shapefiles_m/' + filename
        # check if file already exists
        if not os.path.exists(fo + filename):
            os.system('wget -O {} {}'.format(fo + filename, url))
            zipfile.ZipFile(fo + filename).extractall(fo)
            os.remove(fo + filename)

def rasterize_usdm_file(usdm_path, reference):
    """Rasterize the USDM file to the reference, see rasterize_usdm for more information."""
    gdf = gpd.read_file(usdm_path)
    return rasterize_usdm(gdf, reference)

def rasterize_usdm(gdf, reference):
    """Rasterize the USDM geopandas dataframe to the reference.

    The reference can be a path to a raster, a rasterio dataset or a tuple of shape and transform.
    """
    if isinstance(reference, str):
        with rio.open(reference) as raster:
            out_profile = raster.profile.copy()
            out_shape=raster.shape
            transform=raster.transform
    elif isinstance(reference, rio.io.DatasetReader):
        out_profile = reference.profile.copy()
        out_shape = reference.shape
        transform = reference.transform
    elif isinstance(reference, (list, tuple)):
        out_shape, transform = reference
        out_profile = None
    else:
        raise ValueError('reference should be a path to a raster, a rasterio dataset or a tuple of shape and transform')

    mask = None
    masks = []
    for _, r in gdf.iterrows():
        r_mask = rasterize(r, out_shape=out_shape, transform=transform)
        r_mask *= (r.DM+1)
        masks.append(r_mask)
    mask = np.sum(masks, axis=0)
    return mask, out_profile

def rasterize_USA(tif_path, fo_data, fo_sv, fn_rastered, years):
    # rasterize it to the format of tif_path file (data will not be used only meta-data)

    usdms = glob.glob(fo_data+'/USDM_*.zip')
    for path in tqdm(usdms):
        fn_date = os.path.basename(path).split('_')[1]
        fn_year = fn_date[:4]
        if int(fn_year) not in years:
            continue
        data, profile = rasterize_usdm_file(path, tif_path)

        # save with smaller datatype than float
        profile.update(nodata=np.iinfo(np.int8).min, dtype=np.int8, driver='GTiff')

        os.makedirs(fo_sv + str(fn_year), exist_ok=True)
        with rio.open(fo_sv + str(fn_year) + '/' + fn_rastered.format(fn_date), 'w', **profile) as dst:
            dst.write(data, indexes=1)

def create_usdm_frequency_map(fo_data, fo_sv, years):
    os.makedirs(fo_sv, exist_ok=True)
    usdm_fns = glob.glob(fo_data + '/**/*.tif')
    res = None
    for fn in tqdm(usdm_fns):
        fn_date = os.path.basename(fn).split('_')[1]
        fn_year = fn_date[:4]
        if int(fn_year) not in years:
            continue
        ds = rxr.open_rasterio(fn)
        # create binary maps
        binary_bands = [xr.where(ds == i, 1, 0).astype('uint32') for i in range(1,6)]
        if res is None:
            res = xr.concat(binary_bands, dim='band')
        else:
            res += xr.concat(binary_bands, dim='band')
    res.coords['band'] = np.array(range(1,6))
    res = res.to_dataset(name='freq')
    res.to_netcdf(fo_sv + f'usdm_freq_{years[0]}_{years[-1]}.nc')

def load_usdm_patch(region, fo, time_slice, ref_ds):
    """Load the USDM patch for the given region and time slice.

    The region should be in the same crs as the ref_ds. ref_ds needs to have rioxarray properties.
    """
    # load the data
    patches = []
    start_year, end_year = int(time_slice[0].split('-')[0]), int(time_slice[1].split('-')[0])
    start_date, end_date = int(time_slice[0].replace('-', '')), int(time_slice[1].replace('-', ''))
    for year in range(start_year, end_year+1):
        fns = os.path.join(fo, f'USDM_{year}*_M.zip')
        for fn in glob.glob(fns):
            date = int(fn[-14:-6])
            date_str = str(date)[:4] + '-' + str(date)[4:6] + '-' + str(date)[6:8]
            if date >= start_date and date <= end_date:
                # load shape and rasterize
                gdf = gpd.read_file(fn)
                if not gdf.crs:
                    gdf.set_crs("epsg:4326", inplace=True)
                gdf.to_crs(region.crs, inplace=True)
                gdf = gdf.clip(region.geometry) # clip gpd df to region
                raster, _ = rasterize_usdm(gdf, (ref_ds.rio.shape, ref_ds.rio.transform())) # raster with reference shape
                # use the ref dataset to create a xarray
                raster = xr.DataArray(raster, dims=('y', 'x'), coords={'y': ref_ds.y, 'x': ref_ds.x})
                raster = raster.to_dataset(name='usdm')
                raster = raster.assign_coords(time=pd.to_datetime([date_str]))
                
                raster.rio.write_crs(ref_ds.rio.crs, inplace=True)
                patches.append(raster)
    patch = xr.concat(patches, dim='time').sortby('time')
    patch.attrs['epsg'] = ref_ds.rio.crs.to_epsg()
    return patch

if __name__ == '__main__':
    # download_dataset_USDM(2008, 2022, fo='data/USDM/')
    # create a rasterized summary of the USDM data
    import rootutils
    from utils_daymet import read_dataset_file_daymet
    fo = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    fn = str(fo / 'data/daymet/daymet_v4_daily_na_prcp_1980.tif')
    # create a correct raster for daymet as a template
    crs_daymet = CRS.from_proj4('+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs')
    df_us_county = gpd.read_file(fo / "data/geometry/tl_2018_us_state.shp", crs="epsg:4269")
    df_us_county = df_us_county[~df_us_county['STATEFP'].isin(["02", "15", "60", "66", "69", "72", "78"])]
    df_us_county = df_us_county.to_crs("epsg:4326")
    if not os.path.exists(fn):
        ds = read_dataset_file_daymet(1980, None, 'prcp', 'daymet_v4_daily_na_{}_{}.nc', os.path.join(fo, 'data/daymet/'), crs_daymet, None, chunks='auto')
        ds = ds.isel(time=0)
        ds = ds.rio.reproject("epsg:4326")
        ds = ds.rio.clip(df_us_county.geometry)
        ds.rio.to_raster(fn)
    # rasterize with the daymet template
    rasterize_USA(tif_path=fn, fo_data=os.path.join(fo, 'data/USDM'), fo_sv = os.path.join(fo, 'data/USDM/rastered_USDM/'), fn_rastered = 'usdm_{}.tif', years=range(2018,2023))

    # create frequencies
    create_usdm_frequency_map(fo_data=os.path.join(fo, 'data/USDM/rastered_USDM/'), fo_sv = os.path.join(fo, 'data/USDM/'), years=range(2018,2023))
