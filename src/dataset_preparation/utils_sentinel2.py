import datetime
import xarray as xr

def align_processing_version(ds, geoid_pid):
    if any([(int(str(id).split('_')[3][1:]) < 300) or (9999 == int(str(id).split('_')[3][1:])) for id in ds.id.values]):
        for t in range(len(ds.time)):
            da = ds.isel(time=t)
            version = int(str(da.id.values).split('_')[3][1:])
            if version < 300 or 9999 == version:
                # is 2.x adjust to 05.xx
                da = harmonize_s2_to_new(da)
                ds[{'time': t}] = da
                with open('harmonized.txt', 'a') as f:
                    f.write(f"{geoid_pid}: {da.time.dt.strftime('%Y-%m-%d').values} {id} to 05.xx\n")
    return ds

def harmonize_s2_to_new(da):
    bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12",]
    offset = 1000
    to_process = list(set(bands) & set(da.data_vars.keys()))
    for band in to_process:
        # add offset and clip to make sure there is no overflow
        mask = da[band] != 0
        da[band] = da[band].clip(0, 65535 - offset) + offset
        da[band] = da[band].where(mask, 0)
    return da