from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from sklearn.model_selection import KFold
import yaml
import xarray as xr
import os

import torch
import pandas as pd
import numpy as np
import os
import inspect
from tqdm import tqdm
from src.datasets.yield_dataset import load_yield_labels, str_to_day_of_year
from src.datasets.utils import resample

def preprocess_dataset_to_df(dataset, data_dir, save:str=None, **kwargs):
    """preprocess the whole dataset in case it was not processed before, otherwise load it.
    return list of dfs, one for each modality."""
    # read parameters to string
    transform = dataset.x_transform
    if not isinstance(transform, list):
        transform = [transform]

    dfs = {}
    bands = dataset.bands
    for m in bands:
        if save:
            fn = os.path.join(data_dir, m, save)
            if os.path.exists(fn):
                dfs[m] = pd.read_csv(fn)
                continue
        
        print(f'Preprocessing dataset...')

        # load the dataset
        dataset.bands = {m:bands[m]} # select data for one modality
        from datamodule import collate_yield_dl
        data_loader = DataLoader(dataset=dataset, collate_fn=collate_yield_dl, **kwargs)
        # extract data
        xs, geoids, times = [], [], []
        for batch in tqdm(data_loader):
            xs.extend(batch['x'])
            geoids.extend([m['geoid'] for m in batch['meta']])
            # use first time steps if there are multiple patches
            times.extend([m['time'] if len(m['time']) <= 1 else m['time'][0] for m in batch['meta']])

        # flatten data
        flattened_data = []
        for h in range(len(xs)): # loop over samples
            for j in range(xs[h].shape[0]): # loop over time
                row = {'geoid': f"{geoids[h]}"}
                row.update({'time': times[h][j].item() if len(times[h].shape) <= 1 else times[h][0,j].item()})
                
                if len(xs[0].shape) == 2: # contains: time, band
                    row.update({f'band_{bands[m][k]}': xs[h][j, k].item() for k in range(xs[h].shape[1])}) # loop over bands
                elif len(xs[0].shape) == 3: # contains additional information: time, band, values
                    row.update({f'band_{bands[m][k]}_{l}': xs[h][j, k, l].item() for k in range(xs[h].shape[1]) for l in range(xs[h].shape[-1])})
                else:
                    raise ValueError('Data shape not supported')
                flattened_data.append(row)

        df = pd.DataFrame(flattened_data)
        df['time'] = pd.to_datetime(df['time'], unit='ns')
        dfs[m] = df

        if save:# save it to csv
            dfs[m].to_csv(fn, index=False)

    return dfs

class TabularYieldDataset(Dataset):
    """Load the preprocessed dataset for the yield prediction task."""
    def __init__(self, data_dir: str, file_name:str, x_transform=None, y_transform=None, bands:dict=None,
                crops:list=None, time_range:tuple=(1, 365),years:list=None,geoids:list=None,
                resampling:list=None,handle_nans='zeros',**kwargs) -> None:
        self.data_dir = data_dir
        self.file_name = file_name
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.handle_nans = handle_nans

        if not isinstance(years, list):
            years = list(years)
        if not isinstance(geoids, list):
            geoids = list(geoids)
        if isinstance(crops, str):
            crops = [crops]
        self.labels = load_yield_labels(data_dir, years, crops, geoids)
        self.data = self.load_data(bands, geoids, years, time_range, resampling)

    def __len__(self) -> int:
        return len(self.labels.index)

    def load_data(self, bands, geoids, years, time_range, resampling):
        master = None
        dfs = []
        for m in bands:
            df = pd.read_csv(os.path.join(self.data_dir, m, f'{self.file_name}.csv'), dtype={'geoid': str}, 
                            usecols=['geoid', 'time'] + [f'band_{b}' for b in bands[m]], parse_dates=['time'])
            if geoids and len(geoids) > 0:
                df = df[df['geoid'].isin(geoids)]

            # filter by time
            if years and len(years) > 0:
                start, end = str_to_day_of_year(time_range[0]), str_to_day_of_year(time_range[1]) 
                for year in years:
                    # prep temporal selection
                    if start < end:
                        days = list(range(start, end+1))
                    else:
                        days = list(range(start, 366))
                        period2 = list(range(1, end+1))
                        days.extend(period2)
                df = df[df['time'].dt.dayofyear.isin(days) & df['time'].dt.year.isin(years)]

            # resample time
            if resampling:
                # make sure time is timestamp
                df['time'] = pd.to_datetime(df['time'])
                # use the xarray function to resample
                ds = df.set_index(['time', 'geoid']).to_xarray()
                res_dict = None
                if m in resampling:
                    res_dict = resampling[m]
                elif 'default' in resampling:
                    res_dict = resampling['default']
                ds = resample(ds, res_dict, master=master)
                df = ds.to_dataframe().reset_index()
            dfs.append(df)
            if master is None:
                master = ds
        if any([len(dfs[0]) != len(dfs[i]) for i in range(1, len(dfs))]):
            raise ValueError('Different lengths of modalities, resampling should be applied.')
        dfs = [df.set_index(['geoid', 'time']) for df in dfs]
        comb_df = pd.concat(dfs, axis=1).reset_index()
        return comb_df

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        y = self.labels.iloc[idx]['yield']
        year = self.labels.iloc[idx]['year']
        geoid = self.labels.iloc[idx]['geoid']

        # load the time and geoid
        df = self.data[self.data['geoid'] == geoid]
        df = df[df['time'].dt.year == year]
        df.sort_values(by='time', inplace=True)

        x = df.drop(columns=['geoid', 'time']).values
        x,y = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)

        if self.x_transform:
            x = self.x_transform(x)
        if self.y_transform:
            y = self.y_transform(y)

        if self.handle_nans == 'zeros':
            x = np.where(np.isnan(x), 0, x)
            y = np.where(np.isnan(y), 0, y)

        return {'x':x, 'y': y}

if __name__ == "__main__":

    # create tabular data
    from datamodule import MinicubeDataset, YieldDataset
    from transforms import Mean, Hist, Sample, PermuteDims
    import geopandas as gpd
    import itertools
    # load final geoids
    geoids = list(itertools.chain.from_iterable(yaml.safe_load(open(os.path.join(home_folder, 'configs/data/geoids_split.yaml')))['split']['ratio']))

    transform = [
        [transforms.Compose([Mean(dim=(-2,-1,-5))]), 'Mean', ],
        # [transforms.Compose([Hist(bins=50, range=(0, 1), dim=(-2,-1,-5), normalize=True)]), 'Hist'],
        # [transforms.Compose([PermuteDims(dims=(1,2,0,3,4)), Sample(n=10, start_dim=-3, include_only_valid=True, invalid_value=None)]), 'Sample_n=10'],
    ]

    for t,n in transform:
        dic = {'data_dir': os.path.join(home_folder, 'data/CropClimateX/'), 
            'x_transform': t,
            'bands': {
                'modis': [f'sur_refl_b0{i}' for i in range(1,5)] + ['sur_refl_b06'],
                # 'landsat8': ['B02', 'B03', 'B04', 'B05', 'B06'],
                # 'sen2': ["B02","B03","B04","B06","B08","B11"],
                # 'lai': ["Lai_500m", "Fpar_500m"],
                # 'lst': ['LST_Day_1km', 'LST_Night_1km'],
                # 'daymet': ['tmax', 'tmin', 'prcp', 'vp', 'srad', 'hcw'],
                # 'usdm': ['usdm'],
                },

            'geoids': geoids,
            'resampling': {
                # 'cdl':
                    # {'spatial': {'method': 'nearest/mode', 'size': 12},} # for daymet/usdm/lst
                    # {'spatial': {'method': 'nearest/mode', 'size': 24},} # for modis/lai
                    # {'spatial': {'method': 'nearest/mode', 'size': 600},} # for sen2
            },
            'return_meta_data':True,
            'time_range': ('01-01', '12-31'),
            'crops': 'corn', # does not matter
            'ignore_label_registry': True, # use "fake" data
            'use_county_mask': True,
            'cdl_30m': False, # for landsat8/sen2/dem
        }

        ds = YieldDataset(**dic)
        df = preprocess_dataset_to_df(ds, data_dir=dic['data_dir'], save=f"{n}.csv", batch_size=32, num_workers=6)
        print(df)
