# %%
import xarray as xr
import rioxarray as rxr
import rasterio
from rasterio.enums import Resampling
import numpy as np
import geopandas as gpd
from shapely.geometry import box, mapping
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import pandas as pd
import os
from tqdm import tqdm
import wandb
import skimage as ski
import rootutils
import os
from evolution_utils_pixel import *
from gridding import *
import itertools
import matplotlib
matplotlib.use('agg')
# %%
edge_size= 400 # 400*30m=12km

home_folder = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
data_dir = home_folder / "data/supp_data/"
result_folder = os.path.join(home_folder,  'results/minicube_generation/')
os.makedirs(result_folder, exist_ok=True)
# %%
hparams = dict(
    edge_size=edge_size,
    seed=42,
    score_names=['allocation', 'patch_overlap', 'nr_patches', 'crop'],
    fitness_weights=(-2, +5, +1, -4,),
    algorithm='eaSimple',
    n_pop=1000,   # population size
    n_gen=300,      # nr generations
    cxpb=0.2,       # cross-over prob
    mutpb=0.4,      # mutation prob
    noisepb=.9,  # prob to add noise or to remove / add a patch
    noise_nr_perc=.5, # how many patches to add noise to in percent
    noise_range=edge_size//2,
    noise_decay=60, # decay of the noise every epoch
    mutate='cxTwoPoint_mod',
    tournsize=3,    # tournament size when selecting individuals
    log_stats = dict(
        avg=np.mean,
        std=np.std,
        min=np.min,
        max=np.max,
    ),
    save_best=1, # save the best x individuals
    multi_process=int(os.cpu_count()-2), # nr cpus to use
    log_freq=25,    # log images/minicubes every x generations    
    verbose=True,   # print information on each generation
    early_stop=75,  # stop after x generations without improvement
    monitor='min',  # value to monitor
    online_log=False, # log to wandb
    tags=[''], # tags for wandb
    log_dir='results/minicube_generation/',
)
# %%
fn = data_dir / 'counties.geojson'
gdf = gpd.read_file(fn)
min, max = gdf['geom_sqkm'].min(), gdf['geom_sqkm'].max()
gdf = gdf[gdf['GEOID'].isin(['13155'])] # use this line to run a specific county, or comment for all

seeds = [42, 321, 123]
user = os.environ.get('WANDB_USER')
project = 'minicube-creation'
for idx in tqdm(gdf.index, total=len(gdf.index)):
    row = gdf.loc[[idx]]
    geoid = row['GEOID'].values[0]
    for s in seeds:
        _hparams = hparams.copy()
        if hparams['online_log']:
            api = wandb.Api()
            runs = api.runs(f"{user}/{project}", filters={"tags": _hparams['tags'][0], "config.geoid": geoid, "config.seed": s})
            if len(runs) > 0:
                continue
        print('run:', geoid, 'seed:', s)
        # update params
        n_pop = scale_linear(row['geom_sqkm'].values[0], min, max, 300, 3500)
        n_gen  = scale_linear(row['geom_sqkm'].values[0], min, max, 300, 1200)
        noise_decay = scale_linear(n_gen, 300, 1200, 55, 210)
        tournsize = scale_linear(row['geom_sqkm'].values[0], min, max, 3, 12)

        _hparams.update({'n_pop': n_pop, 'n_gen': n_gen, 'noise_decay': noise_decay, 'seed': s, 'tournsize': tournsize})
        gp_co(row, _hparams, home_folder, use_utm=True)
