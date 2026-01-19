# %%
import random
import numpy as np
import geopandas as gpd
import rasterio
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
import math
import os
import wandb
from shapely.geometry import Polygon, box, mapping
import threading
from gridding import split_image_mask_to_polygons, create_patch_mask, pixels_idx_to_coord
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from pyproj import CRS, Transformer
from evolution_algorithms import EaSimple
from deap import base, creator, tools, algorithms
import skimage as ski
import pandas as pd
import multiprocessing
import json

def scale_linear(n, min_n, max_n, min_o, max_o):
    """Scale a number from one range to another."""
    return math.ceil((n - min_n) / (max_n - min_n) * (max_o - min_o) + min_o)

def get_init_nr_patches(co_mask, edge_size):
    nr_co_pixels = np.sum(co_mask > 0)
    patch_size = edge_size**2 
    return math.ceil(nr_co_pixels / patch_size)

def compute_metrics(patches, co_mask, freq, nr_baseline_patches):
    """compute the metrics for a set of patches."""
    # Create an empty binary mask and add rectangles to the mask
    patch_mask = create_patch_mask(patches, co_mask.shape)

    # obj 1 IoU county and total area (area outside + coverage)
    # nr pixels of patches in county / nr pixels of patches
    allocation = np.sum(patch_mask[co_mask] > 0) / np.sum((patch_mask > 0))
    # obj 2 IoU patches (overlap)
    # nr pixels in overlap / nr pixels in patches
    overlap = np.sum(patch_mask > 1) / np.sum(patch_mask > 0)
    # obj 3 low nr patches
    # nr patches / nr patches in baseline
    nr_patches = len(patches) / nr_baseline_patches
    # obj 4 - as much crop as possible
    # crops in patches / crops in county
    crop = np.sum(freq[patch_mask > 0]) / np.sum(freq[co_mask])

    # weights should be: -, +, +, - to minimize this
    return allocation, overlap, nr_patches, crop,

def compute_metric(i, patches, co_mask, freq):
    """compute the metric for a single patch."""
    patch = patches[i]
    singular_patch_mask = create_patch_mask([patch], co_mask.shape)
    patch_mask = create_patch_mask(patches, co_mask.shape)

    allocation = np.sum(singular_patch_mask[co_mask] > 0) / np.sum((singular_patch_mask > 0))
    crop = np.sum(freq[singular_patch_mask > 0]) / np.sum(freq[co_mask])
    overlap = np.sum(patch_mask[singular_patch_mask > 0] > 1) / np.sum(singular_patch_mask > 0)

    return allocation, overlap, crop

def fitness(ind, weights, freq_raster, co_mask, nr_baseline_patches):
    """wrapper to compute fitness together with weights"""
    scores = np.array(compute_metrics(ind, co_mask, freq_raster, nr_baseline_patches))
    return [np.sum(scores * weights)]

def create_patch_rand(bounds, edge_size):
    minx, miny = np.random.randint(0, bounds[0]-edge_size), np.random.randint(0, bounds[1]-edge_size)
    maxx, maxy = minx + edge_size, miny + edge_size
    rec = (minx, miny, maxx, maxy)
    return rec

def create_ind_rand(co_mask, bounds, edge_size, fluctation=2):
    nr_patches = get_init_nr_patches(co_mask, edge_size)
    low = nr_patches-fluctation if nr_patches-fluctation > 0 else 1
    nr = np.random.randint(low, nr_patches+fluctation)
    ind = [create_patch_rand(bounds, edge_size) for _ in range(nr)]
    return ind

def create_ind_fishnet(polys, edge_size, shift_grid=2):
    # add noise to the polygons
    start_pos = random.choice([0,1,2,3])
    polys = polys[start_pos]
    if shift_grid == 2:
        x_shift = np.random.randint(-edge_size, edge_size)
        y_shift = np.random.randint(-edge_size, edge_size)
        ind = [[poly[0] - x_shift, poly[1] - y_shift, poly[2] - x_shift, poly[3] - y_shift] for poly in polys[:int(len(polys)/2)]]
        for poly in polys[int(len(polys)/2):]:
            x_shift = np.random.randint(-edge_size, edge_size)
            y_shift = np.random.randint(-edge_size, edge_size)
            ind.append([poly[0]-x_shift, poly[1]-y_shift, poly[2]-x_shift, poly[3]-y_shift])
    elif shift_grid: # shift the whole grid
        x_shift = np.random.randint(-edge_size, edge_size)
        y_shift = np.random.randint(-edge_size, edge_size)
        ind = [[poly[0] - x_shift, poly[1] - y_shift, poly[2] - x_shift, poly[3] - y_shift] for poly in polys]
    else: # shift each cell
        ind = []
        for poly in polys:
            x_shift = np.random.randint(-edge_size, edge_size)
            y_shift = np.random.randint(-edge_size, edge_size)
            ind.append([poly[0]-x_shift, poly[1]-y_shift, poly[2]-x_shift, poly[3]-y_shift])
    return ind

def create_individual_rand(icls, co_mask, bounds, edge_size, fluctation=2):
    """wrapper for deap."""
    return icls(create_ind_rand(co_mask, bounds, edge_size, fluctation=fluctation))

def create_individual_fishnet(icls, co_mask, polys, edge_size, fluctation=2):
    """wrapper for deap."""
    return icls(create_ind_fishnet(co_mask, polys, edge_size))

def add_or_remove_part(ind):
    if np.random.random() < 0.5 and len(ind) > 1:
        ind.pop(np.random.randint(0, len(ind)-1))
    else:
        ind.append(create_patch_rand(ind.bounds, ind.edge_size))
    return ind

def add_noise(ind, perc=.2, noise_range=10):
    """add integer noise in range noise_range to the perc randomly selected patches and 
    make sure that the patch stays within the bounds."""
    def add_noise_to_patch(patch, p_idx):
        noise_x = np.random.randint(-noise_range, noise_range)
        noise_y = np.random.randint(-noise_range, noise_range)
        if patch[p_idx][0] + noise_x < 0:
            noise_x = 0 - patch[p_idx][0]
        if patch.bounds[0] < patch[p_idx][2] + noise_x:
            new_noise_x = patch.bounds[0] - patch[p_idx][2]
            if noise_x > new_noise_x:
                noise_x = new_noise_x
        if patch[p_idx][1] + noise_y < 0:
            noise_y = 0 - patch[p_idx][1]
        if patch.bounds[1] < patch[p_idx][3] + noise_y:
            new_noise_y = patch.bounds[1] - patch[p_idx][3]
            if noise_y > new_noise_y:
                noise_y = new_noise_y
        new_patch = (
            max(0, patch[p_idx][0] + noise_x),
            max(0, patch[p_idx][1] + noise_y),
            min(patch.bounds[0], patch[p_idx][2] + noise_x),
            min(patch.bounds[1], patch[p_idx][3] + noise_y)
        )
        patch[p_idx] = new_patch
        return patch

    # add noise to perc patches
    nr = math.ceil(len(ind)*perc)
    if nr == 1:
        ind = add_noise_to_patch(ind, 0)
    elif nr > 1:
        for _ in range(nr):
            p_idx = np.random.randint(0, len(ind)-1)
            ind = add_noise_to_patch(ind, p_idx)

    return ind

def fake_mutate(ind, **kwargs):
    return ind,

def fake_mate(ind1, ind2, **kwargs):
    return ind1, ind2

def mutate(ind, pb, **kwargs):
    """mutate by adding noise or adding/removing a patch."""
    # add noise or add/remove patch
    if np.random.random() < pb:
        ind = add_noise(ind, **kwargs)
    else:
        ind = add_or_remove_part(ind)
    return ind,

def cxPoint(ind1, ind2):
    if np.random.random() < .5:
        ind1[np.random.randint(0, len(ind1)-1)] = ind2[np.random.randint(0, len(ind2)-1)]
    else:
        ind2[np.random.randint(0, len(ind2)-1)] = ind1[np.random.randint(0, len(ind1)-1)]
    return ind1, ind2

def cxTwoPoint_mod(ind1, ind2):
    # modification of the org function
    """Executes a two-point crossover on the input :term:`sequence`
    individuals. The two individuals are modified in place and both keep
    their original length.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    """
    size = min(len(ind1), len(ind2))
    if size > 2: # if size one do not cross
        cxpoint1 = np.random.randint(0, size)
        cxpoint2 = np.random.randint(1, size - 1)
        if cxpoint2 == cxpoint1:
            cxpoint2 += 1
        elif cxpoint2 < cxpoint1:
            # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        h_ind1 = ind1[cxpoint1:cxpoint2]
        ind1[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2]
        ind2[cxpoint1:cxpoint2] = h_ind1
        
    return ind1, ind2

def do_patches_overlap(box1, box2):
    """Check if two patches overlap by using projection to x and y axis."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    overlap_x = x1_max >= x2_min and x2_max >= x1_min
    overlap_y = y1_max >= y2_min and y2_max >= y1_min
    return overlap_x and overlap_y

def remove_overlapping_patches(ind1, idxs=None):
    """Remove overlapping patches from individual."""
    if idxs is None:
        idxs = (0, len(ind1))

    idx_to_rm = []
    for i in range(idxs[0], idxs[1]):
        for j in range(idxs[0]):
            if i != j and do_patches_overlap(ind1[i], ind1[j]):
                idx_to_rm.append(j)

        for j in range(idxs[1],len(ind1)):
            if i != j and do_patches_overlap(ind1[i], ind1[j]):
                idx_to_rm.append(j)

    for i in sorted(set(idx_to_rm), reverse=True):
        del ind1[i]

    return ind1

def cxTwoPoint_remove_overlap(ind1, ind2):
    """Two point crossover, with removing patches overlapping in child after crossover."""
    size = min(len(ind1), len(ind2))
    if size > 2: # if size two do not cross
        cxpoint1 = np.random.randint(0, size)
        cxpoint2 = np.random.randint(1, size - 1)
        if cxpoint2 == cxpoint1:
            cxpoint2 += 1
        elif cxpoint2 < cxpoint1:
            # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        h_ind1 = ind1[cxpoint1:cxpoint2]
        ind1[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2]
        ind2[cxpoint1:cxpoint2] = h_ind1

        # remove overlapping patches
        ind1 = remove_overlapping_patches(ind1, idxs=(cxpoint1, cxpoint2))
        ind2 = remove_overlapping_patches(ind2, idxs=(cxpoint1, cxpoint2))

    return ind1, ind2

def sort_patches(ind, co_mask, freq_raster, reverse=True):
    scores = [compute_metric(i, ind, co_mask, freq_raster) for i in range(len(ind))]
    ind[:] = [x for _,x in sorted(zip(scores,ind))] # sort by the scores
    return ind

def cxBest_remove_overlap(ind1, ind2, co_mask, freq_raster):
    """Best crossover, with removing patches overlapping in child after crossover."""
    ind1 = sort_patches(ind1, co_mask, freq_raster)
    ind2 = sort_patches(ind2, co_mask, freq_raster)

    size = min(len(ind1), len(ind2))
    if size > 2:
        cxPoint = np.random.randint(1, size - 1)

        h_ind1 = ind1[0:cxPoint]
        ind1[0:cxPoint] = ind2[0:cxPoint]
        ind2[0:cxPoint] = h_ind1

        # remove overlapping patches
        ind1 = remove_overlapping_patches(ind1, idxs=(0, cxPoint))
        ind2 = remove_overlapping_patches(ind2, idxs=(0, cxPoint))

    return ind1, ind2

def plot_county_and_patches(polys, shp, freq_raster=None):
    fig, ax = plt.subplots()
    if shp is not None:
        shp.plot(ax=ax, facecolor='whitesmoke', edgecolor='none')
        shp.plot(ax=ax, facecolor='none', edgecolor='black', alpha=.8)
    if polys is not None:
        polys.plot(ax=ax, facecolor='b', edgecolor='none', alpha=.05)
    if freq_raster is not None:
        cmap = plt.get_cmap('Greens')
        cmap.set_under((1, 1, 1, 0))
        im = freq_raster.plot(ax=ax, cmap=cmap, vmin=1, add_colorbar=False)
    if polys is not None:
        polys.plot(ax=ax, facecolor='none', edgecolor='b', alpha=.8)
    ax.set_aspect('equal')
    ax.autoscale()
    ax.axis('off')
    ax.set_title('')
    fig.tight_layout()
    return fig

def save_patches(gdf, geoid, suffix, log_online=False, result_folder='results/'):
    """save the patches to a shapefile."""
    fn = f'{result_folder}minicubes_{geoid}_{suffix}.geojson'
    os.makedirs(result_folder, exist_ok=True)
    gdf.to_file(fn)

    if log_online:
        wandb.save(fn, policy='now')

def log_result(inds, gen, score_names, freq_raster, freq, co_mask, shp, nr_baseline_patches, log_online, edge_size, log_dir, padded=True, plot_coord=True, is_best=False):
    """log patches as image and the fitness."""
    def func(i, ind, gen, score_names, freq_raster, freq, co_mask, shp, nr_baseline_patches, log_online, plot_coord, edge_size, log_dir, padded, is_best=False):
        scores = compute_metrics(ind, co_mask, freq_raster, nr_baseline_patches)
        fitness = {k:v for k,v in zip(score_names, scores)}
        fitness['0_fitness'] = ind.fitness.values

        # bring patches into geopandas format
        if padded:
            patches = [(i-edge_size,j-edge_size,k-edge_size,h-edge_size) for i,j,k,h in ind]
        else:
            patches = ind

        patch_coords = pixels_idx_to_coord(patches, freq)
        geometry = [box(*entry) for entry in patch_coords]
        gdf = gpd.GeoDataFrame(geometry=geometry, crs=freq.rio.crs)
        if plot_coord:
            # gdf = gpd.GeoDataFrame({'geometry':geometry})
            fig = plot_county_and_patches(gdf, shp, freq_raster=freq)
        else:
            patch_mask = create_patch_mask(patches, ind.bounds)
            fig, ax = plt.subplots()
            ax.imshow(patch_mask)
            ax.imshow(co_mask,alpha=0.3)

        if log_online:
            wandb.log({'gen':gen,f'gp.fitness_best_{i}': fitness})
            wandb.log({'gen':gen,f'gp.img_best_{i}': wandb.Image(fig)})
            plt.close()
        else:
            geoid = shp['GEOID'].values[0]
            print(f'Logging to {log_dir}{geoid}/:', f'gp.fitness_best_{i}', ind.fitness.values)
            os.makedirs(f'{log_dir}{geoid}/', exist_ok=True)
            with open(f'{log_dir}{geoid}/{geoid}_gp_fitness_{i}.txt', 'a') as f:
                f.write(f"{gen}: {json.dumps(fitness)}\n")
            fig.savefig(f'{log_dir}{geoid}/{geoid}_gp_img_{i}_{gen}.png', dpi=300, bbox_inches='tight')
        if is_best:
            if log_online:
                for k,v in fitness.items():
                    wandb.run.summary[f'best_{k}'] = v
            save_patches(gdf, shp['GEOID'].values[0], str(i) + '_ga', result_folder=log_dir, log_online=log_online)

    threads = []
    for i, ind in enumerate(inds):
        thread = threading.Thread(target=func, args=(i, ind, gen, score_names, freq_raster, freq, co_mask, shp, nr_baseline_patches, log_online, plot_coord, edge_size, log_dir, padded, is_best))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    plt.close('all')

def load_freqs(sample, home_folder):
    geoid = sample['GEOID'].values[0]

    # crop freq
    crops = ['corn', 'cotton', 'oat', 'soybean', 'winterwheat']
    ds = []
    for crop in crops:
        fn = os.path.join(home_folder, f'data/crop_frequency_county_gee2018_2022/{crop}/{crop}_{geoid}.tif')
        if not os.path.exists(fn):
            continue
        ds.append(rxr.open_rasterio(fn, masked=True).sel(band=1))
    ds = xr.concat(ds, dim='band')
    ds = ds.rio.write_crs("epsg:4269")
    ds = ds.rio.reproject(sample.crs, resolution=30)
    ds = ds.where(~(ds == -128), 0) # replace the filldata (nan) of -128 with 0
    ds = ds.sum('band')
    crop_freq = ds.rio.clip(sample.geometry.apply(mapping), sample.crs)
    co_mask = crop_freq.notnull().to_numpy()
    crop_freq = crop_freq.astype(np.uint8) # convert to int to save memory (needs to be done afterwards)

    return co_mask, (crop_freq, )

def shp_to_utm_crs(shp):
    """convert the shape from WGS84 to UTM crs."""
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(*shp.geometry.bounds.values[0]),
    )

    # Save the CRS
    epsg = utm_crs_list[0].code
    utm_crs = CRS.from_epsg(epsg)
    shp = shp.to_crs(utm_crs)
    return shp

def gp_co(shp, hparams, home_folder, use_utm=True):
    """ execute the genetic program for a county.
    shp: geopandas dataframe with the shape to cover
    edge_size: int, the edge size of the patch
    hparams: dict, hyper-parameters for the evolution
    """
    if use_utm:
        shp = shp_to_utm_crs(shp)
    edge_size = hparams.pop('edge_size')
    # init log
    tags = hparams.pop('tags')
    if hparams['online_log']:
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(entity=os.environ.get('WANDB_USER'), project='minicube-creation', config=hparams, tags=tags)
        wandb.config.geoid = str(shp['GEOID'].values[0])
        wandb.config.edge_size = edge_size

    # prepare frequencies
    co_mask, freqs = load_freqs(shp, home_folder)

    freqs = freqs[0]
    freq_raster = ski.img_as_ubyte(freqs)
    freq_raster = np.pad(freq_raster, edge_size, mode='constant', constant_values=0)

    bounds = freq_raster.shape

    # load the nr of patches in baseline
    new_co_mask = np.pad(co_mask, edge_size, mode='constant', constant_values=0)
    squares = split_image_mask_to_polygons(new_co_mask,thresh=0.01,side_length=edge_size, start_point='top_left')
    nr_baseline_patches = len(squares)

    #check if county is smaller than edge_size -> use the centroid
    if edge_size > freqs.rio.shape[0] and edge_size > freqs.rio.shape[1]:
        print(f"skipping county {shp['GEOID'].values[0]}: size county smaller than edge_size")
        x, y = shp.geometry.centroid.bounds.values[0][:2]
        buf = edge_size/2
        poly = box(x - buf, y - buf, x + buf, y + buf)
        gdf = pd.DataFrame({'geometry': [poly]})
        save_patches(gdf, shp['GEOID'].values[0], log_online=hparams['online_log'])
        return

    # prepare toolbox
    toolbox = base.Toolbox()
    # make it multi threaded
    mp = hparams.pop('multi_process', 0)
    if mp > 0:
        pool = multiprocessing.Pool(mp)
        toolbox.register("map", pool.map)

    # init creator
    weights = hparams.pop('fitness_weights')
    creator.create("FitnessFunc", base.Fitness, weights=(-1,))
    toolbox.register("evaluate", fitness, weights=weights, freq_raster=freq_raster, co_mask=new_co_mask, nr_baseline_patches=nr_baseline_patches) # register the goal / fitness function

    # register the population as a list of individuals
    polys = []
    for start_pos in ["top_left", "top_right", "bottom_left", "bottom_right"]:
        polys.append(split_image_mask_to_polygons(new_co_mask, edge_size, thresh=0.1, start_point=start_pos))

    creator.create("Individual", list, fitness=creator.FitnessFunc, edge_size=edge_size, bounds=bounds)
    toolbox.register("individual", create_individual_rand, creator.Individual, co_mask=new_co_mask, bounds=bounds, edge_size=edge_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("log_image", log_result, score_names=hparams['score_names'], freq_raster=freq_raster, freq=freqs, co_mask=new_co_mask, shp=shp, nr_baseline_patches=nr_baseline_patches, log_online=hparams['online_log'], edge_size=edge_size, log_dir=hparams['log_dir'], plot_coord=True)

    # register the generation creation
    toolbox.register("mutate", mutate, pb=hparams.pop('noisepb'), perc=hparams.pop('noise_nr_perc'))
    func = hparams.pop('mutate')
    if func == 'cxTwoPoint_remove_overlap':
        toolbox.register("mate", cxTwoPoint_remove_overlap)
    elif func == 'cxBest_remove_overlap':
        toolbox.register("mate", cxBest_remove_overlap, co_mask=new_co_mask, freq_raster=freq_raster)
    elif func == 'cxTwoPoint_mod':
        toolbox.register("mate", cxTwoPoint_mod)
    else:
        raise ValueError(f'func {func} unknown')

    # register the selection operator
    toolbox.register("select", tools.selTournament, tournsize=hparams.pop('tournsize', 3))

    algorithm = hparams.pop('algorithm')
    if algorithm == 'eaSimple':
        algo = EaSimple(toolbox, multiobjective=False, **hparams)
    else:
        raise ValueError(f'algo {algorithm} unknown')

    algo.call()

    if hparams['online_log']:
        wandb.finish()

    del creator.Individual
    del creator.FitnessFunc
