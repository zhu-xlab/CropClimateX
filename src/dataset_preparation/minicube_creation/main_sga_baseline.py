# %%
import numpy as np
import geopandas as gpd
from shapely.geometry import box, mapping

from joblib import Parallel, delayed
import os
from tqdm import tqdm
import wandb
import skimage as ski
import rootutils
import os
from evolution_utils_pixel import compute_metrics, compute_metric, load_freqs, plot_county_and_patches, save_patches, scale_linear, shp_to_utm_crs
from gridding import split_image_mask_to_polygons, pixels_idx_to_coord
import itertools
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from pyproj import CRS
import matplotlib.pyplot as plt
home_folder = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
edge_size=400 # 400*30m=12km

# %% create patches by shifting
def create_baselines_sga(shp, hparams, strategy=0, use_utm=True, project=None):
    geoid = shp['GEOID'].values[0]
    if hparams['online_log']:
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(entity=os.environ.get('WANDB_USER'), project=project, config=hparams, tags=hparams['tags'])
        wandb.config.geoid = str(geoid)
        wandb.config.edge_size = edge_size

    if use_utm:
        shp = shp_to_utm_crs(shp)

    co_mask, freqs = load_freqs(shp, home_folder)
    freqs = list(freqs)
    co_mask = np.pad(co_mask, edge_size, mode='constant', constant_values=0)
    freq_raster = ski.img_as_ubyte(freqs[0])
    freq_raster = np.pad(freq_raster, edge_size, mode='constant', constant_values=0)
    squares = split_image_mask_to_polygons(co_mask,thresh=0.01,side_length=edge_size, start_point='top_left')
    nr_baseline_patches = len(squares)

    if strategy != 2:
        # create the baseline
        best_score = None
        best_squares = None
        best_scores = None
        for thr in [0.01]:
            for start_point in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
                squares = split_image_mask_to_polygons(co_mask,thresh=thr,side_length=edge_size, start_point=start_point)
                scores = compute_metrics(squares, co_mask, freq_raster, nr_baseline_patches)
                score = np.sum(scores * np.array(hparams['fitness_weights']))
                if best_score is None or score < best_score:
                    best_score = score
                    best_scores = scores
                    best_squares = squares
            
        # compute the coordinates of the patches
        squares_non_pad = [(i-edge_size,j-edge_size,k-edge_size,h-edge_size) for i,j,k,h in best_squares]
        coords = pixels_idx_to_coord(squares_non_pad, freqs[0])
        poly = [box(*entry) for entry in coords]
        gdf_b = gpd.GeoDataFrame({'geometry':poly}, crs=freqs[0].rio.crs)
        gdf_b.set_geometry('geometry', inplace=True)
        fig_b = plot_county_and_patches(gdf_b, shp, freq_raster=freqs[0])

        save_patches(gdf_b, geoid, suffix='baseline', log_online=hparams['online_log'], result_folder=hparams['log_dir'])
        if hparams['online_log']:
            for n, score in zip(hparams['score_names'], best_scores):
                wandb.log({f'{n}.baseline': score})

            wandb.log({'abs_nr_patches.baseline': len(poly)})
            wandb.log({'best_score.baseline': best_score})
            wandb.log({'img.baseline': wandb.Image(fig_b, caption="Baseline")})
            plt.close()
        else:
            print(f'baseline: {best_scores} = {best_score}')
            os.makedirs(hparams['log_dir'], exist_ok=True)
            fig_b.savefig(f"{hparams['log_dir']}{geoid}_baseline.png")
            print(f"You can find the results for baseline in: {hparams['log_dir']}")

    if strategy != 1:
        # create with sga
        start_polys = split_image_mask_to_polygons(co_mask, edge_size, thresh=0, start_point="top_left", ignore_mask=True)

        # vary the polygon
        end_polys = []
        comb = list(itertools.product(range(-edge_size//2, edge_size//2, hparams['pixel_shift']), range(-edge_size//2, edge_size//2, hparams['pixel_shift'])))
        def score_patches(x_shift, y_shift):
            polys = [[poly[0] - x_shift, poly[1] - y_shift, poly[2] - x_shift, poly[3] - y_shift] for poly in start_polys if poly[0] - x_shift >= 0 and poly[1] - y_shift >= 0 and poly[2] - x_shift < co_mask.shape[0] and poly[3] - y_shift < co_mask.shape[1]]
            final_polys = []
            for i in range(len(polys)):
                allocation, _, crop = compute_metric(i, polys, co_mask, freq_raster)
                # remove patches which have no crop or are outside the county
                if crop > hparams['crop_score'] and (allocation > hparams['allocation_score'] or crop > hparams['crop_score2']):
                    final_polys.append(polys[i])
            return (x_shift, y_shift, final_polys)
        end_polys = Parallel(n_jobs=hparams['multi_process'], backend='threading')(delayed(score_patches)(x_shift, y_shift) for x_shift, y_shift in tqdm(comb, desc='score patches'))

        # this should not be done in a multiprocess -> much faster with single process
        scores = [np.array(compute_metrics(end_polys[i][2], co_mask, freq_raster, nr_baseline_patches)) for i in tqdm(range(len(end_polys)), desc='score grids')]
        res_scores = [np.sum(hparams['fitness_weights'] * score) for score in scores]

        idx = np.argmin(res_scores)
        result = end_polys[idx]

        res = [[x1- edge_size ,x2- edge_size ,x3- edge_size ,x4 - edge_size] for x1,x2,x3,x4 in result[2]] # remove the added edge_size again
        patch_coords = pixels_idx_to_coord(res, freqs[0])
        geometry = [box(*entry) for entry in patch_coords]
        gdf_sga = gpd.GeoDataFrame(geometry=geometry, crs=freqs[0].rio.crs)
        fig_b_sga = plot_county_and_patches(gdf_sga, shp, freq_raster=freqs[0])
        
        save_patches(gdf_sga, geoid, suffix='sga', log_online=hparams['online_log'], result_folder=hparams['log_dir'])
        if hparams['online_log']:
            for n, score in zip(hparams['score_names'], scores[idx]):
                wandb.log({f'{n}.sga': score})
            wandb.log({'abs_nr_patches.sga': len(res)})
            wandb.log({'best_score.sga': res_scores[idx]})
            wandb.log({'img.sga': wandb.Image(fig_b_sga, caption="sga")})
            wandb.finish()
            plt.close()
        else:
            print(f'sga: {scores[idx]} = {res_scores[idx]}')
            os.makedirs(hparams['log_dir'], exist_ok=True)
            with open(hparams['log_dir'] + f"{geoid}_sga_best.txt", "w") as f:
                f.write(f'sga: {scores[idx]} = {res_scores[idx]}\n')
                f.write(f'abs_nr_patches.sga: {len(res)}')
            fig_b_sga.savefig(f"{hparams['log_dir']}{geoid}_sga_best.png")
            print(f"You can find the results for sga in: {hparams['log_dir']}")

        if hparams['online_log']:
            wandb.finish()

# %%
if __name__ == '__main__':
    data_dir = home_folder / "data/supp_data/"
    result_folder = os.path.join(home_folder, 'results/minicube_generation/')
    os.makedirs(result_folder, exist_ok=True)

    hparams = dict(
        score_names=['allocation', 'patch_overlap', 'nr_patches', 'crop'],
        fitness_weights=(-2, +5, +1, -4,),
        online_log=False, # log to wandb
        tags=[''], # tags for wandb
        crop_score=0.05,
        crop_score2=0.2,
        allocation_score=0.4,
        multi_process=int(os.cpu_count()//3),
        pixel_shift=30, # high number for testing
        log_dir=result_folder,
    )
    gdf = gpd.read_file(data_dir / 'counties.geojson')
    min, max = gdf['geom_sqkm'].min(), gdf['geom_sqkm'].max()
    # gdf = gdf.sample(frac=1) # randomize the df
    gdf = gdf[gdf['GEOID'].isin(['13155'])] # use this line to run a specific county, or comment for all

    user = os.environ.get('WANDB_USER')
    project = 'minicube-creation-baseline-sga'
    api = wandb.Api()
    for i in tqdm(gdf.index, total=len(gdf.index)):
        geoid = gdf.loc[[i]]['GEOID'].values[0]
        if hparams['online_log']:
            runs = api.runs(f"{user}/{project}", filters={"tags": hparams['tags'][0], "config.geoid": geoid})
            if len(runs) > 0:
                continue

        # update params
        nr_patches_scaled = scale_linear(gdf.loc[[i]]['geom_sqkm'].values[0], min, max, 4, 173) # smallest vs biggest countie and number of patches
        crop_score = 1/nr_patches_scaled * .5 # needs to have at least 50% crop, if crop would be evenly distributed across the county
        crop_score2 = 0.4 * crop_score # if area outside needs to have at least 40% crop than a usual patch (if evenly distributed)
        pixel_shift = scale_linear(gdf.loc[[i]]['geom_sqkm'].values[0], min, max, 1, 6)
        hparams.update({'pixel_shift': pixel_shift, 'crop_score': crop_score, 'crop_score2': crop_score2})
        create_baselines_sga(gdf.loc[[i]], hparams, project)