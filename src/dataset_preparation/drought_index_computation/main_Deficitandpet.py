import os
import xarray as xr
import numpy as np
import pandas as pd
import json
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, current_process
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless server
import matplotlib.pyplot as plt

# Key references:
# FAO-56 Paper: https://www.researchgate.net/publication/235704197_Crop_evapotranspiration-Guidelines_for_computing_crop_water_requirements-FAO_Irrigation_and_drainage_paper_56
# Hagreaves PET: https://www.researchgate.net/publication/247373660_Reference_Crop_Evapotranspiration_From_Temperature
# Daymet Data Info: https://daymet.ornl.gov/overview

# ================= CONFIG =================
DAYMET_ROOT = "/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/daymet_cleaned"
GEOJSON_ROOT = "/home/zhiyuan/Codes/CropClimateX-main/minicubes_geometry"
OUTPUT_ROOT = "/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/daymet_pet_deficit"

VARIABLES = ['prcp', 'tmax', 'tmin', 'srad']

# 0.0135 * 0.408 is the coefficient for PET
# 0.408 is used to convert from MJ/m2/day to mm/day, it is usually called as the λ, which is the latent heat of vaporization, 2.45 [MJ kg-1], the detailed explanation is in the last part of code.
# 0.0135 = around 0.0023 / 0.17, where 0.0023 is the original coefficient and 0.17 is an empirical adjustment by Hagreaves.
# Originally, hagreaves compute PET from Ra (extraterrestrial radiation), because he does not have Rs (solar radiation) data.
# For this purpose, he used an equatian to substitute Rs with Ra. This equation is: Rs = Krs * Ra * sqrt(Tmax - Tmin)
# Krs is an empirical coefficient, often set to 0.16 for interior regions and 0.19 for coastal regions, for most cases, 0.17 is used.
# Now let us consider how can we transfer original pet equation to the one using Rs directly.
# Original Hagreaves equation: PET = 0.0023 * (Tmean + 17.8) * (Tmax - Tmin)^0.5 * Ra * 0.408.
# Substitute Rs equation into it: Ra = Rs / (Krs * sqrt(Tmax - Tmin))
# Now we have can use Rs to compute PET: PET = 0.0023 / Krs * 0.408 * (Tmean + 17.8) * Rs
# Therefore, the new coefficient becomes 0.0023 / 0.17 = 0.0135
# Finally we have: PET = 0.0135 * 0.408 * (Tmean + 17.8) * Rs.
PET_COEFF = 0.0135 * 0.408

# Processing Range (Based on folder count, not unique county count)
START_INDEX = 0
END_INDEX = 2500  # Increased range to ensure we cover multiple segments if they exist

# Set to True to print calculation details for a randomly selected cube
DEBUG_MODE = True

# Set to True to generate seasonality verification plots after processing
VERIFY_SEASONALITY = True
N_VERIFICATION_CUBES = 20  # Number of random cubes to verify
N_VERIFICATION_YEARS = 3  # Number of random years to plot per pixel
# ==========================================

def get_epsg_from_geojson(county_id):
    """
    Retrieves the EPSG code from the GeoJSON file for a specific county.
    """
    geojson_path = os.path.join(GEOJSON_ROOT, f"minicubes_{county_id}.geojson")
    
    if not os.path.exists(geojson_path):
        # Fallback or warning if GeoJSON is missing
        print(f"[WARN] GeoJSON not found for {county_id}, assuming EPSG:32616")
        return "32616"
        
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
        # format: urn:ogc:def:crs:EPSG::32616
        crs_name = data['crs']['properties']['name']
        return crs_name.split("::")[-1]
    except Exception as e:
        print(f"[ERROR] Failed to read CRS for {county_id}: {e}")
        return "32616"

def calculate_latitude_grid(x_array, y_array, epsg_code):
    """
    Converts projected coordinates (x, y in meters) to Latitude (degrees).
    Returns a 2D grid of Latitudes.
    """
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    xx, yy = np.meshgrid(x_array, y_array)
    _, lat_grid = transformer.transform(xx, yy)
    return lat_grid.astype(np.float32)

def calculate_daylength_seconds_vectorized(lat_grid, day_of_year):
    """
    Vectorized calculation of daylength in seconds using FAO-56 equations.
    """
    lat_rad = np.radians(lat_grid)
    # Solar declination (delta)
    solar_declination = 0.409 * np.sin((2.0 * np.pi / 365.0) * day_of_year - 1.39)
    
    # Sunset hour angle (omega_s)
    # Clip values to [-1, 1] to handle potential floating point errors near poles/equator
    tan_val = -np.tan(lat_rad) * np.tan(solar_declination)
    tan_val = np.clip(tan_val, -1.0, 1.0) 
    
    sunset_hour_angle = np.arccos(tan_val)
    daylength_hours = (24.0 / np.pi) * sunset_hour_angle
    daylength_hours = np.clip(daylength_hours, 0.0, 24.0)  # Ensure valid range
    
    return daylength_hours * 3600.0

def debug_print_calculation(cube_name, lat, doy, date, srad, tmean, daylength, pet, y_idx, x_idx):
    """
    Helper function to print a sanity check of the physics calculations.
    Prints values for a single pixel at a specific time step.
    """
    print(f"\n--- DEBUG CHECK: {cube_name} ---")
    print(f"Random Pixel : (y={y_idx}, x={x_idx})")
    print(f"Random Date  : {date}")
    print(f"Location Lat : {lat:.4f} degrees")
    print(f"Day of Year  : {doy}")
    print(f"Input Srad   : {srad:.2f} W/m^2")
    print(f"Input Tmean  : {tmean:.2f} C")
    print(f"Calc Daylen  : {daylength/3600.0:.2f} hours")
    print(f"Calc PET     : {pet:.4f} mm/day")
    print(f"--------------------------------\n")

def process_single_cube(args):
    """
    Worker function to process a single Zarr group (minicube).
    args: (cube_input_path, cube_output_path, epsg_code, enable_debug)
    """
    in_path, out_path, epsg_code, enable_debug = args
    cube_id = f"{os.path.basename(os.path.dirname(in_path))}/{os.path.basename(in_path)}"

    try:
        # Open dataset (consolidated=False is safer for potentially fragmented zarrs)
        ds = xr.open_zarr(in_path, consolidated=False)

        # Validate variables
        missing_vars = [v for v in VARIABLES if v not in ds]
        if missing_vars:
            return f"[SKIP] {cube_id}: Missing variables {missing_vars}"

        # 1. Load Data
        times = pd.to_datetime(ds.time.values)
        doy = times.dayofyear.values
        ntime = len(times)
        
        # Load arrays into memory (float32 to save RAM)
        tmax = ds['tmax'].values.astype(np.float32)
        tmin = ds['tmin'].values.astype(np.float32)
        prcp = ds['prcp'].values.astype(np.float32)
        srad = ds['srad'].values.astype(np.float32)

        #transfer tmax and tmin from Kelvin to deg C
        tmax = tmax - 273.15
        tmin = tmin - 273.15
        
        tmean = 0.5 * (tmax + tmin)

        # 2. Calculate Latitude (Spatial)
        x_vals = ds['x'].values
        y_vals = ds['y'].values
        # lat_grid shape: (ny, nx)
        lat_grid = calculate_latitude_grid(x_vals, y_vals, epsg_code) 

        # 3. Calculate Daylength (Spatio-Temporal)
        # daylength shape: (ntime, ny, nx)
        daylength = np.zeros_like(srad)
        
        for t_idx, d in enumerate(doy):
            # Broadcast calculation across the entire map for this day
            daylength[t_idx, :, :] = calculate_daylength_seconds_vectorized(lat_grid, d)

        # 4. Calculate PET and Deficit
        # Rs conversion: W/m2 * s / 1,000,000 * 10 ?? -> standard is MJ/m2/day
        # Daymet srad is W/m2 (average over daylight period? or day? usually day average in crop models)
        # Note: FAO-56 requires Rs in MJ/m2/day. 
        # If srad is Average Daily Radiation (W/m2): Rs = srad * 86400 / 1,000,000
        # If srad is Average Daylight Radiation: Rs = srad * daylength / 1,000,000
        # Your original formula: Rs = srad * daylength / 100_000 (Adjusted per your request)
        
        # Assuming your formula is calibrated for your specific srad unit:
        Rs = srad * daylength / 1_000_000.0 # Convert from W/m2 * s to MJ/m2/day
        pet = PET_COEFF * (17.8 + tmean) * Rs
        pet = np.maximum(pet, 1e-6)  # Enforce non-negative PET
        deficit = prcp - pet

        # 5. Debugging / Sanity Check
        # Print info for a randomly selected pixel and timestamp
        if enable_debug:
            rand_t = random.randint(0, ntime - 1)
            rand_y = random.randint(0, ds.dims['y'] - 1)
            rand_x = random.randint(0, ds.dims['x'] - 1)
            rand_date = times[rand_t]
            
            debug_print_calculation(
                cube_id, 
                lat_grid[rand_y, rand_x], 
                doy[rand_t],
                rand_date,
                srad[rand_t, rand_y, rand_x], 
                tmean[rand_t, rand_y, rand_x], 
                daylength[rand_t, rand_y, rand_x], 
                pet[rand_t, rand_y, rand_x],
                rand_y,
                rand_x
            )

        # 6. Save to Zarr
        ds_out = xr.Dataset(
            {
                'pet': (('time','y','x'), pet.astype(np.float32)),
                'Deficit': (('time','y','x'), deficit.astype(np.float32))
            },
            coords=ds.coords
        )
        
        # Preserve spatial_ref if exists
        if 'spatial_ref' in ds:
            ds_out['spatial_ref'] = ds['spatial_ref']
            
            # Link grid_mapping attribute
            ds_out['pet'].attrs['grid_mapping'] = 'spatial_ref'
            ds_out['Deficit'].attrs['grid_mapping'] = 'spatial_ref'

        # Use simple chunking
        ds_out.chunk({'time': -1}).to_zarr(out_path, mode='w', consolidated=True)
        return None # Success

    except Exception as e:
        return f"[ERROR] {cube_id}: {str(e)}"

def process_county_folder(county_zarr_path):
    """
    Scans a county folder (segment) for numeric sub-groups (minicubes).
    Handles names like 'daymet_01003_0-9.zarr' or 'daymet_01003_10-19.zarr'
    """
    folder_name = os.path.basename(county_zarr_path)
    
    # Extract County ID (e.g. 01003) assuming format daymet_{ID}_{segment}.zarr
    try:
        parts = folder_name.split('_')
        if len(parts) >= 2:
            county_id = parts[1]
        else:
            print(f"[SKIP] Invalid folder format: {folder_name}")
            return []
    except:
        return []

    # Get EPSG for this county
    epsg = get_epsg_from_geojson(county_id)

    # Find valid sub-groups (0, 1, 2...)
    tasks = []
    if os.path.isdir(county_zarr_path):
        for item in os.listdir(county_zarr_path):
            full_path = os.path.join(county_zarr_path, item)
            
            # Check if it is a directory and the name is a number
            if os.path.isdir(full_path) and item.isdigit():
                in_path = full_path
                # Mirror the structure in output
                out_path = os.path.join(OUTPUT_ROOT, folder_name, item)
                tasks.append((in_path, out_path, epsg, False))  # Debug flag set to False initially
    
    # Randomly select one cube for debugging if DEBUG_MODE is enabled
    if DEBUG_MODE and len(tasks) > 0:
        debug_idx = random.randint(0, len(tasks) - 1)
        # Recreate the task tuple with debug flag enabled
        tasks[debug_idx] = (tasks[debug_idx][0], tasks[debug_idx][1], tasks[debug_idx][2], True)
    
    return tasks

def verify_seasonality_visualization(n_cubes=3, n_years=5):
    """
    Randomly selects completed output cubes and generates PET time series plots
    to verify seasonality (winter low, summer high).
    
    Args:
        n_cubes: Number of random cubes to verify
        n_years: Number of random years to plot per pixel
    """
    print("\n" + "="*60)
    print("SEASONALITY VERIFICATION MODE")
    print("="*60)
    
    debug_plots_dir = os.path.join(os.path.dirname(OUTPUT_ROOT), "debug_plots")
    os.makedirs(debug_plots_dir, exist_ok=True)
    
    # 1. Find all completed output cubes
    all_output_cubes = []
    for root, dirs, files in os.walk(OUTPUT_ROOT):
        for d in dirs:
            if d.isdigit():  # Valid numeric cube
                cube_path = os.path.join(root, d)
                # Check if it has the expected zarr structure
                if os.path.exists(os.path.join(cube_path, '.zarray')) or \
                   os.path.exists(os.path.join(cube_path, '.zgroup')):
                    all_output_cubes.append(cube_path)
    
    if len(all_output_cubes) == 0:
        print("[WARN] No completed output cubes found for verification.")
        return
    
    # 2. Randomly select N cubes
    n_to_verify = min(n_cubes, len(all_output_cubes))
    selected_cubes = random.sample(all_output_cubes, n_to_verify)
    
    print(f"Found {len(all_output_cubes)} completed cubes.")
    print(f"Randomly selected {n_to_verify} cubes for verification.\n")
    
    # 3. Process each selected cube
    for cube_idx, cube_path in enumerate(selected_cubes, 1):
        try:
            cube_name = "/".join(cube_path.split("/")[-2:])
            print(f"[{cube_idx}/{n_to_verify}] Processing: {cube_name}")
            
            # Open the output zarr
            ds = xr.open_zarr(cube_path, consolidated=True)
            
            if 'pet' not in ds:
                print(f"  [SKIP] No 'pet' variable found in {cube_name}")
                continue
            
            # Get dimensions
            times = pd.to_datetime(ds.time.values)
            ny, nx = ds.dims['y'], ds.dims['x']
            
            # Randomly select a pixel
            rand_y = random.randint(0, ny - 1)
            rand_x = random.randint(0, nx - 1)
            
            print(f"  Random pixel: (y={rand_y}, x={rand_x})")
            
            # Extract all years available
            years = times.year.unique()
            if len(years) < n_years:
                print(f"  [INFO] Only {len(years)} years available, plotting all.")
                selected_years = sorted(years)
            else:
                selected_years = sorted(random.sample(list(years), n_years))
            
            print(f"  Selected years: {selected_years}")
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for year in selected_years:
                # Extract PET for this year and pixel
                year_mask = times.year == year
                year_times = times[year_mask]
                year_pet = ds['pet'].isel(y=rand_y, x=rand_x).values[year_mask]
                
                # Plot day of year vs PET
                doy = year_times.dayofyear
                ax.plot(doy, year_pet, label=f"{year}", alpha=0.7, linewidth=1.5)
            
            ax.set_xlabel("Day of Year", fontsize=12)
            ax.set_ylabel("PET (mm/day)", fontsize=12)
            ax.set_title(f"Seasonality Check: {cube_name}\nPixel (y={rand_y}, x={rand_x})", 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(1, 366)
            
            # Save plot
            safe_name = cube_name.replace("/", "_").replace(".zarr", "")
            plot_filename = f"seasonality_{safe_name}_y{rand_y}_x{rand_x}.png"
            plot_path = os.path.join(debug_plots_dir, plot_filename)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  Saved: {plot_filename}\n")
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {cube_path}: {e}\n")
    
    print("="*60)
    print(f"Verification complete. Plots saved to: {debug_plots_dir}")
    print("="*60 + "\n")

def main():
    print("Starting Processing...")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 1. List all Zarr segment folders
    all_dirs = sorted([
        os.path.join(DAYMET_ROOT, d) 
        for d in os.listdir(DAYMET_ROOT) 
        if d.endswith('.zarr') and os.path.isdir(os.path.join(DAYMET_ROOT, d))
    ])
    
    # 2. Slice the list based on user config
    # Note: This slices 'Segments', not unique counties.
    # If 01003 has 2 segments, they count as index 0 and 1.
    selected_dirs = all_dirs[START_INDEX:END_INDEX]
    print(f"Selected {len(selected_dirs)} segment folders for processing.")
    
    # 3. Generate Task List
    all_tasks = []
    for d in selected_dirs:
        all_tasks.extend(process_county_folder(d))
    
    print(f"Found {len(all_tasks)} individual cubes to process.")
    
    # 4. Execute in Parallel
    n_cpu = max(1, cpu_count() - 2)
    print(f"Using {n_cpu} CPU cores.")
    
    with Pool(processes=n_cpu) as pool:
        # verify results
        results = list(tqdm(pool.imap(process_single_cube, all_tasks), total=len(all_tasks)))
    
    # 5. Report Errors
    errors = [r for r in results if r is not None]
    if errors:
        print(f"\nCompleted with {len(errors)} errors:")
        for e in errors:
            print(e)
    else:
        print("\nAll cubes processed successfully.")
    
    # 6. Seasonality Verification (Optional)
    if VERIFY_SEASONALITY:
        verify_seasonality_visualization(n_cubes=N_VERIFICATION_CUBES, n_years=N_VERIFICATION_YEARS)

if __name__ == "__main__":
    main()


"""
Explanation of the 0.408 conversion factor in PET calculations
-------------------------------------------------------------

In FAO-56 based PET formulas, the factor 0.408 is used to convert radiation energy (MJ/m²/day) into equivalent water depth (mm/day).
This conversion is based on the latent heat of vaporization (λ).

1) Latent heat of vaporization:
   λ ≈ 2.45 MJ/kg
   → This is the amount of energy required to evaporate 1 kg of water.

2) Convert energy to evaporated mass:
   1 MJ / 2.45 MJ/kg = 0.408 kg
   → So 1 MJ of net radiation can evaporate about 0.408 kg of water.

3) Convert mass to volume:
   Density of water ≈ 1 kg/L
   → 0.408 kg = 0.408 L

4) Convert volume to depth:
   Over 1 m², 1 liter of water corresponds to 1 mm of depth.
   → 0.408 L/m² = 0.408 mm

Therefore:
   1 MJ/m²/day ≈ 0.408 mm/day

The density term does not explicitly appear because water density is
approximately 1 kg/L, causing the kg-to-L conversion to simplify directly.

Final interpretation:
   0.408 is the coefficient that converts radiation energy into
   an equivalent depth of evaporated water, assuming λ = 2.45 MJ/kg
   and water density = 1 kg/L.
"""