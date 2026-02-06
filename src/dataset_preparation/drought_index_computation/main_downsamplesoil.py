#%%
#!/usr/bin/env python3
"""
Soil Data Downsampling: 48x48 -> 12x12
Downsamples soil variables (sand, clay, soc) from 48x48 to 12x12 size
using 4x4 block averaging (skipping NaN values).
Coordinates and spatial_ref are copied from corresponding daymet cubes.
Because for original soil data, some pixels behave differently in terms of NaN values, for example, a pixel may have valid sand and clay values but NaN soc value.
Because of the high resolution, this phenomenon is not negligible. Filling at 48*48 is less useful than filling at 12*12.
Therefore this code serves for downsampling only, without any filling operation.
So we perform downsampling for each variable independently to better preserve valid data.
For the purpose of computing Palmer Drought Severity Index (PDSI), we need sand, clay, and soc values to estimate AWC and using AWC with 12*12 daymet data.
Therefore, the downsampled soil data will have the same 12x12 grid as daymet data for easy integration.
And the coordinates and spatial_ref are taken from daymet data to ensure alignment between downsampled soil data and daymet data.
"""

import os
import traceback
import re
import glob
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import time
import xarray as xr
import numpy as np
import multiprocessing
from tqdm import tqdm

# --- Configuration Parameters ---
CROPCLIMATEX_ROOT_DIR = "/home/zhiyuan/CropClimateX"
DAYMET_ROOT_DIR = "/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/daymet"
OUTPUT_ROOT_DIR = "/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/soil_downsampled"
MAX_PARALLEL_PROCESSES = max(1, os.cpu_count() - 30)

COUNTY_PROCESSING_START_INDEX = 0
COUNTY_PROCESSING_END_INDEX = 2500

# --- Helper Functions (Must be defined at top level for multiprocessing) ---
def extract_county_segments(base_root_dir: str, data_types_to_scan: List[str] = ["soil"]) -> Dict[str, Dict[str, List[str]]]:
    """Scan for soil segment zarr directories"""
    county_data = defaultdict(lambda: defaultdict(list))
    for dtype in data_types_to_scan:
        dtype_base_path = os.path.join(base_root_dir, dtype)
        if not os.path.isdir(dtype_base_path):
            continue
        search_pattern = os.path.join(dtype_base_path, '**', f"{dtype}_*_*.zarr")
        found_zarr_roots = glob.glob(search_pattern, recursive=True)
        for zarr_root_path in found_zarr_roots:
            if not os.path.isdir(zarr_root_path):
                continue
            segment_full_name = os.path.basename(zarr_root_path)
            strict_match = re.match(rf"^{re.escape(dtype)}_(\d{{5}})_(\d+-\d+)\.zarr$", segment_full_name)
            if strict_match:
                county_code, _ = strict_match.groups()
                county_data[county_code][dtype].append(zarr_root_path)
    return dict(county_data)

def count_cubes_in_segment_zarr(segment_zarr_path: str) -> int:
    """Count number of cubes in a segment"""
    try:
        if not os.path.exists(segment_zarr_path) or not os.path.isdir(segment_zarr_path):
            return 0
        subdirectories = [d for d in os.listdir(segment_zarr_path)
                          if d.isdigit() and os.path.isdir(os.path.join(segment_zarr_path, d))]
        return len(subdirectories)
    except Exception as e:
        print(f"Warning: Error counting cubes in '{segment_zarr_path}': {str(e)}")
        return 0

def get_segment_start_offset(segment_zarr_name: str) -> int:
    """Extract starting cube index from segment name"""
    match = re.search(r"_(\d+)-\d+\.zarr$", segment_zarr_name)
    if not match:
        raise ValueError(f"Invalid Zarr segment name format: {segment_zarr_name}")
    return int(match.group(1))

def find_corresponding_daymet_cube(county_code: str, cube_id: int, daymet_root: str) -> Optional[str]:
    """Find the corresponding daymet cube path for a given soil cube"""
    # Determine zarr range
    start = (cube_id // 10) * 10
    end = start + 9
    zarr_range = f"{start}-{end}"
    
    # Construct daymet cube path
    daymet_segment_name = f"daymet_{county_code}_{zarr_range}.zarr"
    daymet_cube_path = os.path.join(daymet_root, daymet_segment_name, str(cube_id))
    
    if os.path.isdir(daymet_cube_path):
        return daymet_cube_path
    return None

def downsample_soil_cube_48_to_12(soil_cube_path: str, 
                                   daymet_cube_path: str,
                                   output_cube_path: str,
                                   cube_id: int) -> bool:
    """
    Downsample a single soil cube from 48x48 to 12x12
    
    Args:
        soil_cube_path: Path to input soil cube (48x48)
        daymet_cube_path: Path to corresponding daymet cube (12x12) for coords/spatial_ref
        output_cube_path: Path to save downsampled data
        cube_id: Cube identifier for logging
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load soil data (48x48)
        ds_soil = xr.open_zarr(soil_cube_path, consolidated=False)
        
        # Check required variables
        required_vars = ['sand', 'clay', 'soc']
        missing_vars = [v for v in required_vars if v not in ds_soil]
        if missing_vars:
            print(f"  Warning: Cube {cube_id} missing variables {missing_vars}")
            return False
        
        # Load daymet data for coordinates and spatial_ref
        ds_daymet = xr.open_zarr(daymet_cube_path, consolidated=False)
        
        # Check if daymet has required coords
        if 'x' not in ds_daymet.coords or 'y' not in ds_daymet.coords:
            print(f"  Warning: Daymet cube {cube_id} missing x or y coordinates")
            return False
        
        # Downsample each variable using 4x4 block averaging
        downsampled_vars = {}
        for var_name in required_vars:
            var_data = ds_soil[var_name]
            
            # Use coarsen to do 4x4 block averaging, skipping NaN
            # Assuming dimensions are (y, x) for static soil data
            if var_data.dims == ('y', 'x'):
                downsampled = var_data.coarsen(y=4, x=4, boundary='trim').mean(skipna=True)
            else:
                print(f"  Warning: Unexpected dimensions for {var_name} in cube {cube_id}: {var_data.dims}")
                return False
            
            downsampled_vars[var_name] = downsampled
        
        # Create output dataset with daymet coordinates
        ds_out = xr.Dataset(
            {
                'sand': (('y', 'x'), downsampled_vars['sand'].values.astype(np.float32)),
                'clay': (('y', 'x'), downsampled_vars['clay'].values.astype(np.float32)),
                'soc': (('y', 'x'), downsampled_vars['soc'].values.astype(np.float32))
            },
            coords={
                'x': ds_daymet['x'],
                'y': ds_daymet['y']
            }
        )
        
        # Preserve spatial_ref from daymet if it exists
        if 'spatial_ref' in ds_daymet:
            ds_out['spatial_ref'] = ds_daymet['spatial_ref']
            
            # Link grid_mapping attribute for each variable
            ds_out['sand'].attrs['grid_mapping'] = 'spatial_ref'
            ds_out['clay'].attrs['grid_mapping'] = 'spatial_ref'
            ds_out['soc'].attrs['grid_mapping'] = 'spatial_ref'
        
        # Add metadata
        ds_out['sand'].attrs['long_name'] = 'Sand content (downsampled from 48x48 to 12x12)'
        ds_out['clay'].attrs['long_name'] = 'Clay content (downsampled from 48x48 to 12x12)'
        ds_out['soc'].attrs['long_name'] = 'Soil organic carbon (downsampled from 48x48 to 12x12)'
        ds_out['sand'].attrs['units'] = 'percent'
        ds_out['clay'].attrs['units'] = 'percent'
        ds_out['soc'].attrs['units'] = 'percent'
        
        # Save to zarr
        os.makedirs(os.path.dirname(output_cube_path), exist_ok=True)
        ds_out.to_zarr(output_cube_path, mode='w', consolidated=True)
        
        return True
        
    except Exception as e:
        print(f"  ERROR downsampling cube {cube_id}: {e}")
        # traceback.print_exc()
        return False

def process_single_soil_segment_downsampling(args: Tuple) -> Tuple[str, int, str]:
    """
    Process downsampling for all cubes in a single soil segment.
    
    Args:
        args (Tuple): (county_code, soil_segment_path, daymet_root, output_root)
    
    Returns:
        Tuple[str, int, str]: (segment path, number of cubes processed, status message)
    """
    county_code, soil_segment_path, daymet_root, output_root = args
    
    segment_filename = os.path.basename(soil_segment_path)
    
    # Construct output segment path (replace "soil" with "soil" in the output root)
    output_segment_path = os.path.join(output_root, segment_filename)
    
    cubes_processed = 0
    try:
        segment_start_offset = get_segment_start_offset(segment_filename)
        num_cubes = count_cubes_in_segment_zarr(soil_segment_path)
    except ValueError as e:
        return soil_segment_path, 0, f"Error getting offset/cubes: {e}"
    
    if num_cubes == 0:
        return soil_segment_path, 0, "No cubes found in segment"
    
    for relative_cube_idx in range(num_cubes):
        absolute_cube_id = segment_start_offset + relative_cube_idx
        soil_cube_path = os.path.join(soil_segment_path, str(absolute_cube_id))
        
        if not os.path.isdir(soil_cube_path):
            continue
        
        # Find corresponding daymet cube
        daymet_cube_path = find_corresponding_daymet_cube(county_code, absolute_cube_id, daymet_root)
        if daymet_cube_path is None:
            print(f"  Warning: No corresponding daymet cube found for soil cube {absolute_cube_id}")
            continue
        
        # Construct output cube path
        output_cube_path = os.path.join(output_segment_path, str(absolute_cube_id))
        
        # Downsample
        success = downsample_soil_cube_48_to_12(
            soil_cube_path,
            daymet_cube_path,
            output_cube_path,
            absolute_cube_id
        )
        
        if success:
            cubes_processed += 1
    
    return soil_segment_path, cubes_processed, "Success"

# --- Main Execution Logic ---
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    print("Starting Soil Data Downsampling (48x48 -> 12x12) pipeline (PARALLEL MODE)...")
    
    start_time = time.time()
    
    try:
        # Validate paths
        if not os.path.isdir(CROPCLIMATEX_ROOT_DIR):
            raise FileNotFoundError(f"CropClimateX root not found: {CROPCLIMATEX_ROOT_DIR}")
        if not os.path.isdir(DAYMET_ROOT_DIR):
            raise FileNotFoundError(f"Daymet root not found: {DAYMET_ROOT_DIR}")
        
        # Create output directory
        os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
        
        # Scan for soil segments
        print(f"Scanning soil data in: {CROPCLIMATEX_ROOT_DIR}...")
        all_county_segments = extract_county_segments(CROPCLIMATEX_ROOT_DIR, data_types_to_scan=["soil"])
        
        if not all_county_segments:
            raise ValueError("No soil data segments found")
        
        sorted_county_codes = sorted(all_county_segments.keys())
        print(f"Found {len(sorted_county_codes)} unique counties with soil data")
        
        # Select counties to process
        if COUNTY_PROCESSING_START_INDEX is not None or COUNTY_PROCESSING_END_INDEX is not None:
            start_idx = COUNTY_PROCESSING_START_INDEX if COUNTY_PROCESSING_START_INDEX is not None else 0
            end_idx = COUNTY_PROCESSING_END_INDEX + 1 if COUNTY_PROCESSING_END_INDEX is not None else len(sorted_county_codes)
            start_idx = max(0, start_idx)
            end_idx = min(len(sorted_county_codes), end_idx)
            selected_counties = sorted_county_codes[start_idx:end_idx] if start_idx < end_idx else []
        else:
            selected_counties = sorted_county_codes
        
        if not selected_counties:
            print("No counties selected for processing")
            exit()
        
        print(f"Selected {len(selected_counties)} counties to process")
        
        # Prepare tasks
        tasks = []
        for county_code in selected_counties:
            soil_segments = all_county_segments.get(county_code, {}).get('soil', [])
            for soil_segment_path in soil_segments:
                tasks.append((
                    county_code,
                    soil_segment_path,
                    DAYMET_ROOT_DIR,
                    OUTPUT_ROOT_DIR
                ))
        
        if not tasks:
            print("No segments to process")
            exit()
        
        print(f"Prepared {len(tasks)} segments for downsampling using {MAX_PARALLEL_PROCESSES} processes")
        
        # Process with multiprocessing
        results = []
        with multiprocessing.Pool(processes=MAX_PARALLEL_PROCESSES) as pool:
            for result in tqdm(pool.imap_unordered(process_single_soil_segment_downsampling, tasks),
                             total=len(tasks), desc="Downsampling Segments"):
                results.append(result)
        
        # Summarize results
        total_segments = len(tasks)
        successful_segments = 0
        total_cubes_processed = 0
        failed_segments = []
        
        for segment_path, cubes_done, status in results:
            if status == "Success":
                successful_segments += 1
                total_cubes_processed += cubes_done
            else:
                failed_segments.append(f"{os.path.basename(segment_path)}: {status}")
        
        print(f"\n--- Downsampling Summary ---")
        print(f"Total segments: {total_segments}")
        print(f"Successful segments: {successful_segments}")
        print(f"Total cubes downsampled: {total_cubes_processed}")
        
        if failed_segments:
            print(f"\nFailed segments ({len(failed_segments)}):")
            for detail in failed_segments[:10]:  # Show first 10
                print(f"  - {detail}")
            if len(failed_segments) > 10:
                print(f"  ... and {len(failed_segments) - 10} more")
    
    except FileNotFoundError as e:
        print(f"CRITICAL Error: {e}")
    except ValueError as e:
        print(f"CRITICAL Error: {e}")
    except Exception as e:
        print(f"CRITICAL Error: {e}")
        traceback.print_exc()
    
    finally:
        end_time = time.time()
        duration = end_time - start_time
        hours = duration / 3600
        minutes = (duration % 3600) / 60
        print(f"\nTotal execution time: {hours:.2f} hours ({minutes:.1f} minutes)")
        print("Downsampling pipeline finished.")
# %%
