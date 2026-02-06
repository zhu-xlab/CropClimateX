#!/usr/bin/env python3
"""
Daymet Data Cleaner (County-Level Parallelism)

Goal:
    Create a clean version of Daymet Zarr stores by filling sporadic temporal gaps.

Processing Strategy:
    - One process handles one complete Zarr store (county/segment level)
    - This reduces I/O overhead and avoids excessive file locking

Method:
    - Linear interpolation along the time dimension
    - Conservative interpolation limit to avoid over-smoothing
"""

import os
import xarray as xr
import numpy as np
import multiprocessing
import time
import traceback
from typing import Tuple, Dict
from tqdm import tqdm

# ================= CONFIGURATION =================
# Input path: original Daymet data (may contain temporal gaps)
INPUT_ROOT = "/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/daymet"

# Output path: cleaned Daymet data (must be different from INPUT_ROOT)
OUTPUT_ROOT = "/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/daymet_cleaned"

# Variables to be cleaned
VARS_TO_FIX = ['prcp', 'tmax', 'tmin', 'srad']

# Maximum number of consecutive days allowed for interpolation
# We have found that most gaps are 1-3 days; setting a limit of 5 days
# helps avoid over-smoothing while filling most gaps.
INTERPOLATE_LIMIT = 5

# Start and end index for processing segments (None = process all)
# Example: START_INDEX=0, END_INDEX=10 processes first 10 segments
START_INDEX = 0  # Set to integer to start from specific segment
END_INDEX = 2600    # Set to integer to process up to specific segment
# =================================================


def clean_single_cube(input_cube_path: str, output_cube_path: str) -> bool:
    """
    Clean a single cube by interpolating short temporal gaps.

    Parameters
    ----------
    input_cube_path : str
        Path to the original cube Zarr store
    output_cube_path : str
        Path to the cleaned cube Zarr store

    Returns
    -------
    bool
        True if cleaning succeeds, False otherwise
    """
    try:
        with xr.open_zarr(input_cube_path, consolidated=False) as ds:
            # Create a shallow copy of the dataset
            ds_clean = ds.copy(deep=False)
            cleaned_vars = {}

            for var in VARS_TO_FIX:
                if var in ds:
                    da = ds[var]
                    if 'time' in da.dims:
                        # Core cleaning logic:
                        # Rechunk time dimension to single chunk (required for interpolate_na with dask)
                        da_rechunked = da.chunk({'time': -1})
                        # Linear interpolation along time with a strict limit
                        da_fixed = da_rechunked.interpolate_na(
                            dim='time',
                            method='linear',
                            limit=INTERPOLATE_LIMIT,
                            use_coordinate=True
                        )
                        cleaned_vars[var] = da_fixed

            if cleaned_vars:
                ds_clean = ds_clean.assign(cleaned_vars)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_cube_path), exist_ok=True)

            # Write cleaned data
            ds_clean.to_zarr(output_cube_path, mode='w', consolidated=True)
            return True

    except Exception as e:
        print(f"[ERROR] Failed to clean cube {os.path.basename(input_cube_path)}: {e}")
        return False


def process_county_segment(args: Tuple[str, str]) -> Dict:
    """
    Worker function that processes one full Daymet Zarr store
    (e.g., daymet_16029_0-9.zarr).

    Parameters
    ----------
    args : tuple
        (segment_path, segment_name)

    Returns
    -------
    dict
        Processing statistics for this segment
    """
    segment_path, relative_segment_name = args

    stats = {
        'segment': relative_segment_name,
        'total_cubes': 0,
        'success_cubes': 0,
        'failed_cubes': 0,
        'errors': []
    }

    output_segment_base = os.path.join(OUTPUT_ROOT, relative_segment_name)

    try:
        if not os.path.isdir(segment_path):
            return stats

        # Identify cube directories (numeric folder names)
        # Verify they contain valid zarr data (check for prcp folder)
        items = os.listdir(segment_path)
        cube_folders = []
        for d in items:
            cube_path = os.path.join(segment_path, d)
            if d.isdigit() and os.path.isdir(cube_path):
                # Verify it's a valid zarr cube by checking for variable folders
                if os.path.exists(os.path.join(cube_path, 'prcp')):
                    cube_folders.append(d)

        stats['total_cubes'] = len(cube_folders)

        for cube_id in cube_folders:
            input_cube = os.path.join(segment_path, cube_id)
            output_cube = os.path.join(output_segment_base, cube_id)

            success = clean_single_cube(input_cube, output_cube)
            if success:
                stats['success_cubes'] += 1
            else:
                stats['failed_cubes'] += 1
                stats['errors'].append(f"Cube {cube_id} failed")

    except Exception as e:
        stats['errors'].append(f"Segment-level error: {str(e)}")
        traceback.print_exc()

    return stats


def main():
    # Safety check to prevent overwriting original data
    if os.path.abspath(INPUT_ROOT) == os.path.abspath(OUTPUT_ROOT):
        print("CRITICAL ERROR: Output path is the same as input path. Aborting.")
        return

    print("=" * 60)
    print("DAYMET DATA CLEANER (SEGMENT-LEVEL PARALLEL MODE)")
    print(f"Input path : {INPUT_ROOT}")
    print(f"Output path: {OUTPUT_ROOT}")
    print(f"Variables  : {VARS_TO_FIX}")
    print(f"Interpolation limit (days): {INTERPOLATE_LIMIT}")
    if START_INDEX is not None or END_INDEX is not None:
        print(f"Index range: {START_INDEX} to {END_INDEX}")
    print("=" * 60)
    print()

    if not os.path.exists(INPUT_ROOT):
        print("Input path does not exist.")
        return

    # Scan for Daymet Zarr segments
    tasks = []
    print("Scanning for Daymet Zarr segments...")

    for item in os.listdir(INPUT_ROOT):
        full_path = os.path.join(INPUT_ROOT, item)
        if os.path.isdir(full_path) and item.endswith('.zarr'):
            tasks.append((full_path, item))

    print(f"Number of segments found: {len(tasks)}")

    # Apply start/end index filtering
    if START_INDEX is not None or END_INDEX is not None:
        start = START_INDEX if START_INDEX is not None else 0
        end = END_INDEX if END_INDEX is not None else len(tasks)
        tasks = tasks[start:end]
        print(f"Processing subset: index {start} to {end} ({len(tasks)} segments)")
    else:
        print("Processing all segments.")

    # Determine number of worker processes
    num_cpus = max(1, os.cpu_count() - 4)
    print(f"Using {num_cpus} worker processes.")
    print("Each worker processes one full segment.")

    start_time = time.time()

    total_cubes_processed = 0
    total_cubes_failed = 0

    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_county_segment, tasks),
                total=len(tasks),
                desc="Processing segments"
            )
        )

    # Summarize results
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)

    failed_segments = []

    for res in results:
        total_cubes_processed += res['success_cubes']
        total_cubes_failed += res['failed_cubes']
        if res['errors'] or res['failed_cubes'] > 0:
            failed_segments.append(res)

    print(f"Total segments processed : {len(tasks)}")
    print(f"Total cubes cleaned      : {total_cubes_processed}")
    print(f"Total cubes failed       : {total_cubes_failed}")

    if failed_segments:
        print(f"\nSegments with issues: {len(failed_segments)}")
        for fail in failed_segments[:5]:
            print(f"  {fail['segment']} -> {fail['errors']}")
        if len(failed_segments) > 5:
            print(f"  ... and {len(failed_segments) - 5} more.")
    else:
        print("\nAll segments processed successfully.")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed / 60:.2f} minutes")
    print(f"Cleaned data location: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
