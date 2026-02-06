"""
This script calculates Palmer Drought Indices (PHDI, PDSI, etc.) from pre-aggregated Daymet data
on a per-county basis using multiprocessing. Each worker process handles all segments for a single county
to optimize resource usage and minimize overhead.
Please only use the "psuedo_weekly" method for Palmer calculation in this script.
Other methods were just used for experimental purposes.
The input prcp/pet data should have a time resolution of 1 week (7 days).
The input awc data should also have same size with the prcp/pet (12*12 in this case). If not, the code will implement the resampling internally.
Current input awc data is 12*12, so no resampling is triggered or needed.
"""
# %%
# main_PHDIandPDSI.py
import os
import traceback
from typing import Dict, List, Optional
import multiprocessing
import time
import re
from tqdm import tqdm # Ensure tqdm is imported for the county progress bar

# Ensure the PalmerAggregatedCalculator class is in an importable path
try:
    # Assuming your class is named PalmerAggregatedCalculator
    # and is in a file named PalmerAggregatedCalculator.py
    from Calculator_Week_Palmer_Final import PalmerAggregatedCalculator
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import PalmerAggregatedCalculator: {e}")
    print("Please ensure Calculator_Week_Palmer_Final.py is in the same directory or accessible in PYTHONPATH.")
    exit()

def get_all_segment_paths_for_county(data_root: str, county_id: str) -> List[str]:
    """
    Gets all relevant segment Zarr file paths for a specific County ID from the data root directory.
    """
    county_segment_paths = []
    if not os.path.isdir(data_root):
        print(f"Warning: Data root directory not found: {data_root}")
        return []
    try:
        for folder_name in os.listdir(data_root):
            # Ensure filename matches "PREFIX_COUNTYID_SUFFIX.zarr" structure
            if folder_name.endswith(".zarr") and \
               os.path.isdir(os.path.join(data_root, folder_name)) and \
               f"_{county_id}_" in folder_name: # Ensure filename contains "_COUNTYID_"
                county_segment_paths.append(os.path.join(data_root, folder_name))
    except Exception as e:
        print(f"Error scanning for segments for county {county_id} in '{data_root}': {e}")

    # Sort by segment start number (optional, but helps with ordered logging)
    def get_segment_start_num_from_path(path_str: str) -> int:
        match_seg = re.search(r"_(\d+)-(\d+)\.zarr$", os.path.basename(path_str))
        return int(match_seg.group(1)) if match_seg else -1 # If no match, place at the beginning
    county_segment_paths.sort(key=get_segment_start_num_from_path)
    return county_segment_paths

def count_cubes_in_segment(segment_zarr_path: str) -> int:
    """Counts how many cube subdirectories are in a segment Zarr directory."""
    try:
        if not os.path.exists(segment_zarr_path) or not os.path.isdir(segment_zarr_path): return 0
        subdirectories = [d for d in os.listdir(segment_zarr_path)
                          if d.isdigit() and os.path.isdir(os.path.join(segment_zarr_path, d))]
        return len(subdirectories)
    except Exception:
        # print(f"Warning: Error counting cubes in '{segment_zarr_path}': {str(e)}") # Optional: uncomment for detailed errors
        return 0

# --- Worker Function (processes PHDI/PDSI for all Segments in one County) ---
def worker_process_county_for_palmer(args_tuple: tuple) -> Dict:
    """
    Worker function to calculate PHDI/PDSI for all segments within a single county.
    """
    county_id_to_process, agg_daymet_root, soil_root_path, palmer_vars_to_compute, output_time_chunk, palmer_output_root, enable_nan_handling, calibration_end_date = args_tuple
    
    process_id = os.getpid()
    print(f"\n>>>> PID {process_id}: Starting Palmer processing for County: {county_id_to_process} <<<<")
    
    county_results_summary = {
        "county_id": county_id_to_process,
        "total_segments": 0,
        "successfully_processed_segments": 0,
        "failed_segments": [],
        "start_time": time.time(),
        "error": None # For county-level errors
    }

    try:
        # 1. Get all aggregated daymet segment paths for this county
        agg_daymet_segments_for_county = get_all_segment_paths_for_county(agg_daymet_root, county_id_to_process)
        county_results_summary["total_segments"] = len(agg_daymet_segments_for_county)

        if not agg_daymet_segments_for_county:
            print(f"  PID {process_id}: No aggregated daymet segments found for county {county_id_to_process}. Skipping.")
            county_results_summary["end_time"] = time.time()
            return county_results_summary

        for agg_daymet_segment_path in agg_daymet_segments_for_county:
            segment_basename = os.path.basename(agg_daymet_segment_path)
            print(f"  PID {process_id}: Processing segment {segment_basename} for county {county_id_to_process}...")
            
            try:
                # Find corresponding soil segment path (similar to original worker logic)
                soil_segment_full_path = None
                # Assuming daymet segment prefix (e.g., "daymet") needs to be replaced with "soil"
                # to find the corresponding soil segment. This replacement logic needs to match your filenames.
                daymet_prefix_in_filename = "daymet" 
                soil_prefix_in_filename = "soil"   
                
                # Try to infer soil_segment_name from agg_daymet_segment_path's basename
                # This logic should robustly extract FIPS and suffix (e.g., "0-9")
                # Example: daymet_01003_0-9.zarr -> soil_01003_0-9.zarr
                match_name_parts = re.match(rf"{daymet_prefix_in_filename}_(\d{{5}})_([^_.]+)(?:\.zarr)?", segment_basename)
                if match_name_parts:
                    fips_from_name = match_name_parts.group(1) # Should be county_id_to_process
                    suffix_from_name = match_name_parts.group(2) # e.g., "0-9"
                    expected_soil_segment_name = f"{soil_prefix_in_filename}_{fips_from_name}_{suffix_from_name}.zarr"
                    potential_soil_path = os.path.join(soil_root_path, expected_soil_segment_name)
                    if os.path.exists(potential_soil_path) and os.path.isdir(potential_soil_path):
                        soil_segment_full_path = potential_soil_path
                
                if (not soil_segment_full_path) and any(v in palmer_vars_to_compute for v in ['phdi', 'pdsi', 'cmi', 'rsm', 'rwd']):
                    print(
                        f"    Warning (PID {process_id}): Corresponding soil segment for {segment_basename} "
                        f"(expected at {potential_soil_path if match_name_parts else 'N/A'}) not found. "
                        "Selected variables that depend on AWC (e.g., rsm/rwd/phdi/pdsi/cmi) may be skipped."
                    )

                num_cubes_agg_daymet = count_cubes_in_segment(agg_daymet_segment_path)
                num_cubes_soil_seg = count_cubes_in_segment(soil_segment_full_path) if soil_segment_full_path else 0
                
                if num_cubes_agg_daymet == 0:
                    print(f"    Segment {segment_basename}: No cubes in aggregated daymet data. Skipping this segment.")
                    continue

                # Instantiate PalmerAggregatedCalculator for the current segment
                calculator = PalmerAggregatedCalculator(
                    aggregated_daymet_segment_path=agg_daymet_segment_path,
                    soil_segment_path=soil_segment_full_path, # Pass None if not found
                    num_cubes_agg_daymet=num_cubes_agg_daymet,
                    num_cubes_soil=num_cubes_soil_seg,
                    palmer_output_root=palmer_output_root,  # Pass the Palmer output root
                    enable_nan_handling=enable_nan_handling,  # Pass NaN handling flag
                    calibration_end_date=calibration_end_date  # Pass calibration end date
                )
                
                # Process Palmer indices for all cubes in this segment
                calculator.process_palmer_indices_for_current_segment(
                    target_palmer_variables=palmer_vars_to_compute,
                    output_time_chunk_size=output_time_chunk,
                    method='pseudo_weekly'  # Use the pseudo-weekly method as per your requirement
                )
                county_results_summary["successfully_processed_segments"] += 1
                print(f"  PID {process_id}: Successfully processed Palmer for segment {segment_basename}")
            except Exception as e_seg:
                error_msg_seg = f"Error processing Palmer for segment {segment_basename} within county {county_id_to_process}: {str(e_seg)}"
                print(f"  PID {process_id}: {error_msg_seg}")
                # traceback.print_exc() # Optional: print full traceback for segment error
                county_results_summary["failed_segments"].append({
                    "segment": segment_basename,
                    "error": str(e_seg)
                })
        
        county_results_summary["end_time"] = time.time()
        print(f"<<<< PID {process_id}: Finished Palmer processing for County: {county_id_to_process} >>>>")
        return county_results_summary

    except Exception as e_county_worker:
        error_msg_county = f"Major error in worker for County {county_id_to_process}: {str(e_county_worker)}"
        print(f"PID {process_id}: {error_msg_county}")
        traceback.print_exc()
        county_results_summary["error"] = error_msg_county # Record county-level worker error
        county_results_summary["end_time"] = time.time()
        return county_results_summary

# --- Main Execution Section ---
if __name__ == '__main__':
    multiprocessing.freeze_support() # Important for Windows compatibility
    
    print("Starting Palmer Index Calculation from Aggregated Data (Multiprocessing Per-COUNTY Mode)...")
    start_time_total = time.time()

    # --- Configuration ---
    AGGREGATED_DAYMET_ROOT = "/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/indices_from_daymet_standardallset_unified_1week_right"
    SOIL_INDICES_ROOT = "/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/awc_derived"
    PALMER_OUTPUT_ROOT = "/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/indices_from_daymet_Palmer"  # NEW: Separate output directory

    PALMER_VARIABLES_TO_COMPUTE = ['phdi','pdsi','rsm','rwd']  # List of Palmer variables to compute
    OUTPUT_TIME_CHUNK_SIZE = 256 # This parameter is passed to the worker for the Calculator
    ENABLE_NAN_HANDLING = False  # Set to True to enable NaN preprocessing (filling with mean), False to skip NaN handling
    CALIBRATION_END_DATE = '2023-01-01'  # Calibration period end date (format: 'YYYY-MM-DD')

    # Select counties to process by index from the sorted list of discovered FIPS codes
    COUNTY_PROCESSING_START_INDEX: Optional[int] = 0  # Example: Start with the first found county
    COUNTY_PROCESSING_END_INDEX: Optional[int] = 2500   # Example: Process all counties (None means to the end)
                                                      # To process only FIPS at index 0, set END_INDEX to 0.
    
    # Optional: Directly specify a list of FIPS codes to process. If set, index range is ignored.
    TARGET_FIPS_CODES_LIST: Optional[List[str]] = None # Set to None to use index range
    #TARGET_FIPS_CODES_LIST: Optional[List[str]] = ["01003","01043","06077","08011","13023","13037","16019","17005","17137","18003","18041","18117","18157","19049","19081","19113","20049","20201","20203","21035","21199","22081","24017","28107","29049","29143","31015","31059","34019","37057","37081","37107","40149","42081","46045","46057","46087","48095","55013","55027","05095","05107","17131","19095","19171","24037","29207","31159","36067","39023","48057","55055","39159",]
    #TARGET_FIPS_CODES_LIST: Optional[List[str]] = ['18117','48057','31159'] # Set to None to use index range
    #TARGET_FIPS_CODES_LIST: Optional[List[str]] = ["19095","06077","19081","55027","13037","37057","46087","17005","05095","19171","48421","39023","17131","29207","19049","51083"]  # Set to None to use index range
    # --- Multiprocessing Setup ---
    num_logical_cores = os.cpu_count()
    if num_logical_cores is None: num_logical_cores = 4 # Default if cpu_count fails
    # Adjust n_processes: leave some cores for system, or set to a fixed number
    # n_processes = max(1, num_logical_cores - 4 if num_logical_cores > 1 else 1) 
    n_processes = num_logical_cores - 5
    # n_processes = 1 # For testing or easier log tracking, set to 1
    print(f"Using {n_processes} worker processes for multiprocessing.")
    
    # Create Palmer output directory if it doesn't exist
    if not os.path.exists(PALMER_OUTPUT_ROOT):
        os.makedirs(PALMER_OUTPUT_ROOT, exist_ok=True)
        print(f"Created Palmer output directory: {PALMER_OUTPUT_ROOT}")
    
    try:
        # 1. Discover all available county FIPS codes by scanning filenames in AGGREGATED_DAYMET_ROOT
        all_county_ids_temp = set()
        if os.path.isdir(AGGREGATED_DAYMET_ROOT):
            for folder_name in os.listdir(AGGREGATED_DAYMET_ROOT):
                # Assuming filename format includes "_FIPS_", e.g., "daymet_01003_0-9.zarr"
                if folder_name.endswith(".zarr") and os.path.isdir(os.path.join(AGGREGATED_DAYMET_ROOT, folder_name)):
                    match = re.search(r"_(\d{5})_", folder_name) # Extracts 5-digit FIPS
                    if match:
                        all_county_ids_temp.add(match.group(1))
        all_available_county_codes = sorted(list(all_county_ids_temp))
        
        if not all_available_county_codes:
            print(f"No county codes found by scanning {AGGREGATED_DAYMET_ROOT}. Exiting.")
            exit()
        print(f"Found {len(all_available_county_codes)} unique counties by scanning source. First 5: {all_available_county_codes[:5]}...")

        # 2. Select target counties for processing based on configuration
        selected_target_county_ids_final = []
        if TARGET_FIPS_CODES_LIST is not None:
            selected_target_county_ids_final = [fips for fips in TARGET_FIPS_CODES_LIST if fips in all_available_county_codes]
            if len(selected_target_county_ids_final) != len(TARGET_FIPS_CODES_LIST):
                print("Warning: Some FIPS codes in TARGET_FIPS_CODES_LIST were not found among scanned available FIPS codes.")
        else: # Use index range
            start_idx = COUNTY_PROCESSING_START_INDEX if COUNTY_PROCESSING_START_INDEX is not None else 0
            end_idx = COUNTY_PROCESSING_END_INDEX if COUNTY_PROCESSING_END_INDEX is not None else len(all_available_county_codes) - 1
            
            start_idx = max(0, start_idx) # Ensure start_idx is not negative
            end_idx = min(end_idx, len(all_available_county_codes) - 1) # Ensure end_idx is within bounds

            if start_idx <= end_idx:
                selected_target_county_ids_final = all_available_county_codes[start_idx : end_idx + 1]
            else:
                print(f"Warning: Invalid county index range (Start: {start_idx}, End: {end_idx}). No counties selected via index.")
        
        if not selected_target_county_ids_final:
            print("No counties selected for processing. Exiting.")
            exit()
        print(f"Selected {len(selected_target_county_ids_final)} counties for Palmer calculation: {selected_target_county_ids_final if len(selected_target_county_ids_final) < 10 else str(len(selected_target_county_ids_final)) + ' counties'}")

        # 3. Prepare argument list for worker processes, one task per county
        # Each tuple: (county_id, agg_daymet_root, soil_root_path, palmer_vars_to_compute, output_time_chunk_size, palmer_output_root, enable_nan_handling, calibration_end_date)
        tasks_args_list = [
            (county_id, 
             AGGREGATED_DAYMET_ROOT,
             SOIL_INDICES_ROOT, 
             PALMER_VARIABLES_TO_COMPUTE, 
             OUTPUT_TIME_CHUNK_SIZE,
             PALMER_OUTPUT_ROOT,  # Add Palmer output root to args
             ENABLE_NAN_HANDLING,  # Add NaN handling flag to args
             CALIBRATION_END_DATE  # Add calibration end date to args
            ) 
            for county_id in selected_target_county_ids_final
        ]

        if tasks_args_list:
            print(f"\nSubmitting {len(tasks_args_list)} county-level Palmer calculation tasks to multiprocessing pool ({n_processes} processes)...")
            print(f"Palmer results will be saved to: {PALMER_OUTPUT_ROOT}")
            with multiprocessing.Pool(processes=n_processes) as pool:
                results = []
                # Using imap_unordered to get results as they complete, with a tqdm progress bar for counties
                for result_summary in tqdm(pool.imap_unordered(worker_process_county_for_palmer, tasks_args_list), 
                                           total=len(tasks_args_list), desc="Counties Processed"):
                    results.append(result_summary)
            
            print("\nMultiprocessing pool for Palmer calculation finished. Results per county:")
            total_segments_attempted_all = 0
            total_segments_successful_all = 0
            
            for res_summary in results: # results is a list of dictionaries
                print(f"  County ID: {res_summary['county_id']}")
                print(f"    Total segments for this county: {res_summary['total_segments']}")
                print(f"    Successfully processed segments: {res_summary['successfully_processed_segments']}")
                total_segments_attempted_all += res_summary['total_segments']
                total_segments_successful_all += res_summary['successfully_processed_segments']
                if res_summary.get("failed_segments"):
                    print(f"    Failed segments ({len(res_summary['failed_segments'])}):")
                    for failed_seg_info in res_summary['failed_segments']:
                        # Print only a snippet of the error to keep log concise
                        print(f"      - Segment: {failed_seg_info['segment']}, Error: {failed_seg_info['error'][:150]}...") 
                if res_summary.get("error"): # County-level worker error
                     print(f"    County-level worker error: {res_summary['error']}")
                processing_duration = res_summary.get("end_time", time.time()) - res_summary.get("start_time", time.time())
                print(f"    Processing time for this county: {processing_duration:.2f} seconds.")
            
            print(f"\nOverall Execution Summary:")
            print(f"  Total counties tasked: {len(selected_target_county_ids_final)}")
            print(f"  Total segments attempted across all selected counties: {total_segments_attempted_all}")
            print(f"  Total segments successfully processed: {total_segments_successful_all}")
            print(f"  Palmer results saved to: {PALMER_OUTPUT_ROOT}")

        else:
            print("No tasks were generated for Palmer index calculation.")

    except Exception as e_main:
        print(f"Error: An unexpected error occurred in the main Palmer calculation pipeline: {str(e_main)}")
        traceback.print_exc()
    finally:
        end_time_total = time.time()
        total_duration_seconds = end_time_total - start_time_total
        total_duration_hours = total_duration_seconds / 3600
        print(f"\nTotal Palmer (from aggregated) pipeline execution time: {total_duration_hours:.2f} hours ({total_duration_seconds:.2f} seconds).")
        print("Palmer Index Calculation (from aggregated data) Script finished.")
# %%