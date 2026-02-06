#%%
# unified_aggregate_main_multiprocessing.py
import os
import traceback
from typing import Dict, List, Optional, Any
import multiprocessing
import time
import re
import json
try:
    from UnifiedAggregator_selected_trial import UnifiedTemporalAggregator
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import UnifiedTemporalAggregator: {e}")
    print("Please ensure correct file is in the same directory or accessible in PYTHONPATH.")
    exit()

def get_county_codes_from_geojson(geojson_path: str) -> List[str]:
    """
    Reads a GeoJSON file and extracts a list of county GEOIDs in the order they appear.

    Args:
        geojson_path: The full path to the .geojson file.

    Returns:
        A list of county GEOID strings. Returns an empty list if an error occurs.
    """
    print(f"Attempting to load county list from GeoJSON: {geojson_path}")
    if not os.path.exists(geojson_path):
        print(f"CRITICAL ERROR: GeoJSON file not found at '{geojson_path}'")
        return []
    
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
        
        # Extract GEOID from each feature's properties
        # The order is preserved exactly as it is in the file
        county_codes = [feature['properties']['GEOID'] for feature in data['features']]
        
        print(f"Successfully loaded {len(county_codes)} county codes from GeoJSON.")
        return county_codes
    except json.JSONDecodeError as e:
        print(f"CRITICAL ERROR: Failed to parse GeoJSON file. Error: {e}")
        return []
    except (KeyError, TypeError) as e:
        print(f"CRITICAL ERROR: GeoJSON file has an unexpected structure. Could not find features or properties.GEOID. Error: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading the GeoJSON file: {e}")
        return []


def get_county_codes_from_generic_source(source_root: str) -> List[str]:
    """Generic county code scanner for various data structures"""
    county_codes = set()
    if not os.path.isdir(source_root):
        print(f"Warning: County scan source directory not found: {source_root}")
        return []
    try:
        for folder_name in os.listdir(source_root):
            folder_path = os.path.join(source_root, folder_name)
            if os.path.isdir(folder_path):
                # Try to extract county code from folder name
                match = re.search(r"_(\d{5})_", folder_name)
                if match:
                    county_codes.add(match.group(1))
    except Exception as e:
        print(f"Error scanning for county codes in '{source_root}': {e}")
    return sorted(list(county_codes))

def get_county_codes_from_nested_zarr_source(source_root: str) -> List[str]:
    """For nested zarr structure: daymet_01003_0-9/daymet_01003_0-9.zarr/0/variable"""
    county_codes = set()
    if not os.path.isdir(source_root):
        print(f"Warning: Nested zarr source directory not found: {source_root}")
        return []
    try:
        for folder_name in os.listdir(source_root):
            folder_path = os.path.join(source_root, folder_name)
            if os.path.isdir(folder_path) and folder_name.startswith("daymet_"):
                match = re.search(r"daymet_(\d{5})_", folder_name)
                if match:
                    county_code = match.group(1)
                    # Check for nested zarr structure
                    zarr_folder = os.path.join(folder_path, folder_name + ".zarr")
                    if os.path.isdir(zarr_folder):
                        try:
                            cube_folders = [d for d in os.listdir(zarr_folder) 
                                          if d.isdigit() and os.path.isdir(os.path.join(zarr_folder, d))]
                            if cube_folders:
                                county_codes.add(county_code)
                                print(f"Debug: Found nested zarr county {county_code} with {len(cube_folders)} cubes in {zarr_folder}")
                        except Exception as e:
                            print(f"Warning: Error checking cubes in {zarr_folder}: {e}")
    except Exception as e:
        print(f"Error scanning for nested zarr county codes in '{source_root}': {e}")
    return sorted(list(county_codes))

def get_county_codes_from_direct_zarr_source(source_root: str) -> List[str]:
    """For direct zarr structure: daymet_01003_0-9.zarr/0/variable"""
    county_codes = set()
    if not os.path.isdir(source_root):
        print(f"Warning: Direct zarr source directory not found: {source_root}")
        return []
    try:
        for folder_name in os.listdir(source_root):
            folder_path = os.path.join(source_root, folder_name)
            if os.path.isdir(folder_path) and folder_name.startswith("daymet_") and folder_name.endswith(".zarr"):
                # Extract county code from folder name like "daymet_01003_0-9.zarr"
                match = re.search(r"daymet_(\d{5})_", folder_name)
                if match:
                    county_code = match.group(1)
                    # Check if there are cube folders inside the zarr folder
                    try:
                        cube_folders = [d for d in os.listdir(folder_path) 
                                      if d.isdigit() and os.path.isdir(os.path.join(folder_path, d))]
                        if cube_folders:
                            county_codes.add(county_code)
                            print(f"Debug: Found direct zarr county {county_code} with {len(cube_folders)} cubes in {folder_path}")
                        else:
                            print(f"Warning: No cube folders found in {folder_path}")
                    except Exception as e:
                        print(f"Warning: Error checking cubes in {folder_path}: {e}")
    except Exception as e:
        print(f"Error scanning for direct zarr county codes in '{source_root}': {e}")
    return sorted(list(county_codes))

def worker_aggregate_county(args_tuple: tuple) -> str:
    """Worker function for multiprocessing county aggregation"""
    county_id_to_process, common_processing_config = args_tuple
    
    output_template = common_processing_config['output_template']
    agg_period = common_processing_config['agg_period']
    variables_config = common_processing_config['variables_config']
    max_segments = common_processing_config['max_segments_per_county']
    time_chunk_size_default = common_processing_config['time_chunk_size_default']
    mode = common_processing_config['mode']
    output_resolution = common_processing_config.get('output_resolution', None)

    print(f"\n>>>> [PID: {os.getpid()}] Starting {mode.upper()} Aggregation for County: {county_id_to_process} <<<<")
    
    try:
        aggregator = UnifiedTemporalAggregator(
            output_aggregated_root_template=output_template,
            aggregation_period=agg_period,
            variables_config=variables_config,
            mode=mode,
            output_resolution=output_resolution
        )

        # If time_chunk_size_default is None, the aggregator will auto-detect from input data
        if time_chunk_size_default is not None:
            aggregator.aggregate_indices(
                target_county_ids_list=[county_id_to_process],
                max_segments_per_county=max_segments,
                output_time_chunk_size=time_chunk_size_default
            )
        else:
            # Let the aggregator use its default (4) or auto-detect from config
            aggregator.aggregate_indices(
                target_county_ids_list=[county_id_to_process],
                max_segments_per_county=max_segments
            )
        return f"Successfully aggregated County {county_id_to_process} ({mode} mode)"
    except Exception as e_worker:
        print(f"Error during {mode} aggregation for County {county_id_to_process} (PID: {os.getpid()}): {str(e_worker)}")
        return f"Error aggregating County {county_id_to_process} ({mode} mode): {e_worker}"

def get_variables_config_for_mode(mode: str) -> Dict[str, Dict[str, Any]]:
    """Get variable configuration for specific mode"""
    
    if mode == "rolling":
        # Rolling mode: process prcp and Deficit
        return {
            # Processing Deficit from indices (direct zarr structure) - rolling only
            'Deficit': {
                'method': 'sum',
                'source_root': '/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/daymet_pet_deficit',
                'source_type': 'daymet_prcp',
                'data_structure': 'direct_zarr',
                'consolidated_open': False,
                'output_mode': 'a',
                'skip_logic': 'check_var_exists',
                'chunk_ref_var': 'Deficit',
                'target_spatial_chunks': {'y': 12, 'x': 12},
            },
            # Processing precipitation data (nested zarr structure) - both modes
            'prcp': {
                'method': 'sum',
                'source_root': '/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/daymet_cleaned',
                'source_type': 'daymet_prcp',
                'data_structure': 'direct_zarr',
                'consolidated_open': False,
                'output_mode': 'a',
                'skip_logic': 'check_var_exists',
                'chunk_ref_var': 'prcp',
                'target_spatial_chunks': {'y': 12, 'x': 12},
            }
        }
    
    elif mode == "standard":
        # Standard mode: process prcp and pet
        return {
            # Processing PET from indices (direct zarr structure) - standard only
            'pet': {
                'method': 'sum',
                'source_root': '/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/daymet_pet_deficit',
                'source_type': 'daymet_prcp',
                'data_structure': 'direct_zarr',
                'consolidated_open': False,
                'output_mode': 'a',
                'skip_logic': 'check_var_exists',
                'chunk_ref_var': 'pet',
                'target_spatial_chunks': {'y': 12, 'x': 12},
            },
            # Processing precipitation data (nested zarr structure) - both modes
            'prcp': {
                'method': 'sum',
                'source_root': '/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/daymet_cleaned',
                'source_type': 'daymet_prcp',
                'data_structure': 'direct_zarr',
                'consolidated_open': False,
                'output_mode': 'a',
                'skip_logic': 'check_var_exists',
                'chunk_ref_var': 'prcp',
                'target_spatial_chunks': {'y': 12, 'x': 12},
            }
        }
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    print("Starting Unified Temporal Aggregation Script (Multi-Period Support)...")
    start_time_total = time.time()

    # ==================== CONFIGURATION SECTION ====================
    
    # MULTI-MODE CONFIGURATION - Process multiple modes in one run
    MODES_TO_PROCESS = ["standard"]  # Options: "rolling", "standard"

    # AGGREGATION PARAMETERS - Now supports multiple periods
    ROLLING_AGGREGATION_PERIODS = ["1M","3M","6M","12M","24M"]
    ROLLING_OUTPUT_RESOLUTION = "1W"
    
    STANDARD_AGGREGATION_PERIODS = ["1W"]
    
    # TARGET COUNTIES CONFIGURATION - Manual county selection for testing/validation
    # If None or empty, will use index-based selection from GeoJSON
    # Example: ["01003", "01019"] to process only these specific counties
    TARGET_COUNTIES: Optional[List[str]] = ["01003","01043","06077","08011","13023","13037","16019","17005","17137","18003","18041","18117","18157","19049","19081","19113","20049","20201","20203","21035","21199","22081","24017","28107","29049","29143","31015","31059","34019","37057","37081","37107","40149","42081","46045","46057","46087","48095","55013","55027","05095","05107","17131","19095","19171","24037","29207","31159","36067","39023","48057","55055","39159",]
    #TARGET_COUNTIES: Optional[List[str]] = None  # Set to None to use index-based selection

    # UNIFIED OUTPUT TEMPLATE CONFIGURATION
    def _generate_resolution_label(period: str) -> str:
        """Generate resolution label for folder naming"""
        period_upper = period.upper()
        if 'D' in period_upper:
            days = re.search(r'(\d+)D', period_upper)
            return f"{days.group(1)}day" if days else "1day"
        elif 'W' in period_upper:
            weeks = re.search(r'(\d+)W', period_upper)
            return f"{weeks.group(1)}week" if weeks else "1week"
        elif 'M' in period_upper:
            months = re.search(r'(\d+)M', period_upper)
            return f"{months.group(1)}month" if months else "1month"
        return period.lower()

    resolution_label = _generate_resolution_label(STANDARD_AGGREGATION_PERIODS[0])
    OUTPUT_AGGREGATED_ROOT_TEMPLATE = f"/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/indices_from_daymet_standard_subset_{resolution_label}"
    
    # Get all source roots for the aggregator (still needed to find data)
    sample_variables_config = get_variables_config_for_mode("standard")
    ALL_SOURCE_ROOTS = list(set(var_config['source_root'] for var_config in sample_variables_config.values()))
    
    # PROCESSING RANGE CONFIGURATION (now applies to the ordered GeoJSON list)
    COUNTY_PROCESSING_START_INDEX: Optional[int] = 0
    COUNTY_PROCESSING_END_INDEX: Optional[int] = 2500 
    MAX_SEGMENTS_PER_COUNTY_FOR_AGG: Optional[int] = None
    # Set to None to auto-detect from input data (prcp/pet/Deficit chunk size)
    DEFAULT_OUTPUT_TIME_CHUNK_SIZE: Optional[int] = None

    # ==================== VALIDATION SECTION ====================
    
    valid_modes = ["rolling", "standard"]
    for mode in MODES_TO_PROCESS:
        if mode not in valid_modes:
            print(f"Error: Invalid mode '{mode}'. Valid modes are: {valid_modes}")
            exit()
    
    if "rolling" in MODES_TO_PROCESS and ROLLING_OUTPUT_RESOLUTION is None:
        print("Error: ROLLING_OUTPUT_RESOLUTION must be specified when using rolling mode")
        exit()
    
    if "rolling" in MODES_TO_PROCESS and not ROLLING_AGGREGATION_PERIODS:
        print("Error: ROLLING_AGGREGATION_PERIODS cannot be empty when using rolling mode")
        exit()
    
    if "standard" in MODES_TO_PROCESS and not STANDARD_AGGREGATION_PERIODS:
        print("Error: STANDARD_AGGREGATION_PERIODS cannot be empty when using standard mode")
        exit()
    
    print(f"\n=== MULTI-PERIOD CONFIGURATION SUMMARY ===")
    print(f"Modes to Process: {MODES_TO_PROCESS}")
    if "rolling" in MODES_TO_PROCESS:
        print(f"Rolling Config: Windows={ROLLING_AGGREGATION_PERIODS}, Output={ROLLING_OUTPUT_RESOLUTION}")
    if "standard" in MODES_TO_PROCESS:
        print(f"Standard Config: Periods={STANDARD_AGGREGATION_PERIODS}")
    print(f"Unified Output Root: {OUTPUT_AGGREGATED_ROOT_TEMPLATE}")
    print(f"Variables: {list(sample_variables_config.keys())}")
    print(f"Source Roots for Data Lookup: {ALL_SOURCE_ROOTS}")
    print(f"Processing Counties by GeoJSON index: {COUNTY_PROCESSING_START_INDEX} to {COUNTY_PROCESSING_END_INDEX}")
    print(f"============================================\n")

    # Setup multiprocessing
    num_logical_cores = os.cpu_count()
    if num_logical_cores is None: 
        num_logical_cores = 4
    # n_processes = max(1, num_logical_cores - 2 if num_logical_cores > 4 else 1)
    n_processes = num_logical_cores - 20  # You can adjust this as needed
    print(f"Attempting to use {n_processes} worker processes for multiprocessing.")

    try:
        # ==================== COUNTY LISTING SECTION (NEW METHOD) ====================
        # Get county codes from GeoJSON file
        
        # Current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        # Construct the path to the GeoJSON file relative to the script directory
        # geojson_file_path = os.path.normpath(os.path.join(script_dir, "../../county_list.geojson"))
        geojson_file_path = os.path.normpath(os.path.join(script_dir, "../county_list.geojson"))

        all_available_county_codes = get_county_codes_from_geojson(geojson_file_path)

        # Check if county codes were successfully loaded
        if not all_available_county_codes:
            print(f"No county codes loaded from GeoJSON. Exiting.")
            exit()
            
        print(f"Total unique counties loaded from file: {len(all_available_county_codes)}")

        # Select counties to process - Priority: TARGET_COUNTIES > Index range
        selected_county_codes_for_processing = []
        
        if TARGET_COUNTIES is not None and len(TARGET_COUNTIES) > 0:
            # Use manually specified target counties
            selected_county_codes_for_processing = TARGET_COUNTIES
            print(f"\n*** Using TARGET_COUNTIES configuration ***")
            print(f"Manually selected {len(selected_county_codes_for_processing)} counties: {selected_county_codes_for_processing}")
            
            # Validate that specified counties exist in GeoJSON
            invalid_counties = [c for c in TARGET_COUNTIES if c not in all_available_county_codes]
            if invalid_counties:
                print(f"WARNING: The following counties in TARGET_COUNTIES were not found in GeoJSON: {invalid_counties}")
                print(f"Available counties: {len(all_available_county_codes)} total")
        else:
            # Use index-based selection from GeoJSON
            if COUNTY_PROCESSING_START_INDEX is None and COUNTY_PROCESSING_END_INDEX is None:
                selected_county_codes_for_processing = all_available_county_codes
                print(f"Processing all {len(all_available_county_codes)} available counties from GeoJSON list.")
            else:
                start = COUNTY_PROCESSING_START_INDEX if COUNTY_PROCESSING_START_INDEX is not None else 0
                if start < 0: 
                    start = 0

                # Python slicing does not include the end index, so add 1
                end_slice = len(all_available_county_codes)
                if COUNTY_PROCESSING_END_INDEX is not None:
                    end_slice = COUNTY_PROCESSING_END_INDEX + 1
                    
                if start >= len(all_available_county_codes) or start >= end_slice:
                    print(f"Warning: Invalid county range (Start index: {COUNTY_PROCESSING_START_INDEX}, End index: {COUNTY_PROCESSING_END_INDEX}). No counties selected.")
                else:
                    selected_county_codes_for_processing = all_available_county_codes[start:end_slice]
                    actual_end_index_processed = start + len(selected_county_codes_for_processing) - 1
                    print(f"Selected {len(selected_county_codes_for_processing)} counties from GeoJSON by index range [{start}-{actual_end_index_processed}].")

        if not selected_county_codes_for_processing:
            print("No counties selected for aggregation. Exiting.")
            exit()
            
        print(f"Will process these counties: {selected_county_codes_for_processing if len(selected_county_codes_for_processing) < 10 else str(len(selected_county_codes_for_processing)) + ' counties'}")

        # ==================== MULTI-PERIOD PROCESSING SECTION ====================
        
        overall_success_count = 0
        overall_error_count = 0
        
        for mode_index, mode in enumerate(MODES_TO_PROCESS):
            print(f"\n{'='*80}")
            print(f"PROCESSING MODE {mode_index + 1}/{len(MODES_TO_PROCESS)}: {mode.upper()}")
            print(f"{'='*80}")
            
            if mode == "rolling":
                periods_to_process = ROLLING_AGGREGATION_PERIODS
                output_resolution = ROLLING_OUTPUT_RESOLUTION
            else: # standard
                periods_to_process = STANDARD_AGGREGATION_PERIODS
                output_resolution = None
            
            for period_index, agg_period in enumerate(periods_to_process):
                print(f"\n{'-'*60}")
                print(f"PROCESSING {mode.upper()} MODE - PERIOD {period_index + 1}/{len(periods_to_process)}: {agg_period}")
                print(f"{'-'*60}")
                
                output_template = OUTPUT_AGGREGATED_ROOT_TEMPLATE
                print(f"Output folder: {output_template}")
                
                variables_config = get_variables_config_for_mode(mode)
                
                agg_common_config_for_workers = {
                    'output_template': output_template,
                    'agg_period': agg_period,
                    'variables_config': variables_config,
                    'max_segments_per_county': MAX_SEGMENTS_PER_COUNTY_FOR_AGG,
                    'time_chunk_size_default': DEFAULT_OUTPUT_TIME_CHUNK_SIZE,
                    'mode': mode,
                    'output_resolution': output_resolution
                }
                
                tasks_args_list = [(county_code, agg_common_config_for_workers) for county_code in selected_county_codes_for_processing]
                
                if tasks_args_list:
                    print(f"Submitting {len(tasks_args_list)} county tasks for {mode} mode ({agg_period}) to multiprocessing pool ({n_processes} processes)...")
                    
                    period_start_time = time.time()
                    with multiprocessing.Pool(processes=n_processes) as pool:
                        results = pool.map(worker_aggregate_county, tasks_args_list)
                    period_end_time = time.time()
                    
                    print(f"\nMultiprocessing pool for {mode} mode ({agg_period}) finished. Results:")
                    period_success_count = 0
                    period_error_count = 0
                    for i, res_message in enumerate(results):
                        county_tasked = tasks_args_list[i][0]
                        print(f"  County {county_tasked}: {res_message}")
                        if "Successfully" in res_message: 
                            period_success_count += 1
                        else: 
                            period_error_count += 1
                    
                    overall_success_count += period_success_count
                    overall_error_count += period_error_count
                    
                    period_duration = period_end_time - period_start_time
                    print(f"\n{mode.upper()} Mode ({agg_period}) Summary: {period_success_count} counties processed successfully, {period_error_count} encountered errors.")
                    print(f"{mode.upper()} Mode ({agg_period}) Duration: {period_duration/60:.2f} minutes")
                else:
                    print(f"No aggregation tasks were generated for {mode} mode ({agg_period}).")
        
        print(f"\n{'='*80}")
        print(f"OVERALL MULTI-PERIOD PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"Total Success: {overall_success_count} county-mode-period combinations")
        print(f"Total Errors: {overall_error_count} county-mode-period combinations")
        print(f"Counties Processed: {len(selected_county_codes_for_processing)}")
        print(f"Output Location: {OUTPUT_AGGREGATED_ROOT_TEMPLATE}")
            
    except Exception as e_main:
        print(f"An error occurred in the main aggregation script: {e_main}")
        traceback.print_exc()
    finally:
        end_time_total = time.time()
        total_duration_seconds = end_time_total - start_time_total
        total_duration_hours = total_duration_seconds / 3600
        print(f"\nTotal multi-period aggregation pipeline execution time: {total_duration_hours:.2f} hours ({total_duration_seconds:.2f} seconds).")
        print("Unified Multi-Period Temporal Aggregation Script finished.")
#%%