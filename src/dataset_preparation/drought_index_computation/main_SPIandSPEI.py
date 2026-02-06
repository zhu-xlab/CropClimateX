"""
Docstring for main_SPIandSPEI
Main script to compute SPI and SPEI indices using parallel processing.
This script needs the DataCalculator class from Calculator_SPIandSPEI.py. Please make sure the name and path are correct.
Some variables "ndvi, evi" are reserved for MODIS indices, related functions have not been used, please ignore them.
"""         
#%%
import os
import traceback
import re
import glob
from collections import defaultdict
from typing import Dict, List, Union, Optional, Tuple
import time
import multiprocessing # Multiprocessing library
from tqdm import tqdm # Progress bar library

try:
    # Ensure the filename "Calculator_SPIandSPEI.py" (or your actual filename) is correct
    # and the file is in the same directory or Python path.
    from Calculator_SPIandSPEI import DataCalculator 
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import DataCalculator: {e}")
    print("Please ensure the DataCalculator class file (e.g., Calculator_SPIandSPEI.py) is in the same directory or accessible in PYTHONPATH.")
    exit()

# --- Helper Functions ---
def extract_counties_from_input_source(input_source_root: str) -> Dict[str, List[str]]:
    """
    Scans the input source directory to find all available counties and their segments.
    Returns a dict mapping county FIPS codes to lists of segment paths.
    
    This function is similar to get_all_segment_paths_for_county in main_RWDandRSM.py,
    but discovers all counties first from the input source folder.
    """
    county_segments = defaultdict(list)
    
    if not os.path.isdir(input_source_root):
        print(f"Warning: Input source directory not found: {input_source_root}")
        return dict(county_segments)
    
    try:
        for folder_name in os.listdir(input_source_root):
            folder_path = os.path.join(input_source_root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            # Match pattern: daymet_FIPS_START-END.zarr (e.g., daymet_01003_0-9.zarr)
            match = re.match(r"^daymet_(\d{5})_(\d+-\d+)\.zarr$", folder_name)
            if match:
                county_fips = match.group(1)
                county_segments[county_fips].append(folder_path)
        
        # Sort segments for each county by start number
        for county_fips in county_segments:
            county_segments[county_fips].sort(key=lambda p: int(re.search(r"_(\d+)-\d+\.zarr$", p).group(1)))
                
    except Exception as e:
        print(f"Error scanning input source directory '{input_source_root}': {e}")
    
    return dict(county_segments)

def extract_county_segments(base_root_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Scans the base_root_dir for raw data Zarr segments (daymet, soil, modis)
    and organizes them by county FIPS code and data type, searching recursively.
    
    NOTE: This function is now DEPRECATED for SPI/SPEI processing.
    Use extract_counties_from_input_source() instead to scan the actual input source folder.
    """
    county_data = defaultdict(lambda: defaultdict(list))
    data_types_expected = ["daymet"] 

    for dtype in data_types_expected:
        dtype_base_path = os.path.join(base_root_dir, dtype) # e.g., D:/CropClimateX/daymet
        if not os.path.isdir(dtype_base_path):
            print(f"Warning: Data type directory for raw data not found, skipping: {dtype_base_path}")
            continue
        
        # --- MODIFIED GLOB PATTERN FOR RECURSIVE SEARCH ---
        # This pattern will search in dtype_base_path and all its subdirectories
        # for directories ending in .zarr that start with the dtype.
        # The f"{dtype}_*_*.zarr" is a general pattern; the regex below will do the strict FIPS and range check.
        search_pattern = os.path.join(dtype_base_path, '**', f"{dtype}_*.zarr") # Use wildcard for FIPS and range part for glob
        
        # Use recursive=True to enable '**' functionality
        potential_zarr_stores = glob.glob(search_pattern, recursive=True)
        # --- END OF MODIFICATION ---

        if not potential_zarr_stores:
            print(f"Info: No Zarr stores found matching pattern '{search_pattern}' under '{dtype_base_path}' (recursive).")

        for zarr_root_path in potential_zarr_stores:
            if not os.path.isdir(zarr_root_path): # A Zarr store is a directory
                continue # Should not happen if glob found it as such, but good check
            
            segment_full_name = os.path.basename(zarr_root_path)
            # Strict pattern match: dtype_FIPS(5digits)_START-END.zarr
            strict_match = re.match(rf"^{re.escape(dtype)}_(\d{{5}})_(\d+-\d+)\.zarr$", segment_full_name)
            
            if strict_match:
                county_code, _ = strict_match.groups() # Group 1 is FIPS, Group 2 is range
                county_data[county_code][dtype].append(zarr_root_path)
                # print(f"  Found and matched: {zarr_root_path}") # Uncomment for verbose success logging
            # else:
                # Uncomment for debugging paths that glob found but regex didn't match
                # print(f"Warning: Glob found '{zarr_root_path}', but its basename '{segment_full_name}' does not match expected REGEX pattern for RAW dtype '{dtype}'. Example: {dtype}_12345_0-9.zarr. Skipping.")
    
    if not county_data:
        print(f"Warning: extract_county_segments finished but found no matching data for any county under {base_root_dir} with expected patterns.")
    return dict(county_data)

def count_cubes(zarr_root_path: str) -> int:
    """Counts the number of numerical subdirectories (cubes) within a Zarr segment store."""
    try:
        if not os.path.exists(zarr_root_path) or not os.path.isdir(zarr_root_path):
            return 0
        # Cube IDs are purely numerical directory names
        subdirectories = [d for d in os.listdir(zarr_root_path)
                          if d.isdigit() and os.path.isdir(os.path.join(zarr_root_path, d))]
        return len(subdirectories)
    except Exception as e:
        # print(f"Warning: Error counting cubes in '{zarr_root_path}': {str(e)}")
        return 0

# --- Main Segment Processing Worker Function (for multiprocessing) ---
def process_county_segment_worker(
    args_tuple: Tuple 
) -> Tuple[str, str, str]: # Returns (county_code, log_segment_name, status)
    """
    Worker function to process all index calculations for a single data segment of a county.
    Called by multiprocessing.Pool.
    """
    # Unpack arguments at the beginning of the function - added SPEI fitting method parameter
    county_code, segment_info, cropclimatex_root_dir, \
    daymet_target_vars, modis_target_vars, soil_target_vars, \
    daymet_derived_subfolder_override, \
    time_scale_spi_config, time_scale_spei_config, skip_nan_calculations, \
    spi_spei_input_source_subfolder, spi_spei_use_derived_inputs, \
    use_seasonal_method, spei_fitting_method = args_tuple  # Added SPEI fitting method parameter

    # Convert single values to lists for uniform processing
    spi_time_scales = time_scale_spi_config if isinstance(time_scale_spi_config, list) else [time_scale_spi_config]
    spei_time_scales = time_scale_spei_config if isinstance(time_scale_spei_config, list) else [time_scale_spei_config]

    # Determine a representative name for logging this segment
    representative_segment_path = next(iter(segment_info.values()), None)
    log_segment_name = "UnknownSegment" # Default value
    if representative_segment_path:
        base_name = os.path.basename(representative_segment_path)
        segment_identifier_match = re.search(r"(\d+-\d+)\.zarr$", base_name) # Extracts range like "0-9"
        segment_range_str = segment_identifier_match.group(1) if segment_identifier_match else "UnknownRange"
        log_segment_name = f"{county_code}_{segment_range_str}" 
    
    # print(f"\nWorker PID {os.getpid()}: Starting County: {county_code}, Segment: {log_segment_name}") # Can be noisy with many workers

    active_dtypes_in_segment = list(segment_info.keys()) # e.g., ['daymet', 'soil']
    daymet_vars_effective = list(daymet_target_vars) if daymet_target_vars is not None else None
    modis_vars_effective = list(modis_target_vars) if modis_target_vars is not None else None
    soil_vars_effective = list(soil_target_vars) if soil_target_vars is not None else None

    # If Daymet variables are targeted but no Daymet data for this segment, clear Daymet targets
    if (daymet_vars_effective and 'daymet' not in active_dtypes_in_segment):
        daymet_vars_effective = [] 
    # If PHDI/PDSI (Daymet vars) are targeted but no Soil data, allow calculation (DataCalculator handles missing WHC)
    if (daymet_vars_effective and any(v in daymet_vars_effective for v in ['phdi','pdsi']) and 'soil' not in active_dtypes_in_segment):
        pass 
    
    # Map of data types to number of cubes for this specific segment
    num_cubes_map = {}
    for dtype_raw, zarr_path_raw in segment_info.items():
        if dtype_raw in ["daymet", "soil", "modis"]:
            current_cubes = count_cubes(zarr_path_raw)
            num_cubes_map[dtype_raw] = current_cubes
            
            # Check if this data type is actually required for any targeted variable
            is_dtype_required = False
            if dtype_raw == 'daymet' and daymet_vars_effective: is_dtype_required = True
            if dtype_raw == 'soil' and (soil_vars_effective or (daymet_vars_effective and any(v in daymet_vars_effective for v in ['phdi','pdsi']))): is_dtype_required = True
            if dtype_raw == 'modis' and modis_vars_effective: is_dtype_required = True
            
            # If required but no cubes, clear the effective targets for that type
            if current_cubes == 0 and is_dtype_required:
                # print(f"Info for {log_segment_name}: Data type '{dtype_raw}' is required but has 0 cubes. Clearing its target variables.")
                if dtype_raw == 'daymet': daymet_vars_effective = []
                if dtype_raw == 'modis': modis_vars_effective = []
                if dtype_raw == 'soil': soil_vars_effective = [] # PHDI/PDSI might still be attempted if daymet_vars_effective still contains them
        else:
            num_cubes_map[dtype_raw] = 0 # Should not happen if segment_info is clean
    
    # If no data types in segment_info or all have 0 cubes, but targets were specified
    if not num_cubes_map: 
        is_any_target_specified = bool(daymet_vars_effective or modis_vars_effective or soil_vars_effective)
        if is_any_target_specified:
            return county_code, log_segment_name, "Error: No processable data types with cube counts in segment_info, but targets were specified."
        else: # No targets, so it's fine if there's no data.
            return county_code, log_segment_name, "Success (No targets for this segment)"

    try:
        calculator = DataCalculator(
            segment_info,  # Base paths for this specific segment
            num_cubes_map, # Cube counts for this specific segment
            cropclimatex_root_dir,
            daymet_indices_subfolder=daymet_derived_subfolder_override, # Control output subfolder for Daymet indices
            skip_nan_calculations=skip_nan_calculations,  # Pass NaN calculation control parameter
            # SPI/SPEI input data source configuration
            spi_spei_input_source_subfolder=spi_spei_input_source_subfolder,
            spi_spei_use_derived_inputs=spi_spei_use_derived_inputs,
            # Seasonal method configuration
            use_seasonal_method=use_seasonal_method,
            # SPEI fitting method configuration
            spei_fitting_method=spei_fitting_method
        )

        # Process Soil-derived indices
        if soil_vars_effective: 
            if "soil" in segment_info and num_cubes_map.get("soil", 0) > 0:
                # print(f"Worker for {log_segment_name}: Processing soil...") # Reduce verbosity
                calculator.process_soil_derived_indices(target_variables=soil_vars_effective)
        
        # Process Daymet-derived indices for multiple time scales
        if daymet_vars_effective: 
            if "daymet" in segment_info and num_cubes_map.get("daymet", 0) > 0:
                # Check if SPI or SPEI are in the target variables
                has_spi = any(var in ['spi'] or var.startswith('spi-') for var in daymet_vars_effective)
                has_spei = any(var in ['spei'] or var.startswith('spei-') for var in daymet_vars_effective)
                has_other_vars = any(var not in ['spi', 'spei'] and not var.startswith('spi-') and not var.startswith('spei-') for var in daymet_vars_effective)
                
                # Process other variables (non-SPI/SPEI) first with default time scales
                if has_other_vars:
                    other_vars = [var for var in daymet_vars_effective 
                                 if var not in ['spi', 'spei'] and not var.startswith('spi-') and not var.startswith('spei-')]
                    print(f"Worker for {log_segment_name}: Processing other Daymet variables: {other_vars}")
                    calculator.process_daymet_derived_indices(
                        time_scale_spi=1,    # Default for other variables
                        time_scale_spei=1,   # Default for other variables
                        target_variables=other_vars
                    )
                
                # Process SPI and SPEI together for all time scales in one call
                if has_spi or has_spei:
                    spi_spei_vars = []
                    if has_spi:
                        spi_spei_vars.append('spi')
                    if has_spei:
                        spi_spei_vars.append('spei')
                    
                    print(f"Worker for {log_segment_name}: Processing SPI/SPEI for all time scales: SPI{spi_time_scales}, SPEI{spei_time_scales}")
                    calculator.process_daymet_derived_indices_multiple_timescales(
                        spi_time_scales=spi_time_scales,
                        spei_time_scales=spei_time_scales,
                        target_variables=spi_spei_vars
                    )

        # Process MODIS-derived indices
        if modis_vars_effective: 
            if "modis" in segment_info and num_cubes_map.get("modis", 0) > 0:
                # print(f"Worker for {log_segment_name}: Processing modis...") # Reduce verbosity
                calculator.process_modis_derived_indices(target_variables=modis_vars_effective)

        return county_code, log_segment_name, "Success"
    except Exception as e_process_seg:
        error_msg = f"Failed segment {log_segment_name} (County: {county_code}): {str(e_process_seg)}"
        # print(f"Worker PID {os.getpid()}: {error_msg}") # Print error from worker
        # traceback.print_exc() # Printing full traceback in multiprocessing can be messy.
        return county_code, log_segment_name, error_msg

# --- Main Execution Logic (Using Parallel Mode) ---
if __name__ == '__main__':
    multiprocessing.freeze_support() # Important for Windows compatibility

    print("Starting data processing pipeline (PARALLEL MODE)...")

    # --- Configure Variables to Compute ---
    # List the Daymet-derived indices you want to compute.
    # Example: ['spi', 'spei', 'pet', 'Deficit', 'prcp']
    # If you want SPI-3, still list 'spi' here; the time scale is set below.
    DAYMET_VARIABLES_TO_COMPUTE = ['spei','spi'] 
    MODIS_VARIABLES_TO_COMPUTE = []  # e.g., ['NDVI', 'EVI'] or empty list []
    SOIL_VARIABLES_TO_COMPUTE = []   # e.g., ['whc'] or empty list []

    # --- Skip NaN computation ---
    SKIP_NAN_CALCULATIONS = True  # 
    # --- End NaN Calculation Control ---

    # --- New: Configure Seasonal Method for SPI/SPEI ---
    # If True, use seasonal method (fit distributions separately for each week of year)
    # If False, use non-seasonal method (fit single distribution for entire time series)
    USE_SEASONAL_METHOD = True  # Recommended for long time series data (2000-2022)
    # --- End Seasonal Method Configuration ---
    
    # --- New: Configure SPEI Fitting Method ---
    # Choose fitting method for SPEI calculation:
    # 'lmoments_glo' - L-Moments with Generalized Logistic distribution (recommended, more robust)
    # 'logistic' - Scipy MLE with Logistic distribution (traditional method)
    SPEI_FITTING_METHOD = 'lmoments_glo'  # Default: lmoments_glo
    # --- End SPEI Fitting Method Configuration ---

    # --- Configure SPI/SPEI input data source ---
    # Specify the path for processed data to use for SPI/SPEI calculations (instead of raw daymet data)
    #SPI_SPEI_INPUT_SOURCE_SUBFOLDER = "/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/indices_from_daymet_unified_7day/" # Read Deficit and prcp from this subfolder
    SPI_SPEI_INPUT_SOURCE_SUBFOLDER = "/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/indices_from_daymet_allset_unified_revised_1week_right/" # Example: "indices_from_daymet_unified_1week_bycube"
    SPI_SPEI_USE_DERIVED_INPUTS = True  # True=use processed data, False=use raw daymet data
    # --- End SPI/SPEI input data source configuration ---

    # Subfolder name for saving Daymet-derived indices (under CROPCLIMATEX_ROOT_DIR/)
    DAYMET_DERIVED_SUBFOLDER_CONFIG = "indices_from_daymet_SPIandSPEI_allset_right"
    #DAYMET_DERIVED_SUBFOLDER_CONFIG = "/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/indices_from_daymet_SPIandSPEI" 
                                    # Example: "indices_from_daymet_7day"
                                    # Note: If you want SPI/SPEI for different time scales in different subfolders,
                                    # you might run this script multiple times with different configurations,
                                    # or use the folder naming logic in DataCalculator if processing a single target.

    # --- Configure Time Scales for SPI and SPEI ---
    # Set these to the desired time scale (e.g., in months for monthly data, or corresponding time steps)
    # Example: Set to 3 for SPI-3 and SPEI-3.
    # Set to 1 for standard monthly SPI/SPEI (if input is monthly).
    TIME_SCALE_SPI_CONFIG = [1,3,6,12,24]  # Multiple time scales for SPI
    TIME_SCALE_SPEI_CONFIG = [1,3,6,12,24]  # Multiple time scales for SPEI
    # --- End Time Scale Configuration ---
    
    # --- Configure County Processing Range ---
    COUNTY_PROCESSING_START_ID = None # Example: "01003" (starting) or None to start from beginning
    COUNTY_PROCESSING_START_INDEX = 0   # Start index for counties (0-based), used if START_ID is None or invalid
    COUNTY_PROCESSING_END_INDEX = 2500  # End index for counties (inclusive), None to process till the end. Set to 1 for first two counties.

    # --- Configure Number of Parallel Processes ---
    # Leaves 2 cores for the system by default if more than 2 cores available
    # NUM_PARALLEL_PROCESSES = max(1, os.cpu_count() - 5 if os.cpu_count() and os.cpu_count() > 2 else 1)
    NUM_PARALLEL_PROCESSES = os.cpu_count()-5 # For single-process debugging

    start_time_total = time.time()

    try:
        cropclimatex_root_dir = "/home/zhiyuan/Codes/CropClimateX-main/CropClimateX"

        if not os.path.isdir(cropclimatex_root_dir):
            raise FileNotFoundError(f"Error: CropClimateX root directory not found at: {cropclimatex_root_dir}")

        # Build full path to input source directory
        input_source_full_path = os.path.join(cropclimatex_root_dir, SPI_SPEI_INPUT_SOURCE_SUBFOLDER)
        
        print(f"Scanning INPUT SOURCE for available counties: {input_source_full_path}...")
        print(f"  (Using processed data instead of raw data for better reliability)")
        
        # Use new function to extract counties directly from input source
        county_segments_from_input = extract_counties_from_input_source(input_source_full_path)

        if not county_segments_from_input:
            raise ValueError(f"CRITICAL: No valid county segments found in input source: {input_source_full_path}")

        sorted_county_codes = sorted(county_segments_from_input.keys())
        print(f"Scan complete. Found {len(sorted_county_codes)} unique counties in input source.")

        if not sorted_county_codes:
            print("No counties found. Exiting.")
            exit()
            
        # --- County Selection Logic ---
        actual_start_index = 0 
        if COUNTY_PROCESSING_START_ID is not None:
            try:
                start_id_str = str(COUNTY_PROCESSING_START_ID) # Ensure it's a string for comparison
                actual_start_index = sorted_county_codes.index(start_id_str)
                print(f"Found specified start county ID '{start_id_str}' at index {actual_start_index}.")
            except ValueError:
                print(f"Warning: Specified COUNTY_PROCESSING_START_ID '{COUNTY_PROCESSING_START_ID}' was not found in the list of discovered counties.")
                print(f"  Available counties start with: {sorted_county_codes[:5]} ... (total {len(sorted_county_codes)})")
                if COUNTY_PROCESSING_START_INDEX is not None:
                    actual_start_index = COUNTY_PROCESSING_START_INDEX
                    print(f"  Falling back to COUNTY_PROCESSING_START_INDEX: {actual_start_index}.")
                else:
                    print(f"  Proceeding from the beginning (index 0).")
                    actual_start_index = 0 
        elif COUNTY_PROCESSING_START_INDEX is not None:
            actual_start_index = COUNTY_PROCESSING_START_INDEX
            print(f"Using specified start index: {actual_start_index}.")
        else:
            print("No specific start county ID or index provided. Starting from the beginning (index 0).")
        
        actual_start_index = max(0, int(actual_start_index)) # Ensure integer and non-negative
        if actual_start_index >= len(sorted_county_codes):
            print(f"Error: Start index {actual_start_index} is beyond the available counties (total: {len(sorted_county_codes)}). Exiting.")
            exit()
        
        actual_end_index = len(sorted_county_codes) - 1 # Default: process until the end
        if COUNTY_PROCESSING_END_INDEX is not None:
            actual_end_index = min(len(sorted_county_codes) - 1, int(COUNTY_PROCESSING_END_INDEX))
            print(f"Using specified end index: {actual_end_index}.")
        
        if actual_start_index > actual_end_index:
            print(f"Error: Start index {actual_start_index} > End index {actual_end_index}. No counties to process.")
            exit()
        
        # Select counties to process
        counties_to_process = sorted_county_codes[actual_start_index:actual_end_index + 1]
        print(f"Counties to process: {len(counties_to_process)} counties (indices {actual_start_index} to {actual_end_index})")
        print(f"  First few: {counties_to_process[:3]}")
        print(f"  Last few: {counties_to_process[-3:]}")
        
        # Prepare all segments (county-segment combinations) for processing
        all_segments_to_process = []
        
        for county_code in counties_to_process:
            # Get segment paths from input source for this county
            county_segment_paths = county_segments_from_input.get(county_code, [])
            
            if not county_segment_paths:
                print(f"Warning: No segments found for county {county_code} in input source. Skipping.")
                continue
            
            # Process each segment for this county
            for segment_path in county_segment_paths:
                segment_basename = os.path.basename(segment_path)
                
                # Create segment_info dict (only 'daymet' since we're reading from processed data)
                segment_info = {'daymet': segment_path}
                
                # Create processing arguments for this segment
                args_tuple = (
                    county_code,
                    segment_info,
                    cropclimatex_root_dir,
                    DAYMET_VARIABLES_TO_COMPUTE,
                    MODIS_VARIABLES_TO_COMPUTE,
                    SOIL_VARIABLES_TO_COMPUTE,
                    DAYMET_DERIVED_SUBFOLDER_CONFIG,
                    TIME_SCALE_SPI_CONFIG,
                    TIME_SCALE_SPEI_CONFIG,
                    SKIP_NAN_CALCULATIONS,
                    SPI_SPEI_INPUT_SOURCE_SUBFOLDER,
                    SPI_SPEI_USE_DERIVED_INPUTS,
                    USE_SEASONAL_METHOD,  # Seasonal method parameter
                    SPEI_FITTING_METHOD  # SPEI fitting method parameter
                )
                all_segments_to_process.append(args_tuple)
        
        if not all_segments_to_process:
            print("No segments to process. Exiting.")
            exit()
        
        print(f"Total segments to process: {len(all_segments_to_process)}")
        print(f"Using {NUM_PARALLEL_PROCESSES} parallel processes")
        
        # Process segments in parallel
        successful_segments = 0
        failed_segments = 0
        
        if NUM_PARALLEL_PROCESSES == 1:
            # Single-process mode for debugging
            print("Running in single-process mode...")
            for args_tuple in tqdm(all_segments_to_process, desc="Processing segments"):
                county_code, log_segment_name, status = process_county_segment_worker(args_tuple)
                if status == "Success" or status.startswith("Success"):
                    successful_segments += 1
                else:
                    failed_segments += 1
                    print(f"Failed: {county_code}_{log_segment_name}: {status}")
        else:
            # Multi-process mode
            print("Running in multi-process mode...")
            try:
                with multiprocessing.Pool(processes=NUM_PARALLEL_PROCESSES) as pool:
                    results = []
                    
                    # Submit all tasks
                    for args_tuple in all_segments_to_process:
                        result = pool.apply_async(process_county_segment_worker, (args_tuple,))
                        results.append(result)
                    
                    # Collect results with progress bar
                    for result in tqdm(results, desc="Processing segments"):
                        try:
                            county_code, log_segment_name, status = result.get()  # timeout
                            if status == "Success" or status.startswith("Success"):
                                successful_segments += 1
                            else:
                                failed_segments += 1
                                print(f"Failed: {county_code}_{log_segment_name}: {status}")
                        except multiprocessing.TimeoutError:
                            failed_segments += 1
                            print(f"Timeout error for segment")
                        except Exception as e:
                            failed_segments += 1
                            print(f"Error processing segment: {e}")
            
            except Exception as e:
                print(f"Error in multiprocessing: {e}")
                traceback.print_exc()
        
        # Final summary
        total_time = time.time() - start_time_total
        print(f"\n" + "="*60)
        print(f"PROCESSING COMPLETE")
        print(f"="*60)
        print(f"Total segments processed: {successful_segments + failed_segments}")
        print(f"Successful: {successful_segments}")
        print(f"Failed: {failed_segments}")
        print(f"Success rate: {(successful_segments/(successful_segments + failed_segments)*100):.1f}%" if (successful_segments + failed_segments) > 0 else "N/A")
        print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        
        # Enhanced summary information
        method_type = "Seasonal (weekly)" if USE_SEASONAL_METHOD else "Non-seasonal"
        print(f"  - SPI/SPEI calculation method: {method_type}")
        print(f"  - SPEI fitting method: {SPEI_FITTING_METHOD}")
        print(f"  - SPI time scales: {TIME_SCALE_SPI_CONFIG}")
        print(f"  - SPEI time scales: {TIME_SCALE_SPEI_CONFIG}")
        if SPI_SPEI_USE_DERIVED_INPUTS:
            print(f"  - Input source subfolder: {SPI_SPEI_INPUT_SOURCE_SUBFOLDER}")
        print(f"  - Output subfolder: {DAYMET_DERIVED_SUBFOLDER_CONFIG}")
        print(f"  - Counties processed: {len(counties_to_process)} (indices {actual_start_index}-{actual_end_index})")
        print(f"  - Parallel processes: {NUM_PARALLEL_PROCESSES}")
        print(f"  - Skip NaN calculations: {SKIP_NAN_CALCULATIONS}")
        
        if failed_segments > 0:
            print(f"\nNote: {failed_segments} segments failed. Check the error messages above for details.")
        
        print(f"\nProcessing pipeline completed successfully!")
        
    except Exception as e:
        print(f"CRITICAL ERROR in main execution: {e}")
        traceback.print_exc()
        exit(1)
    
    finally:
        # Cleanup
        print(f"Cleaning up...")
#%%