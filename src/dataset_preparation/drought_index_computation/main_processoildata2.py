"""
Description: This script processes soil data cubes to compute Available Water Capacity (AWC)
using the Saxton & Rawls (2006) formula. It applies a strict spatial filling logic to handle missing data
and generates a detailed audit report, including targeted sampling of repaired cubes.
Because original soil data (sand, clay, soc) have NaN values at different locations, which means a pixel may be valid in sand but invalid in clay or soc.
Hence, the calculated AWC will also have NaNs scattered throughout the 12x12 cubes.
This will hinder the usage of AWC in calculating Palmer Drought Severity Index (PDSI) later on.
When assessing the drought, we won't say that a pixel is drought-free if its AWC or other soil data is NaN.
Furthermore, the PDSI and PHDI are not very sensitive to the exact value of AWC, as long as it is within a reasonable range (Palmer, 1965).
Therefore, filling missing AWC values with estimated values based on spatial context and global mean is acceptable.
Therefore, this script fills these NaNs using a two-step approach: first, a strict spatial fill based on neighboring pixels,
and second, a global mean fill for any remaining NaNs. The audit report highlights cubes that were repaired,
including matrix dumps for targeted samples.
The cubes which require filling are displayed in the report, along with their before-and-after matrices for transparency.
This result has been verified to be suitable for PDSI calculation.
"""
import xarray as xr
import numpy as np
import os
import glob
import random
from scipy.signal import convolve2d
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- Configuration ---
SOIL_ROOT_DIR = "/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/soil_downsampled"
AWC_OUTPUT_DIR = "/home/zhiyuan/Codes/CropClimateX-main/CropClimateX/awc_derived"
REPORT_FILE = "/home/zhiyuan/Codes/CropClimateX-main/awc_fill_audit_report.txt"
LUCKY_SAMPLE_PROBABILITY = 1.00  # Increased probability for targeted sampling 

MAX_PARALLEL_PROCESSES = max(1, os.cpu_count() - 80)

# --- AWC Formula ---
def calculate_awc_core(sand, clay, soc, assumed_soil_depth_mm=1500):
    """
    This function is usef for computing AWC from sand, clay, and SOC.
    The key reference is Saxton & Rawls (2006), some other similar functions can also be used.
    Available water capacity (AWC) is typically calculated as the difference between field capacity (theta33) and permanent wilting point (theta1500).
    """
    s = sand / 100.0
    c = clay / 100.0
    om = 1.724 * soc / 100.0 # Convert SOC to organic matter fraction
    # Reference for transfering soc to om: 
    # Phosphatase activities and available nutrients in soil aggregates affected by straw returning to a calcareous soil under the maize–wheat cropping system
    theta_33_t = -0.251 * s + 0.195 * c + 0.011 * om + 0.006 * s * om - 0.027 * c * om + 0.452 * s * c + 0.299
    theta_33 = theta_33_t + (1.283 * (theta_33_t ** 2) - 0.374 * theta_33_t - 0.015)
    theta_1500_t = -0.024 *s + 0.487 * c + 0.006 * om + 0.005 * s * om - 0.013 * c * om + 0.068 * s * c + 0.031
    theta_1500 = theta_1500_t + (0.14 * theta_1500_t - 0.02)
    awc = np.maximum(theta_33 - theta_1500, 0.0)  # Ensure AWC is not negative
    awc = awc * assumed_soil_depth_mm  # Convert to mm
    return awc

# --- Spatial Logic ---
def strict_spatial_fill(data_array):
    """
    Judging from the 8-neighborhood, fill missing values only when:
    - Center pixels: at least 4 valid neighbors
    - Edge pixels: at least 3 valid neighbors
    - Corner pixels: at least 2 valid neighbors
    Otherwise, keep as NaN.
    """
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    valid_mask = (~np.isnan(data_array)).astype(float)
    
    # Calculate number of valid neighbors by convolution
    actual_valid = convolve2d(valid_mask, kernel, mode='same', boundary='fill', fillvalue=0)
    ones_grid = np.ones_like(data_array)
    max_possible = convolve2d(ones_grid, kernel, mode='same', boundary='fill', fillvalue=0)
    
    data_zero = np.nan_to_num(data_array, nan=0.0)
    sum_neighbors = convolve2d(data_zero, kernel, mode='same', boundary='fill', fillvalue=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        local_mean = sum_neighbors / actual_valid
    
    is_missing = np.isnan(data_array)
    
    cond_center = (max_possible == 8) & (actual_valid >= 4)
    cond_edge   = (max_possible == 5) & (actual_valid >= 3)
    cond_corner = (max_possible == 3) & (actual_valid >= 2)
    
    fill_mask = is_missing & (cond_center | cond_edge | cond_corner)
    
    output = data_array.copy()
    output[fill_mask] = local_mean[fill_mask]
    
    return output

def format_matrix_string(arr, title):
    """Helper function: Format a matrix into a readable string"""
    # Set print options: 2 decimal places, NaN displayed as nan
    matrix_str = np.array2string(arr, precision=2, separator=', ', suppress_small=True, floatmode='fixed')
    return f"\n--- {title} ---\n{matrix_str}\n"

def process_single_cube_awc_audit(args):
    county_path, cube_name, output_segment_path = args
    
    county_name = os.path.basename(county_path)
    cube_id = int(cube_name)
    
    log_info = {
        "county_name": county_name,  # For sorting
        "cube_id": cube_id,  # For sorting
        "id": f"{county_name}/{cube_name}",
        "total": 144, # 12x12
        "raw_valid": 0,
        "step1_valid": 0,
        "step2_valid": 0,
        "zeros_final": 0,  # NEW: Count of zero values in final result
        "lucky_dump": None # if lucky, will hold the matrix dump string
    }

    try:
        ds = xr.open_zarr(county_path, group=cube_name, consolidated=False)
        sand = ds['sand'].values
        clay = ds['clay'].values
        soc = ds['soc'].values
        
        # 1. Calculate raw AWC
        valid_mask = (~np.isnan(sand)) & (~np.isnan(clay)) & (~np.isnan(soc))
        awc_raw = np.full_like(sand, np.nan)
        
        if np.any(valid_mask):
            awc_vals = calculate_awc_core(sand[valid_mask], clay[valid_mask], soc[valid_mask], assumed_soil_depth_mm=1500)
            awc_raw[valid_mask] = awc_vals
            
        log_info["raw_valid"] = np.sum(~np.isnan(awc_raw))
        
        # 2. Step 1: fill using strict spatial logic
        awc_step1 = strict_spatial_fill(awc_raw)
        log_info["step1_valid"] = np.sum(~np.isnan(awc_step1))
        
        # 3. Step 2: fill using global mean as fallback
        awc_final = awc_step1.copy()
        nans_remaining = np.isnan(awc_final)
        
        if np.any(nans_remaining):
            global_mean = np.nanmean(awc_final)
            if not np.isnan(global_mean):
                awc_final[nans_remaining] = global_mean
        
        # Physical clipping
        awc_final = np.clip(awc_final, 0, None)
        log_info["step2_valid"] = np.sum(~np.isnan(awc_final))
        
        # NEW: Count zeros in final result (non-NaN values that equal 0.0)
        log_info["zeros_final"] = np.sum((~np.isnan(awc_final)) & (awc_final == 0.0))

        # 4. MODIFIED: Targeted lucky sample - only if cube had NaNs in raw calculation
        is_lucky = False
        if log_info["raw_valid"] < 144:  # Only sample cubes that needed repair
            is_lucky = random.random() < LUCKY_SAMPLE_PROBABILITY
        
        if is_lucky:
            dump_str = f"\n>>> TARGETED SAMPLE MATRIX DUMP: {log_info['id']} <<<\n"
            dump_str += f"(Raw Valid: {log_info['raw_valid']}/144, showcasing repair process)\n"
            dump_str += format_matrix_string(awc_raw, "1. Raw AWC (Calculated)")
            dump_str += format_matrix_string(awc_step1, "2. After Spatial Fill (Corner>=2, Edge>=3, Center>=4)")
            dump_str += format_matrix_string(awc_final, "3. Final Result (After Global Mean Fill)")
            dump_str += "="*60 + "\n"
            log_info["lucky_dump"] = dump_str

        # 5. Save final AWC to Zarr
        coords = ds['sand'].coords
        ds_out = xr.Dataset(
            {"awc": (ds['sand'].dims, awc_final.astype(np.float32))},
            coords=coords
        )
        ds_out.to_zarr(output_segment_path, group=cube_name, mode='a', consolidated=False)
        
        return log_info

    except Exception as e:
        return {"error": True, "county_name": county_name, "cube_id": cube_id, 
                "id": f"{county_name}/{cube_name}", "message": str(e)}

def process_county_awc(county_path):
    county_name = os.path.basename(county_path)
    output_segment_path = os.path.join(AWC_OUTPUT_DIR, county_name)
    os.makedirs(output_segment_path, exist_ok=True)
    
    cube_names = [d for d in os.listdir(county_path) if os.path.isdir(os.path.join(county_path, d)) and d.isdigit()]
    
    results = []
    for cube_name in cube_names:
        res = process_single_cube_awc_audit((county_path, cube_name, output_segment_path))
        results.append(res)
    return results

if __name__ == "__main__":
    print("Starting AWC Pipeline with Detailed Audit Report...")
    print(f"Output Zarrs: {AWC_OUTPUT_DIR}")
    print(f"Audit Report: {REPORT_FILE}")
    
    county_paths = glob.glob(os.path.join(SOIL_ROOT_DIR, "*.zarr"))
    os.makedirs(AWC_OUTPUT_DIR, exist_ok=True)
    
    # NEW: Collect all log entries in memory for sorting
    all_log_entries = []
    total_errors = 0
    
    # Process all cubes
    with Pool(MAX_PARALLEL_PROCESSES) as pool:
        for county_results in tqdm(pool.imap_unordered(process_county_awc, county_paths), total=len(county_paths)):
            for res in county_results:
                if isinstance(res, dict) and res.get("error"):
                    print(f"ERROR: {res['id']}: {res['message']}")
                    total_errors += 1
                    continue
                
                all_log_entries.append(res)
    
    # NEW: Sort log entries by county_name (alphabetical) then cube_id (numerical)
    all_log_entries.sort(key=lambda x: (x['county_name'], x['cube_id']))
    
    # Calculate statistics
    total_processed = len(all_log_entries)
    total_perfect = sum(1 for res in all_log_entries if res['raw_valid'] == 144)
    total_healed = sum(1 for res in all_log_entries if res['step2_valid'] == 144 and res['raw_valid'] < 144)
    total_zeros = sum(res['zeros_final'] for res in all_log_entries)
    
    # NEW: Write sorted report to file
    with open(REPORT_FILE, "w") as f:
        f.write("AWC CALCULATION & FILLING AUDIT REPORT\n")
        f.write("======================================\n")
        f.write("Columns: CubeID | Total | Raw Valid | Step1 | Final Valid | Zeros (Final)\n")
        f.write("-" * 95 + "\n")
        
        for res in all_log_entries:
            # Write main log line
            line = f"{res['id']:<35} | {res['total']:<5} | {res['raw_valid']:<9} | {res['step1_valid']:<5} | {res['step2_valid']:<11} | {res['zeros_final']:<4}\n"
            f.write(line)
            
            # Write matrix dump if this was a lucky sample
            if res['lucky_dump']:
                f.write(res['lucky_dump'])
                print(f"Targeted sample captured: {res['id']}")  # Console notification

        f.write("\n" + "="*40 + "\n")
        f.write("SUMMARY\n")
        f.write(f"Total Cubes: {total_processed}\n")
        f.write(f"Originally Perfect (No NaNs): {total_perfect} ({total_perfect/total_processed*100:.2f}%)\n")
        f.write(f"Repaired Cubes: {total_healed} ({total_healed/total_processed*100:.2f}%)\n")
        f.write(f"Total Zero Values (Final): {total_zeros}\n")
        f.write(f"Errors: {total_errors}\n")
    
    print(f"\nPipeline Done. Check {REPORT_FILE} for targeted repair samples.")