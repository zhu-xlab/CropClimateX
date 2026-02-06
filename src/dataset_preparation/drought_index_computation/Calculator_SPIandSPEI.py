"""
Docstring for Calculator_SPIandSPEI
Calculator for SPI and SPEI indices from precipitation and deficit data.
The utilized method is seasonal fitting, which means fitting distributions separately for each week of the year.
The non-seasonal method is also contained, but just for previous experiments. It is not used in the current implementation.
"""
import os
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Union, Tuple, Optional
import xarray as xr
from tqdm import tqdm
import shutil
import warnings
import time
import traceback 
from scipy.stats import gamma as gamma_dist, logistic as logistic_dist, norm as norm_dist
from pathlib import Path
import lmoments3 as lm
from lmoments3 import distr

class DataCalculator:
    def __init__(self,
                 base_paths: Dict[str, str],
                 num_cubes_map: Dict[str, int],
                 cropclimatex_root: str,
                 daymet_indices_subfolder: Optional[str] = None,
                 skip_nan_calculations: bool = False,
                 spi_spei_input_source_subfolder: Optional[str] = None,
                 spi_spei_use_derived_inputs: bool = False,
                 use_seasonal_method: bool = False,
                 spei_fitting_method: str = 'lmoments_glo'):
        """
        Initialize DataCalculator with configuration for SPI/SPEI input data sources.
        
        Parameters:
        -----------
        use_seasonal_method : bool, default False
            If True, use seasonal method for SPI/SPEI calculation (fit distributions separately for each week of year)
            If False, use non-seasonal method (fit single distribution for entire time series)
        spei_fitting_method : str, default 'lmoments_glo'
            Fitting method for SPEI: 'logistic' (scipy MLE) or 'lmoments_glo' (L-Moments GLO distribution)
        """
        self.base_paths = base_paths
        self.num_cubes_map = num_cubes_map
        self.cropclimatex_root = cropclimatex_root
        self.actual_daymet_indices_subfolder = daymet_indices_subfolder if daymet_indices_subfolder else "indices_from_daymet"
        self.skip_nan_calculations = skip_nan_calculations
        
        # New SPI/SPEI input configuration
        self.spi_spei_input_source_subfolder = spi_spei_input_source_subfolder
        self.spi_spei_use_derived_inputs = spi_spei_use_derived_inputs
        
        # Seasonal calculation method
        self.use_seasonal_method = use_seasonal_method
        
        # SPEI fitting method configuration
        if spei_fitting_method not in ['logistic', 'lmoments_glo']:
            raise ValueError(f"Invalid SPEI fitting method: {spei_fitting_method}. Must be 'logistic' or 'lmoments_glo'.")
        self.spei_fitting_method = spei_fitting_method

        # Initialize output paths for different data types
        self.modis_indices_paths: Dict[str, str] = {}
        if 'modis' in base_paths and base_paths.get('modis'):
            modis_path = base_paths['modis']
            if isinstance(modis_path, str):
                segment_name = os.path.basename(modis_path)
                self.modis_indices_paths['modis'] = os.path.join(self.cropclimatex_root, "indices", segment_name)

        self.daymet_indices_paths: Dict[str, str] = {}
        if 'daymet' in base_paths and base_paths.get('daymet'):
            daymet_path_raw = base_paths['daymet'] 
            if isinstance(daymet_path_raw, str):
                segment_name = os.path.basename(daymet_path_raw) 
                self.daymet_indices_paths['daymet'] = os.path.join(self.cropclimatex_root,
                                                                   self.actual_daymet_indices_subfolder,
                                                                   segment_name)
        
        self.soil_indices_paths: Dict[str, str] = {}
        if 'soil' in base_paths and base_paths.get('soil'):
            soil_path = base_paths['soil']
            if isinstance(soil_path, str):
                segment_name = os.path.basename(soil_path)
                self.soil_indices_paths['soil'] = os.path.join(self.cropclimatex_root, "indices_from_soil", segment_name)

        # Create output directories
        for path_value in list(self.modis_indices_paths.values()) + list(self.daymet_indices_paths.values()) + list(self.soil_indices_paths.values()):
            if path_value and isinstance(path_value, str):
                dir_to_create = os.path.dirname(path_value)
                if dir_to_create: 
                    os.makedirs(dir_to_create, exist_ok=True)

        # Calculate cube offsets for each data type
        self.cube_offsets: Dict[str, int] = {
            dtype: self._get_segment_start(path_val)
            for dtype, path_val in base_paths.items() if path_val and isinstance(path_val, str)
        }

    def _get_segment_start(self, zarr_path: str) -> int:
        """Extract segment start index from Zarr path filename"""
        match = re.search(r"_(\d+)-(\d+)\.zarr$", zarr_path)
        if not match: 
            raise ValueError(f"Invalid Zarr path format for segment start: {zarr_path}")
        return int(match.group(1))

    def _get_cube_path(self, dtype: str, cube_idx: int, input_data: bool = True) -> str:
        """Get path to a specific cube within a data type segment"""
        path_map = None
        segment_zarr_store = None
        offset = 0
        
        if input_data:
            if dtype not in self.base_paths or not self.base_paths[dtype]: 
                raise ValueError(f"Base path for RAW data dtype '{dtype}' not configured or empty.")
            segment_zarr_store = self.base_paths[dtype]
            if dtype not in self.cube_offsets: 
                raise ValueError(f"Cube offset for RAW data dtype '{dtype}' not configured.")
            offset = self.cube_offsets[dtype]
        else:
            if dtype == 'daymet': 
                path_map = self.daymet_indices_paths
            elif dtype == 'soil': 
                path_map = self.soil_indices_paths
            elif dtype == 'modis': 
                path_map = self.modis_indices_paths
            else: 
                raise ValueError(f"Unknown dtype '{dtype}' for derived indices path.")
            
            if dtype not in path_map or not path_map.get(dtype): 
                raise ValueError(f"Derived indices path for dtype '{dtype}' not configured.")
            segment_zarr_store = path_map[dtype]
            raw_dtype_for_offset = dtype 
            if raw_dtype_for_offset not in self.cube_offsets: 
                raise ValueError(f"Cube offset for raw dtype '{raw_dtype_for_offset}' not configured.")
            offset = self.cube_offsets[raw_dtype_for_offset]
        
        actual_cube_id_in_store = offset + cube_idx
        return os.path.join(segment_zarr_store, str(actual_cube_id_in_store))

    def _get_spi_spei_input_cube_path(self, cube_idx: int) -> str:
        """Get path to cube for SPI/SPEI input data (either raw or derived)"""
        if self.spi_spei_use_derived_inputs and self.spi_spei_input_source_subfolder:
            # Use derived data from specified subfolder
            if 'daymet' not in self.base_paths:
                raise ValueError("Daymet base path required for derived input path construction")
            
            daymet_segment_name = os.path.basename(self.base_paths['daymet'])
            derived_segment_path = os.path.join(self.cropclimatex_root, 
                                              self.spi_spei_input_source_subfolder, 
                                              daymet_segment_name)
            actual_cube_id = self.cube_offsets['daymet'] + cube_idx
            return os.path.join(derived_segment_path, str(actual_cube_id))
        else:
            # Use raw daymet data
            return self._get_cube_path('daymet', cube_idx, input_data=True)

    def _handle_nans(self, da: xr.DataArray, var_name: str, s_name: str, c_idx: int) -> Union[xr.DataArray, None]:
        """Handle NaN values in DataArray - preserve original NaN values for SPI/SPEI calculations"""
        if da is None: 
            return None
        if da.size == 0: 
            return da
        
        # For SPI/SPEI calculations, preserve original NaN values without filling
        return da

    def _calculate_nan_percentage(self, data_arr: xr.DataArray, var_name: str) -> float:
        """Calculate the percentage of NaN values in a DataArray"""
        # Skip NaN calculations if configured
        if self.skip_nan_calculations:
            return 0.0
            
        if data_arr is None or data_arr.size == 0:
            return 100.0
        
        total_count = data_arr.size
        nan_count = data_arr.isnull().sum()
        
        # Handle dask arrays
        if hasattr(nan_count, 'compute'):
            nan_count = nan_count.compute()
        if hasattr(nan_count, 'item'):
            nan_count = nan_count.item()
            
        nan_percentage = (nan_count / total_count) * 100
        return float(nan_percentage)

    def _write_nan_statistics(self, stats_dict: Dict[str, float], cube_id: int, output_zarr_store_path: str):
        """Write NaN statistics to a text file"""
        # Skip writing if NaN calculations are disabled
        if self.skip_nan_calculations:
            return
            
        # Create stats directory within the zarr store path
        stats_dir = os.path.join(output_zarr_store_path, "nan_statistics")
        os.makedirs(stats_dir, exist_ok=True)
        
        stats_file_path = os.path.join(stats_dir, f"cube_{cube_id}_nan_stats.txt")
        
        try:
            with open(stats_file_path, 'w') as f:
                f.write(f"NaN Statistics for Cube {cube_id}\n")
                f.write("=" * 40 + "\n\n")
                
                # Input variables section
                f.write("Input Variables:\n")
                f.write("-" * 20 + "\n")
                for var_name in ['prcp', 'Deficit']:
                    if var_name in stats_dict:
                        f.write(f"{var_name}: {stats_dict[var_name]:.2f}% NaN\n")
                
                f.write("\n")
                
                # Output variables section
                f.write("Output Variables:\n")
                f.write("-" * 20 + "\n")
                for var_name in stats_dict:
                    if var_name.startswith('spi') or var_name.startswith('spei'):
                        f.write(f"{var_name}: {stats_dict[var_name]:.2f}% NaN\n")
                        
            print(f"NaN statistics saved to: {stats_file_path}")
            
        except Exception as e:
            print(f"Error writing NaN statistics to {stats_file_path}: {e}")

    def _save_cube_dataset_with_time_chunking(self, abs_cube_num: int, dataset_to_save: xr.Dataset, 
                                            out_zarr_root_seg: str, log_tag: str, time_chunk_size: int = 4):
        """Save dataset to Zarr with time-based chunking like Palmer indices"""
        if not dataset_to_save.data_vars:
            print(f"No vars to save for {log_tag}, cube {abs_cube_num}.")
            return
        
        # Get the time dimension size
        time_var = None
        for var_name, data_var in dataset_to_save.data_vars.items():
            if 'time' in data_var.dims:
                time_var = data_var
                break
        
        if time_var is None:
            print(f"No time dimension found in dataset for cube {abs_cube_num}")
            return
        
        total_time_steps = time_var.sizes['time']
        print(f"Processing cube {abs_cube_num} with {total_time_steps} time steps, chunking by {time_chunk_size}")
        
        # Calculate number of time chunks
        num_time_chunks = (total_time_steps + time_chunk_size - 1) // time_chunk_size
        
        for chunk_idx in range(num_time_chunks):
            start_time_idx = chunk_idx * time_chunk_size
            end_time_idx = min(start_time_idx + time_chunk_size, total_time_steps)
            
            # Create chunk dataset
            chunk_dataset = {}
            for var_name, data_var in dataset_to_save.data_vars.items():
                if 'time' in data_var.dims:
                    chunk_data = data_var.isel(time=slice(start_time_idx, end_time_idx))
                else:
                    chunk_data = data_var
                chunk_dataset[var_name] = chunk_data
            
            chunk_ds = xr.Dataset(chunk_dataset)
            
            # Calculate the absolute chunk number
            abs_chunk_num = abs_cube_num + chunk_idx
            cube_out_path = os.path.join(out_zarr_root_seg, str(abs_chunk_num))
            
            try:
                os.makedirs(out_zarr_root_seg, exist_ok=True)
                
                if os.path.exists(cube_out_path):
                    chunk_ds.to_zarr(cube_out_path, mode='a', consolidated=True)
                    print(f"{log_tag} chunk {chunk_idx} (time steps {start_time_idx}-{end_time_idx-1}) "
                          f"for cube {abs_chunk_num} saved (append mode) to: {cube_out_path}")
                else:
                    chunk_ds.to_zarr(cube_out_path, mode='w', consolidated=True)
                    print(f"{log_tag} chunk {chunk_idx} (time steps {start_time_idx}-{end_time_idx-1}) "
                          f"for cube {abs_chunk_num} saved to: {cube_out_path}")
                    
            except Exception as e:
                print(f"Error saving {log_tag} chunk {chunk_idx} to {cube_out_path}: {e}")
                traceback.print_exc()

    def _save_cube_dataset(self, abs_cube_num: int, dataset_to_save: xr.Dataset, out_zarr_root_seg: str, log_tag: str):
        """Save dataset to Zarr cube with proper error handling"""
        if not dataset_to_save.data_vars:
            print(f"No vars to save for {log_tag}, cube {abs_cube_num}.")
            return
        
        cube_out_path = os.path.join(out_zarr_root_seg, str(abs_cube_num))
        try:
            # Ensure parent directory exists
            os.makedirs(out_zarr_root_seg, exist_ok=True)
            
            # Simple and safe: use append mode if exists, write mode if new
            if os.path.exists(cube_out_path):
                dataset_to_save.to_zarr(cube_out_path, mode='a', consolidated=True)
                print(f"{log_tag} (vars: {list(dataset_to_save.data_vars.keys())}) for cube {abs_cube_num} saved (append mode) to: {cube_out_path}")
            else:
                dataset_to_save.to_zarr(cube_out_path, mode='w', consolidated=True)
                print(f"{log_tag} (vars: {list(dataset_to_save.data_vars.keys())}) for cube {abs_cube_num} saved to: {cube_out_path}")
                
        except Exception as e:
            print(f"Error saving {log_tag} to {cube_out_path}: {e}")
            traceback.print_exc()

    def _calculate_spi_from_prcp_arr(self, prcp_arr: xr.DataArray, ts: int, s_n: str, c_i: int) -> Optional[xr.DataArray]:
        """Calculate SPI using Gamma distribution fitting (seasonal or non-seasonal based on configuration)"""
        if prcp_arr is None: 
            return None
        
        # Choose calculation method based on configuration
        if self.use_seasonal_method:
            return self._calculate_spi_seasonal_from_prcp_arr(prcp_arr, ts, s_n, c_i)
        else:
            return self._calculate_spi_nonseasonal_from_prcp_arr(prcp_arr, ts, s_n, c_i)

    def _calculate_spi_nonseasonal_from_prcp_arr(self, prcp_arr: xr.DataArray, ts: int, s_n: str, c_i: int) -> Optional[xr.DataArray]:
        """Calculate SPI using non-seasonal Gamma distribution fitting (original method)"""
        if prcp_arr is None: 
            return None
        
        accumulated_prcp = prcp_arr.copy()

        def _spi_gamma_1d_for_pixel_timeseries(pixel_accumulated_series_np: np.ndarray) -> np.ndarray:
            """Calculate SPI for a single pixel time series using Gamma distribution"""
            spi_values_1d = np.full_like(pixel_accumulated_series_np, np.nan, dtype=float)
            valid_indices_mask = ~np.isnan(pixel_accumulated_series_np)
            series_no_nan = pixel_accumulated_series_np[valid_indices_mask]
            
            if series_no_nan.size < 20: 
                return spi_values_1d
            
            zeros = series_no_nan[series_no_nan == 0]
            non_zeros = series_no_nan[series_no_nan > 0]
            q_total = zeros.size / series_no_nan.size if series_no_nan.size > 0 else 0.0
            calculated_spi_for_valid_series = np.full_like(series_no_nan, np.nan, dtype=float)
            
            if non_zeros.size > 0:
                try:
                    alpha, loc_fit, beta_scale = gamma_dist.fit(non_zeros, floc=0) 
                    if alpha <= 0 or beta_scale <= 0: 
                        raise ValueError("Gamma fit non-positive params.")
                except Exception: 
                    pass
                else:
                    for i_enum, val_x in enumerate(series_no_nan):
                        if val_x > 0:
                            cdf_g_x = gamma_dist.cdf(val_x, alpha, loc=0, scale=beta_scale)
                            h_x = q_total + (1 - q_total) * cdf_g_x
                        elif val_x == 0: 
                            h_x = q_total
                        else: 
                            h_x = np.nan 
                        
                        if not np.isnan(h_x):
                            epsilon = 1e-5 
                            h_x_clipped = np.clip(h_x, epsilon, 1 - epsilon)
                            calculated_spi_for_valid_series[i_enum] = norm_dist.ppf(h_x_clipped)
            elif q_total == 1.0 and series_no_nan.size > 0: 
                calculated_spi_for_valid_series[:] = 0.0 
            
            spi_values_1d[valid_indices_mask] = calculated_spi_for_valid_series
            return spi_values_1d

        try:
            accumulated_prcp_rechunked = accumulated_prcp
            is_dask_array = hasattr(accumulated_prcp.data, 'chunks') 
            
            if is_dask_array:
                if 'time' in accumulated_prcp.dims:
                    time_axis_num = accumulated_prcp.get_axis_num('time')
                    if time_axis_num < len(accumulated_prcp.chunks) and \
                       accumulated_prcp.chunks[time_axis_num] is not None and \
                       len(accumulated_prcp.chunks[time_axis_num]) > 1:
                        accumulated_prcp_rechunked = accumulated_prcp.chunk({'time': -1})
            
            spi_temp = xr.apply_ufunc(
                _spi_gamma_1d_for_pixel_timeseries,
                accumulated_prcp_rechunked, 
                input_core_dims=[['time']],  
                output_core_dims=[['time']], 
                dask="forbidden",         # Disable dask to avoid parallelization issues
                output_dtypes=[float],
                keep_attrs=False 
            )
            
            # Fix dimension naming if needed
            if 'time' not in spi_temp.dims and 'time' in accumulated_prcp_rechunked.dims:
                input_time_axis_index = accumulated_prcp_rechunked.get_axis_num('time')
                if input_time_axis_index < len(spi_temp.dims):
                    potential_misnamed_dim = spi_temp.dims[input_time_axis_index]
                    if potential_misnamed_dim != 'time' and \
                       spi_temp.sizes.get(potential_misnamed_dim) == accumulated_prcp_rechunked.sizes.get('time'):
                        spi_temp = spi_temp.rename({potential_misnamed_dim: 'time'})

            # Assign time coordinates if missing
            if 'time' in spi_temp.dims and 'time' not in spi_temp.coords:
                if 'time' in accumulated_prcp.coords:
                    spi_temp = spi_temp.assign_coords(time=accumulated_prcp.time)
                elif 'time' in accumulated_prcp_rechunked.coords:
                    spi_temp = spi_temp.assign_coords(time=accumulated_prcp_rechunked.time)
            
            spi_final = spi_temp.reindex_like(accumulated_prcp) 
            if hasattr(accumulated_prcp, 'attrs'):
                spi_final.attrs = accumulated_prcp.attrs.copy()

        except Exception as e: 
            return None
            
        # spi_final = spi_final.clip(-4, 4)
        
        # Set variable name and attributes
        if ts == 1:
            spi_final.name = 'spi-1'
            long_name_str = 'Standardized Precipitation Index (1-month, Single Gamma Fit - Non-Seasonal)'
        else:
            spi_final.name = f'spi-{ts}'
            long_name_str = f'Standardized Precipitation Index (Time Scale: {ts} months, Single Gamma Fit - Non-Seasonal)'

        spi_final.attrs.update({ 
            'time_scale_months': ts, 
            'long_name': long_name_str,
            'calculation_method': 'Fitted single Gamma distribution over entire accumulated series per pixel (non-seasonal), transformed to Z-score.'
        })
        return spi_final

    def _calculate_spei_from_deficit(self, deficit_arr: xr.DataArray, ts: int, s_n: str, c_i: int) -> Optional[xr.DataArray]:
        """Calculate SPEI using Logistic distribution fitting (seasonal or non-seasonal based on configuration)"""
        if deficit_arr is None: 
            return None
        
        # Choose calculation method based on configuration
        if self.use_seasonal_method:
            return self._calculate_spei_seasonal_from_deficit(deficit_arr, ts, s_n, c_i)
        else:
            return self._calculate_spei_nonseasonal_from_deficit(deficit_arr, ts, s_n, c_i)

    def _calculate_spei_nonseasonal_from_deficit(self, deficit_arr: xr.DataArray, ts: int, s_n: str, c_i: int) -> Optional[xr.DataArray]:
        """Calculate SPEI using non-seasonal distribution fitting (supports both Logistic and L-Moments GLO methods)"""
        if deficit_arr is None: 
            return None
        
        accumulated_deficit = deficit_arr.copy()

        if 'time' not in accumulated_deficit.coords or not np.issubdtype(accumulated_deficit.time.dtype, np.datetime64):
            return None
        if 'time' not in accumulated_deficit.dims:
            return None

        # Choose fitting function based on configuration
        if self.spei_fitting_method == 'lmoments_glo':
            #print(f"DEBUG SPEI Non-seasonal: Using L-Moments GLO method for ts={ts}")
            # def _spei_glo_1d_for_pixel_timeseries(pixel_accumulated_series_np: np.ndarray) -> np.ndarray:
            #     """Calculate SPEI for a single pixel time series using L-Moments GLO distribution"""
            #     spei_values_1d = np.full_like(pixel_accumulated_series_np, np.nan, dtype=float)
            #     valid_indices_mask = ~np.isnan(pixel_accumulated_series_np)
            #     series_no_nan = pixel_accumulated_series_np[valid_indices_mask]
                
            #     if series_no_nan.size < 20:
            #         return spei_values_1d
                
            #     calculated_spei_for_valid_series = np.full_like(series_no_nan, np.nan, dtype=float)
            #     try:
            #         # Fit Generalized Logistic (GLO) using L-moments
            #         params = distr.glo.lmom_fit(series_no_nan)
            #         # Calculate CDF
            #         cdf_glo = distr.glo.cdf(series_no_nan, **params)
            #         # Convert to standard normal
            #         epsilon = 1e-5
            #         cdf_glo_clipped = np.clip(cdf_glo, epsilon, 1 - epsilon)
            #         calculated_spei_for_valid_series = norm_dist.ppf(cdf_glo_clipped)
            #     except Exception as e:
            #         pass  # Keep NaN if fitting fails
                
            #     spei_values_1d[valid_indices_mask] = calculated_spei_for_valid_series
            #     return spei_values_1d
            
            def _spei_glo_1d_for_pixel_timeseries(pixel_accumulated_series_np: np.ndarray) -> np.ndarray:
                """Calculate SPEI for a single pixel time series using L-Moments GLO distribution"""
                spei_values_1d = np.full_like(pixel_accumulated_series_np, np.nan, dtype=float)
                valid_indices_mask = ~np.isnan(pixel_accumulated_series_np)
                series_no_nan = pixel_accumulated_series_np[valid_indices_mask]
                
                if series_no_nan.size < 20:
                    return spei_values_1d
                
                calculated_spei_for_valid_series = np.full_like(series_no_nan, np.nan, dtype=float)
                try:
                    # Fit Generalized Logistic (GLO) using L-moments
                    params = distr.glo.lmom_fit(series_no_nan)
                    
                    # --- Robust CDF Calculation Start ---
                    # Initialize CDF array with NaNs
                    cdf_glo = np.full_like(series_no_nan, np.nan, dtype=float)
                    
                    # Try standard calculation first
                    try:
                        cdf_glo = distr.glo.cdf(series_no_nan, **params)
                    except Exception:
                        # If calculation crashes completely, we continue to clamping logic below
                        pass
                        
                    # Fix outliers that caused NaNs (GLO distribution bounds / "The Wall" issue)
                    # Check for NaNs in the calculated CDF
                    nan_mask = np.isnan(cdf_glo)
                    
                    if np.any(nan_mask):
                        loc_param = params.get('loc')
                        if loc_param is not None:
                            # Clamp values below lower bound (Extreme Drought) -> CDF = 0.0
                            # Logic: The CDF is NaN AND the value is significantly lower than location parameter
                            lower_bound_mask = nan_mask & (series_no_nan < loc_param)
                            cdf_glo[lower_bound_mask] = 0.0
                            
                            # Clamp values above upper bound (Extreme Wet) -> CDF = 1.0
                            # Logic: The CDF is NaN AND the value is significantly higher than location parameter
                            upper_bound_mask = nan_mask & (series_no_nan > loc_param)
                            cdf_glo[upper_bound_mask] = 1.0
                    # --- Robust CDF Calculation End ---

                    # Convert to standard normal
                    epsilon = 1e-5
                    cdf_glo_clipped = np.clip(cdf_glo, epsilon, 1 - epsilon)
                    calculated_spei_for_valid_series = norm_dist.ppf(cdf_glo_clipped)
                    
                except Exception as e:
                    pass  # Keep NaN if fitting fails completely
                
                spei_values_1d[valid_indices_mask] = calculated_spei_for_valid_series
                return spei_values_1d
            
            fitting_function = _spei_glo_1d_for_pixel_timeseries
            method_name = "L-Moments GLO"
        else:  # logistic
            #print(f"DEBUG SPEI Non-seasonal: Using Logistic MLE method for ts={ts}")
            def _spei_logistic_1d_for_pixel_timeseries(pixel_accumulated_series_np: np.ndarray) -> np.ndarray:
                """Calculate SPEI for a single pixel time series using Logistic distribution"""
                spei_values_1d = np.full_like(pixel_accumulated_series_np, np.nan, dtype=float)
                valid_indices_mask = ~np.isnan(pixel_accumulated_series_np)
                series_no_nan = pixel_accumulated_series_np[valid_indices_mask]
                
                if series_no_nan.size < 20: 
                    return spei_values_1d
                
                calculated_spei_for_valid_series = np.full_like(series_no_nan, np.nan, dtype=float)
                try:
                    loc_fit, scale_fit = logistic_dist.fit(series_no_nan)
                    if scale_fit <= 0: 
                        raise ValueError("Logistic fit non-positive scale.")
                except Exception: 
                    pass 
                else:
                    cdf_logistic = logistic_dist.cdf(series_no_nan, loc=loc_fit, scale=scale_fit)
                    epsilon = 1e-5
                    cdf_logistic_clipped = np.clip(cdf_logistic, epsilon, 1 - epsilon)
                    calculated_spei_for_valid_series = norm_dist.ppf(cdf_logistic_clipped)
                
                spei_values_1d[valid_indices_mask] = calculated_spei_for_valid_series
                return spei_values_1d
            
            fitting_function = _spei_logistic_1d_for_pixel_timeseries
            method_name = "Logistic MLE"

        try:
            accumulated_deficit_rechunked = accumulated_deficit
            is_dask_array = hasattr(accumulated_deficit.data, 'chunks')
            
            if is_dask_array:
                if 'time' in accumulated_deficit.dims: 
                    time_axis_num = accumulated_deficit.get_axis_num('time')
                    if time_axis_num < len(accumulated_deficit.chunks) and \
                       accumulated_deficit.chunks[time_axis_num] is not None and \
                       len(accumulated_deficit.chunks[time_axis_num]) > 1:
                        accumulated_deficit_rechunked = accumulated_deficit.chunk({'time': -1})
            
            #print(f"DEBUG SPEI Non-seasonal: Applying {method_name} fitting function...")
            spei_temp = xr.apply_ufunc(
                fitting_function,
                accumulated_deficit_rechunked, 
                input_core_dims=[['time']],
                output_core_dims=[['time']], 
                dask="forbidden",  # Disable dask to avoid parallelization issues
                output_dtypes=[float],
                keep_attrs=False 
            )
            
            # Fix dimension naming if needed
            if 'time' not in spei_temp.dims and 'time' in accumulated_deficit_rechunked.dims:
                input_time_dim_name = 'time'
                if len(spei_temp.dims) == len(accumulated_deficit_rechunked.dims):
                    time_axis_idx_in_input = accumulated_deficit_rechunked.get_axis_num(input_time_dim_name)
                    if time_axis_idx_in_input < len(spei_temp.dims):
                        potential_misnamed_dim = spei_temp.dims[time_axis_idx_in_input]
                        if potential_misnamed_dim != input_time_dim_name and \
                           spei_temp.sizes.get(potential_misnamed_dim) == accumulated_deficit_rechunked.sizes.get(input_time_dim_name):
                            spei_temp = spei_temp.rename({potential_misnamed_dim: input_time_dim_name})
            
            # Assign time coordinates if missing
            if 'time' in spei_temp.dims and 'time' not in spei_temp.coords:
                if 'time' in accumulated_deficit.coords:
                    spei_temp = spei_temp.assign_coords(time=accumulated_deficit.time)
                elif 'time' in accumulated_deficit_rechunked.coords:
                    spei_temp = spei_temp.assign_coords(time=accumulated_deficit_rechunked.time)
            
            spei_final = spei_temp.reindex_like(accumulated_deficit)
            if hasattr(accumulated_deficit, 'attrs'):
                spei_final.attrs = accumulated_deficit.attrs.copy()
            
            #print(f"DEBUG SPEI Non-seasonal: Fitting completed using {method_name}")

        except Exception as e:
            print(f"ERROR in non-seasonal SPEI calculation: {e}")
            return None
            
        # spei_final = spei_final.clip(-4, 4) 

        # Set variable name and attributes
        if ts == 1:
            spei_final.name = 'spei-1'
            long_name_str = f'Standardized Precipitation-Evapotranspiration Index (1-month, {method_name} - Non-Seasonal)'
        else:
            spei_final.name = f'spei-{ts}'
            long_name_str = f'Standardized Precipitation-Evapotranspiration Index (Time Scale: {ts} months, {method_name} - Non-Seasonal)'
        
        spei_final.attrs.update({ 
            'time_scale_months': ts,
            'long_name': long_name_str,
            'calculation_method': f'Fitted single {method_name} distribution over entire accumulated series per pixel (non-seasonal), transformed to Z-score.',
            'fitting_method': self.spei_fitting_method
        })
        return spei_final
    

    
    def _calculate_spi_seasonal_from_prcp_arr(self, prcp_arr: xr.DataArray, ts: int, s_n: str, c_i: int) -> Optional[xr.DataArray]:
        """Calculate SPI using seasonal Gamma distribution fitting (separate distributions for each week of year)"""
        if prcp_arr is None: 
            return None
        
        accumulated_prcp = prcp_arr.copy()

        # Check if time coordinate exists and has datetime dtype
        if 'time' not in accumulated_prcp.coords or not np.issubdtype(accumulated_prcp.time.dtype, np.datetime64):
            print(f"Warning: Time coordinate missing or not datetime type for seasonal SPI calculation (ts={ts})")
            return None
        if 'time' not in accumulated_prcp.dims:
            print(f"Warning: Time dimension missing for seasonal SPI calculation (ts={ts})")
            return None

        # Extract time information before the apply_ufunc call
        time_index = pd.to_datetime(accumulated_prcp.time.values)
        week_numbers = time_index.isocalendar().week.values
        # Handle week 53 by merging with week 52
        week_numbers = np.where(week_numbers > 52, 52, week_numbers)
        
        print(f"Seasonal SPI calculation for ts={ts}: {len(time_index)} time steps, {len(np.unique(week_numbers))} unique weeks")

        def _spi_gamma_seasonal_1d_for_pixel_timeseries(pixel_accumulated_series_np: np.ndarray, week_numbers_np: np.ndarray) -> np.ndarray:
            """
            #Calculate seasonal SPI for a single pixel time series using week-specific Gamma distributions"""
            try:
                #print(f"DEBUG SPI: Function called with data shape: {pixel_accumulated_series_np.shape}")
                #print(f"DEBUG SPI: Function called with week numbers shape: {week_numbers_np.shape}")
                
                # This function operates on a single time series (1D) 
                # pixel_accumulated_series_np should be 1D array of length [time]
                # week_numbers_np should also be 1D array of length [time]
                
                if pixel_accumulated_series_np.ndim != 1:
                    #print(f"DEBUG SPI: ERROR - Expected 1D input but got {pixel_accumulated_series_np.ndim}D")
                    return np.full_like(pixel_accumulated_series_np, np.nan, dtype=float)
                
                if week_numbers_np.ndim != 1:
                    #print(f"DEBUG SPI: ERROR - Expected 1D week numbers but got {week_numbers_np.ndim}D")
                    return np.full_like(pixel_accumulated_series_np, np.nan, dtype=float)
                
                #print(f"DEBUG SPI: Data length: {len(pixel_accumulated_series_np)}")
                #print(f"DEBUG SPI: Week numbers length: {len(week_numbers_np)}")
                
                # Data validation
                if len(pixel_accumulated_series_np) != len(week_numbers_np):
                    #print(f"DEBUG SPI: Shape mismatch: data={len(pixel_accumulated_series_np)}, weeks={len(week_numbers_np)}")
                    return np.full_like(pixel_accumulated_series_np, np.nan, dtype=float)
                
                #print(f"DEBUG SPI: Data range: {np.nanmin(pixel_accumulated_series_np):.3f} to {np.nanmax(pixel_accumulated_series_np):.3f}")
                #print(f"DEBUG SPI: Week range: {np.min(week_numbers_np)} to {np.max(week_numbers_np)}")
                
                # Create output array
                spi_result = np.full_like(pixel_accumulated_series_np, np.nan, dtype=float)
                
                # Valid data mask (non-NaN and non-negative for precipitation)
                valid_mask = (~np.isnan(pixel_accumulated_series_np)) & (pixel_accumulated_series_np >= 0)
                valid_count = np.sum(valid_mask)
                
                #print(f"DEBUG SPI: Valid data points: {valid_count}/{len(pixel_accumulated_series_np)}")
                
                if valid_count < 5:  # Reduced from 10 to 5 for more flexibility
                    #print(f"DEBUG SPI: Insufficient valid data ({valid_count} < 5), returning NaN array")
                    return spi_result
                
                # For each unique week, collect all values from that week across all years
                # Only use valid data points for finding unique weeks
                valid_indices = np.where(valid_mask)[0]
                unique_weeks = np.unique(week_numbers_np[valid_indices])
                #print(f"DEBUG SPI: Found {len(unique_weeks)} unique weeks: {unique_weeks[:10]}...")  # Show first 10 weeks
                
                # Dictionary to store fitted parameters for each week
                week_distributions = {}
                
                # Fit distributions for each week
                successful_fits = 0
                for week in unique_weeks:
                    if week < 1 or week > 52:  # Skip invalid weeks
                        #print(f"DEBUG SPI: Skipping invalid week {week}")
                        continue
                        
                    # Get all values for this week across all years
                    # Create boolean mask for this week among valid indices
                    week_mask_indices = week_numbers_np == week
                    week_valid_mask = week_mask_indices & valid_mask
                    week_count = np.sum(week_valid_mask)
                    if week_count < 2:  # Reduced from 3 to 2 observations minimum
                        #print(f"DEBUG SPI: Week {week} has only {week_count} observations, skipping")
                        continue
                        
                    week_values = pixel_accumulated_series_np[week_valid_mask]
                    #print(f"DEBUG SPI: Week {week}: {week_count} values, range {np.min(week_values):.3f} to {np.max(week_values):.3f}")
                    
                    # Handle zero precipitation values
                    non_zeros = week_values[week_values > 0]
                    prob_zero = np.sum(week_values == 0) / len(week_values)
                    zero_count = np.sum(week_values == 0)
                    
                    #print(f"DEBUG SPI: Week {week}: {len(non_zeros)} non-zero values, {zero_count} zeros, prob_zero={prob_zero:.3f}")
                    
                    if len(non_zeros) < 1:  # Reduced from 2 to 1 for more flexibility
                        # If all values are zero for this week, assign special handling
                        #print(f"DEBUG SPI: Week {week}: All values are zero, using special handling")
                        week_distributions[week] = {
                            'alpha': None,
                            'loc_fit': 0,
                            'beta_scale': None,
                            'prob_zero': 1.0,
                            'all_zero': True
                        }
                        successful_fits += 1
                        continue
                        
                    try:
                        # For single non-zero value, use simple approximation
                        if len(non_zeros) == 1:
                            #print(f"DEBUG SPI: Week {week}: Single non-zero value {non_zeros[0]:.3f}, using approximation")
                            alpha = 1.0
                            beta_scale = non_zeros[0]
                        else:
                            # Fit Gamma distribution to non-zero values
                            #print(f"DEBUG SPI: Week {week}: Fitting Gamma to {len(non_zeros)} non-zero values")
                            alpha, loc_fit, beta_scale = gamma_dist.fit(non_zeros, floc=0)
                            #print(f"DEBUG SPI: Week {week}: Fitted alpha={alpha:.3f}, beta_scale={beta_scale:.3f}")
                        
                        if alpha <= 0 or beta_scale <= 0:
                            #print(f"DEBUG SPI: Week {week}: Invalid parameters alpha={alpha}, beta_scale={beta_scale}, skipping")
                            continue
                        
                        # Store distribution parameters for this week
                        week_distributions[week] = {
                            'alpha': alpha,
                            'loc_fit': 0,  # Always use 0 for location
                            'beta_scale': beta_scale,
                            'prob_zero': prob_zero,
                            'all_zero': False
                        }
                        successful_fits += 1
                        #print(f"DEBUG SPI: Week {week}: Successfully fitted distribution (fit #{successful_fits})")
                        
                    except Exception as e:
                        # If fitting fails for this week, skip it
                        #print(f"DEBUG SPI: Week {week}: Fitting failed with error: {e}")
                        continue
                
                #print(f"DEBUG SPI: Successfully fitted distributions for {successful_fits} weeks out of {len(unique_weeks)} unique weeks")
                
                # Now calculate SPI values using the fitted distributions
                successful_calculations = 0
                total_calculations = 0
                for i in range(len(pixel_accumulated_series_np)):
                    if not valid_mask[i]:
                        continue
                        
                    week = week_numbers_np[i]
                    value = pixel_accumulated_series_np[i]
                    total_calculations += 1
                    
                    if week in week_distributions:
                        params = week_distributions[week]
                        
                        try:
                            if params.get('all_zero', False):
                                # All values for this week are zero
                                spi_result[i] = 0.0  # Assign neutral SPI value
                                successful_calculations += 1
                                if total_calculations <= 10:  # Log first 10 calculations
                                    #print(f"DEBUG SPI: Time {i}, Week {week}, Value {value:.3f}: All-zero week, assigned SPI=0.0")
                                    pass
                            elif value == 0:
                                # For zero precipitation
                                cdf_val = params['prob_zero'] / 2
                                cdf_val = np.clip(cdf_val, 1e-5, 1 - 1e-5)
                                spi_val = norm_dist.ppf(cdf_val)
                                spi_result[i] = spi_val
                                successful_calculations += 1
                                if total_calculations <= 10:  # Log first 10 calculations
                                    #print(f"DEBUG SPI: Time {i}, Week {week}, Value {value:.3f}: Zero precip, cdf={cdf_val:.6f}, SPI={spi_val:.3f}")
                                    pass
                            else:
                                # For non-zero precipitation
                                gamma_cdf = gamma_dist.cdf(value, params['alpha'], 
                                                         loc=params['loc_fit'], 
                                                         scale=params['beta_scale'])
                                if gamma_cdf is None or np.isnan(gamma_cdf):
                                    #print(f"DEBUG SPI: Time {i}, Week {week}, Value {value:.3f}: Gamma CDF is NaN")
                                    gamma_cdf = 1.0  # Fallback to avoid NaN

                                cdf_val = params['prob_zero'] + (1 - params['prob_zero']) * gamma_cdf
                                epsilon = 1e-5
                                cdf_val = np.clip(cdf_val, epsilon, 1 - epsilon)
                                spi_val = norm_dist.ppf(cdf_val)
                                spi_result[i] = spi_val
                                successful_calculations += 1
                                if total_calculations <= 10:  # Log first 10 calculations
                                    #  print(f"DEBUG SPI: Time {i}, Week {week}, Value {value:.3f}: gamma_cdf={gamma_cdf:.6f}, cdf={cdf_val:.6f}, SPI={spi_val:.3f}")
                                    pass
                            
                        except Exception as e:
                            # If calculation fails for this value, leave as NaN
                            if total_calculations <= 10:  # Log first 10 calculation errors
                                # print(f"DEBUG SPI: Time {i}, Week {week}, Value {value:.3f}: Calculation failed with error: {e}")
                                pass
                            continue
                    else:
                        if total_calculations <= 10:  # Log first 10 missing distributions
                            # print(f"DEBUG SPI: Time {i}, Week {week}, Value {value:.3f}: No distribution found for this week")
                            pass
                
                # print(f"DEBUG SPI: Successfully calculated {successful_calculations}/{total_calculations} SPI values")
                
                # Check final results
                final_valid = np.sum(~np.isnan(spi_result))
                final_unique = len(np.unique(spi_result[~np.isnan(spi_result)])) if final_valid > 0 else 0
                # if final_valid > 0:
                #     print(f"DEBUG SPI: Final results: {final_valid} valid values, {final_unique} unique values")
                #     print(f"DEBUG SPI: SPI range: {np.nanmin(spi_result):.3f} to {np.nanmax(spi_result):.3f}")
                #     print(f"DEBUG SPI: First 10 SPI values: {spi_result[:10]}")
                #     pass
                # else:
                #     print(f"DEBUG SPI: WARNING: No valid SPI values calculated!")
                #     pass
                
                return spi_result
                
            except Exception as e:
                # If any error occurs, return NaN array
                # print(f"DEBUG SPI: Fatal error in seasonal SPI calculation: {e}")
                import traceback
                # print(f"DEBUG SPI: Traceback: {traceback.format_exc()}")
                return np.full_like(pixel_accumulated_series_np, np.nan, dtype=float)

        try:
            accumulated_prcp_rechunked = accumulated_prcp
            # This is necessary to ensure that the data is fully loaded before applying the function
            accumulated_prcp_rechunked = accumulated_prcp_rechunked.load()
            
            # print(f"DEBUG SPI Main: Input data loaded successfully")
            # print(f"DEBUG SPI Main: Data shape: {accumulated_prcp_rechunked.shape}")
            # print(f"DEBUG SPI Main: Data dimensions: {accumulated_prcp_rechunked.dims}")
            # print(f"DEBUG SPI Main: Data range: {float(accumulated_prcp_rechunked.min())} to {float(accumulated_prcp_rechunked.max())}")
            # print(f"DEBUG SPI Main: NaN count: {int(accumulated_prcp_rechunked.isnull().sum())}")
            
            # Create week numbers array for broadcasting
            week_numbers_array = xr.DataArray(week_numbers, dims=['time'], coords={'time': accumulated_prcp_rechunked.time})
            # print(f"DEBUG SPI Main: Week numbers array shape: {week_numbers_array.shape}")
            # print(f"DEBUG SPI Main: Week numbers range: {int(week_numbers_array.min())} to {int(week_numbers_array.max())}")

            # print(f"DEBUG SPI Main: About to call xr.apply_ufunc...")
            spi_temp = xr.apply_ufunc(
                _spi_gamma_seasonal_1d_for_pixel_timeseries,
                accumulated_prcp_rechunked,
                week_numbers_array,
                input_core_dims=[['time'], ['time']],
                output_core_dims=[['time']],
                dask="forbidden",  # Disable dask to avoid parallelization issues
                output_dtypes=[float],
                keep_attrs=False,
                vectorize=True  # This tells xarray to vectorize the function over non-core dimensions
            )

            # print(f"DEBUG SPI Main: xr.apply_ufunc completed successfully")
            # print(f"DEBUG SPI Main: Result shape: {spi_temp.shape}")
            # print(f"DEBUG SPI Main: Result dimensions: {spi_temp.dims}")

            # Fix dimension naming if needed
            if 'time' not in spi_temp.dims and 'time' in accumulated_prcp_rechunked.dims:
                # print(f"DEBUG SPI Main: Fixing dimension naming...")
                input_time_axis_index = accumulated_prcp_rechunked.get_axis_num('time')
                if input_time_axis_index < len(spi_temp.dims):
                    potential_misnamed_dim = spi_temp.dims[input_time_axis_index]
                    if potential_misnamed_dim != 'time' and \
                       spi_temp.sizes.get(potential_misnamed_dim) == accumulated_prcp_rechunked.sizes.get('time'):
                        spi_temp = spi_temp.rename({potential_misnamed_dim: 'time'})
                        # print(f"DEBUG SPI Main: Renamed dimension {potential_misnamed_dim} to 'time'")

            # Assign time coordinates if missing
            if 'time' in spi_temp.dims and 'time' not in spi_temp.coords:
                # print(f"DEBUG SPI Main: Assigning time coordinates...")
                if 'time' in accumulated_prcp.coords:
                    spi_temp = spi_temp.assign_coords(time=accumulated_prcp.time)
                elif 'time' in accumulated_prcp_rechunked.coords:
                    spi_temp = spi_temp.assign_coords(time=accumulated_prcp_rechunked.time)

            spi_final = spi_temp.reindex_like(accumulated_prcp)
            if hasattr(accumulated_prcp, 'attrs'):
                spi_final.attrs = accumulated_prcp.attrs.copy()
                
            # print(f"DEBUG SPI Main: Final SPI data prepared")
            # print(f"DEBUG SPI Main: Final shape: {spi_final.shape}")
            # print(f"DEBUG SPI Main: Final NaN count: {int(spi_final.isnull().sum())}")
            # print(f"DEBUG SPI Main: Final valid count: {int((~spi_final.isnull()).sum())}")
            if int((~spi_final.isnull()).sum()) > 0:
                #print(f"DEBUG SPI Main: Final range: {float(spi_final.min())} to {float(spi_final.max())}")
                pass

        except Exception as e:
            print(f"Error in seasonal SPI calculation: {e}")
            #print(f"DEBUG SPI Main: Full traceback:")
            import traceback
            print(traceback.format_exc())
            return None

        # spi_final = spi_final.clip(-4, 4)

        # Set variable name and attributes
        if ts == 1:
            spi_final.name = 'spi-1'
            long_name_str = 'Standardized Precipitation Index (1-month, Seasonal Gamma Fit - Weekly)'
        else:
            spi_final.name = f'spi-{ts}'
            long_name_str = f'Standardized Precipitation Index (Time Scale: {ts} months, Seasonal Gamma Fit - Weekly)'

        spi_final.attrs.update({
            'time_scale_months': ts,
            'long_name': long_name_str,
            'calculation_method': 'Fitted separate Gamma distributions for each week of year (seasonal), transformed to Z-score.'
        })
        return spi_final
    

    def _calculate_spei_seasonal_from_deficit(self, deficit_arr: xr.DataArray, ts: int, s_n: str, c_i: int) -> Optional[xr.DataArray]:
        """Calculate SPEI using seasonal distribution fitting (supports both Logistic and L-Moments GLO methods)"""
        if deficit_arr is None:
            return None
        
        accumulated_deficit = deficit_arr.copy()

        # Check if time coordinate exists and has datetime dtype
        if 'time' not in accumulated_deficit.coords or not np.issubdtype(accumulated_deficit.time.dtype, np.datetime64):
            print(f"Warning: Time coordinate missing or not datetime type for seasonal SPEI calculation")
            return None
        if 'time' not in accumulated_deficit.dims:
            print(f"Warning: Time dimension missing for seasonal SPEI calculation")
            return None

        # Extract time information before the apply_ufunc call
        time_index = pd.to_datetime(accumulated_deficit.time.values)
        week_numbers = time_index.isocalendar().week.values
        # Handle week 53 by merging with week 52
        week_numbers = np.where(week_numbers > 52, 52, week_numbers)
        
        # print(f"Seasonal SPEI calculation for ts={ts}: {len(time_index)} time steps, {len(np.unique(week_numbers))} unique weeks")
        # print(f"DEBUG SPEI Seasonal: Using {self.spei_fitting_method} method")

        # Choose fitting function based on configuration
        if self.spei_fitting_method == 'lmoments_glo':
            print (f"DEBUG SPEI Seasonal: Using L-Moments GLO method for ts={ts}")
            def _spei_glo_seasonal_1d_for_pixel_timeseries(pixel_accumulated_series_np: np.ndarray, week_numbers_np: np.ndarray) -> np.ndarray:
                """Calculate seasonal SPEI for a single pixel time series using week-specific L-Moments GLO distributions"""
                try:
                    if pixel_accumulated_series_np.ndim != 1 or week_numbers_np.ndim != 1:
                        return np.full_like(pixel_accumulated_series_np, np.nan, dtype=float)
                    
                    if len(pixel_accumulated_series_np) != len(week_numbers_np):
                        return np.full_like(pixel_accumulated_series_np, np.nan, dtype=float)
                    
                    spei_result = np.full_like(pixel_accumulated_series_np, np.nan, dtype=float)
                    valid_mask = ~np.isnan(pixel_accumulated_series_np)
                    valid_count = np.sum(valid_mask)
                    
                    if valid_count < 10:
                        return spei_result
                    
                    # Find unique weeks in valid data
                    valid_indices = np.where(valid_mask)[0]
                    unique_weeks = np.unique(week_numbers_np[valid_indices])
                    
                    # Dictionary to store fitted parameters for each week
                    week_params = {}
                    
                    # Step 1: Fit GLO distribution for each week
                    for week_num in unique_weeks:
                        week_mask = week_numbers_np == week_num
                        week_data = pixel_accumulated_series_np[week_mask]
                        week_data_valid = week_data[~np.isnan(week_data)]
                        
                        if len(week_data_valid) < 10:
                            continue
                        
                        try:
                            # Fit GLO using L-moments
                            params = distr.glo.lmom_fit(week_data_valid)
                            week_params[week_num] = params
                        except Exception:
                            continue
                    
                    # Step 2: Calculate SPEI for each time step
                    for idx in valid_indices:
                        week_num = week_numbers_np[idx]
                        value = pixel_accumulated_series_np[idx]
                        
                        if week_num not in week_params:
                            continue
                        
                        try:
                            params = week_params[week_num]
                            cdf = np.nan # intial value
                            
                            try:
                                cdf = distr.glo.cdf(value, **params)

                                if np.isnan(cdf):
                                    raise ValueError("CDF calculation returned NaN")
                            except Exception:
                                # Now we have nan cdf for a specific time step, but deficit is valid is at this step
                                # Therefore, we need to fix this distribution outliers by cutting tails
                                loc_param = params.get('loc')
                                if loc_param is not None:
                                    if value < loc_param:
                                        cdf = 0.0
                                    else:
                                        cdf = 1.0

                            epsilon = 1e-5
                            cdf_clipped = np.clip(cdf, epsilon, 1 - epsilon)
                            spei_result[idx] = norm_dist.ppf(cdf_clipped)
                        except Exception:
                            continue
                    
                    return spei_result
                    
                except Exception as e:
                    return np.full_like(pixel_accumulated_series_np, np.nan, dtype=float)
            
            fitting_function = _spei_glo_seasonal_1d_for_pixel_timeseries
            method_name = "L-Moments GLO"
        else:  # logistic
            def _spei_logistic_seasonal_1d_for_pixel_timeseries(pixel_accumulated_series_np: np.ndarray, week_numbers_np: np.ndarray) -> np.ndarray:
                """Calculate seasonal SPEI for a single pixel time series using week-specific Logistic distributions"""
                try:
                    # print(f"DEBUG SPEI: Input data shape: {pixel_accumulated_series_np.shape}")
                    # print(f"DEBUG SPEI: Week numbers shape: {week_numbers_np.shape}")
                    # print(f"DEBUG SPEI: Data range: {np.nanmin(pixel_accumulated_series_np):.3f} to {np.nanmax(pixel_accumulated_series_np):.3f}")
                    # print(f"DEBUG SPEI: Week range: {np.min(week_numbers_np)} to {np.max(week_numbers_np)}")
                    
                    # Create output array
                    spei_result = np.full_like(pixel_accumulated_series_np, np.nan, dtype=float)
                    
                    # Valid data mask (non-NaN)
                    valid_mask = ~np.isnan(pixel_accumulated_series_np)
                    valid_count = np.sum(valid_mask)
                    
                    # print(f"DEBUG SPEI: Valid data points: {valid_count}/{len(pixel_accumulated_series_np)}")
                    
                    if valid_count < 5:  # Reduced from 10 to 5 for more flexibility
                        # print(f"DEBUG SPEI: Insufficient valid data ({valid_count} < 5), returning NaN array")
                        return spei_result
                    
                    # For each unique week, collect all values from that week across all years
                    unique_weeks = np.unique(week_numbers_np[valid_mask])
                    # print(f"DEBUG SPEI: Found {len(unique_weeks)} unique weeks: {unique_weeks[:10]}...")  # Show first 10 weeks
                    
                    # Dictionary to store fitted parameters for each week
                    week_distributions = {}
                    
                    # Fit distributions for each week
                    successful_fits = 0
                    for week in unique_weeks:
                        if week < 1 or week > 52:  # Skip invalid weeks
                            # print(f"DEBUG SPEI: Skipping invalid week {week}")
                            continue
                            
                        # Get all values for this week across all years
                        week_mask = (week_numbers_np == week) & valid_mask
                        week_count = np.sum(week_mask)
                        if week_count < 2:  # Reduced from 3 to 2 observations minimum
                            # print(f"DEBUG SPEI: Week {week} has only {week_count} observations, skipping")
                            continue
                            
                        week_values = pixel_accumulated_series_np[week_mask]
                        # print(f"DEBUG SPEI: Week {week}: {week_count} values, range {np.min(week_values):.3f} to {np.max(week_values):.3f}")
                        
                        if len(week_values) < 2:  # Need at least 2 values for logistic fit
                            # For single value, use normal approximation
                            if len(week_values) == 1:
                                # print(f"DEBUG SPEI: Week {week}: Single value {week_values[0]:.3f}, using special handling")
                                week_distributions[week] = {
                                    'loc_fit': week_values[0],
                                    'scale_fit': 1.0,  # Default scale
                                    'single_value': True
                                }
                                successful_fits += 1
                            continue
                            
                        try:
                            # Fit Logistic distribution
                            if len(week_values) == 2:
                                # For exactly 2 values, use simple approximation
                                # print(f"DEBUG SPEI: Week {week}: Two values, using approximation")
                                loc_fit = np.mean(week_values)
                                scale_fit = np.std(week_values) * 1.8  # Approximate conversion to logistic scale
                                if scale_fit <= 0:
                                    scale_fit = 1.0
                            else:
                                # Standard logistic fit for 3+ values
                                # print(f"DEBUG SPEI: Week {week}: Fitting Logistic to {len(week_values)} values")
                                loc_fit, scale_fit = logistic_dist.fit(week_values)
                                # print(f"DEBUG SPEI: Week {week}: Fitted loc={loc_fit:.3f}, scale={scale_fit:.3f}")
                            
                            if scale_fit <= 0:
                                # print(f"DEBUG SPEI: Week {week}: Invalid scale {scale_fit}, skipping")
                                continue
                            
                            # Store distribution parameters for this week
                            week_distributions[week] = {
                                'loc_fit': loc_fit,
                                'scale_fit': scale_fit,
                                'single_value': False
                            }
                            successful_fits += 1
                            # print(f"DEBUG SPEI: Week {week}: Successfully fitted distribution (fit #{successful_fits})")
                            
                        except Exception as e:
                            # If fitting fails for this week, skip it
                            # print(f"DEBUG SPEI: Week {week}: Fitting failed with error: {e}")
                            continue
                    
                    # print(f"DEBUG SPEI: Successfully fitted distributions for {successful_fits} weeks out of {len(unique_weeks)} unique weeks")
                    
                    # Now calculate SPEI values using the fitted distributions
                    successful_calculations = 0
                    total_calculations = 0
                    for i in range(len(pixel_accumulated_series_np)):
                        if not valid_mask[i]:
                            continue
                            
                        week = week_numbers_np[i]
                        value = pixel_accumulated_series_np[i]
                        total_calculations += 1
                        
                        if week in week_distributions:
                            params = week_distributions[week]
                            
                            try:
                                if params.get('single_value', False):
                                    # For weeks with single value, assign neutral SPEI
                                    spei_result[i] = 0.0
                                    successful_calculations += 1
                                    if total_calculations <= 10:  # Log first 10 calculations
                                        #print(f"DEBUG SPEI: Time {i}, Week {week}, Value {value:.3f}: Single-value week, assigned SPEI=0.0")
                                        pass
                                else:
                                    # Calculate CDF using fitted logistic distribution
                                    cdf_val = logistic_dist.cdf(value, loc=params['loc_fit'], scale=params['scale_fit'])
                                    
                                    # Convert to normal distribution
                                    cdf_val = np.clip(cdf_val, 1e-5, 1 - 1e-5)
                                    spei_val = norm_dist.ppf(cdf_val)
                                    spei_result[i] = spei_val
                                    successful_calculations += 1
                                    if total_calculations <= 10:  # Log first 10 calculations
                                        #print(f"DEBUG SPEI: Time {i}, Week {week}, Value {value:.3f}: cdf={cdf_val:.6f}, SPEI={spei_val:.3f}")
                                        pass
                                
                            except Exception as e:
                                # If calculation fails for this value, leave as NaN
                                if total_calculations <= 10:  # Log first 10 calculation errors
                                    #print(f"DEBUG SPEI: Time {i}, Week {week}, Value {value:.3f}: Calculation failed with error: {e}")
                                    pass
                                continue
                        else:
                            if total_calculations <= 10:  # Log first 10 missing distributions
                                #print(f"DEBUG SPEI: Time {i}, Week {week}, Value {value:.3f}: No distribution found for this week")
                                pass
                    
                    #print(f"DEBUG SPEI: Successfully calculated {successful_calculations}/{total_calculations} SPEI values")
                    
                    # Check final results
                    final_valid = np.sum(~np.isnan(spei_result))
                    final_unique = len(np.unique(spei_result[~np.isnan(spei_result)])) if final_valid > 0 else 0
                    if final_valid > 0:
                        #print(f"DEBUG SPEI: Final results: {final_valid} valid values, {final_unique} unique values")
                        #print(f"DEBUG SPEI: SPEI range: {np.nanmin(spei_result):.3f} to {np.nanmax(spei_result):.3f}")
                        #print(f"DEBUG SPEI: First 10 SPEI values: {spei_result[:10]}")
                        pass
                    else:
                        #print(f"DEBUG SPEI: WARNING: No valid SPEI values calculated!")
                        pass
                    
                    return spei_result
                    
                except Exception as e:
                    # If any error occurs, return NaN array
                    #print(f"DEBUG SPEI: Fatal error in seasonal SPEI calculation: {e}")
                    import traceback
                    #print(f"DEBUG SPEI: Traceback: {traceback.format_exc()}")
                    return np.full_like(pixel_accumulated_series_np, np.nan, dtype=float)
            
            fitting_function = _spei_logistic_seasonal_1d_for_pixel_timeseries
            method_name = "Logistic MLE"

        try:
            accumulated_deficit_rechunked = accumulated_deficit
            # Force load to memory to avoid dask array issues with apply_ufunc
            accumulated_deficit_rechunked = accumulated_deficit_rechunked.load()
            
            #print(f"DEBUG SPEI Seasonal: Input data loaded successfully")
            #print(f"DEBUG SPEI Seasonal: Data shape: {accumulated_deficit_rechunked.shape}")
            #print(f"DEBUG SPEI Seasonal: Data range: {float(accumulated_deficit_rechunked.min())} to {float(accumulated_deficit_rechunked.max())}")
            
            # Create week numbers array for broadcasting
            week_numbers_array = xr.DataArray(week_numbers, dims=['time'], coords={'time': accumulated_deficit_rechunked.time})

            #print(f"DEBUG SPEI Seasonal: Applying {method_name} fitting function...")
            spei_temp = xr.apply_ufunc(
                fitting_function,
                accumulated_deficit_rechunked,
                week_numbers_array,
                input_core_dims=[['time'], ['time']],
                output_core_dims=[['time']],
                dask="forbidden",  # Disable dask to avoid parallelization issues
                output_dtypes=[float],
                vectorize=True,  # This tells xarray to vectorize the function over non-core dimensions
                keep_attrs=False
            )

            # Fix dimension naming if needed
            if 'time' not in spei_temp.dims and 'time' in accumulated_deficit_rechunked.dims:
                input_time_dim_name = 'time'
                if len(spei_temp.dims) == len(accumulated_deficit_rechunked.dims):
                    time_axis_idx_in_input = accumulated_deficit_rechunked.get_axis_num(input_time_dim_name)
                    if time_axis_idx_in_input < len(spei_temp.dims):
                        potential_misnamed_dim = spei_temp.dims[time_axis_idx_in_input]
                        if potential_misnamed_dim != input_time_dim_name and \
                           spei_temp.sizes.get(potential_misnamed_dim) == accumulated_deficit_rechunked.sizes.get(input_time_dim_name):
                            spei_temp = spei_temp.rename({potential_misnamed_dim: input_time_dim_name})

            # Assign time coordinates if missing
            if 'time' in spei_temp.dims and 'time' not in spei_temp.coords:
                if 'time' in accumulated_deficit.coords:
                    spei_temp = spei_temp.assign_coords(time=accumulated_deficit.time)
                elif 'time' in accumulated_deficit_rechunked.coords:
                    spei_temp = spei_temp.assign_coords(time=accumulated_deficit_rechunked.time)

            spei_final = spei_temp.reindex_like(accumulated_deficit)
            if hasattr(accumulated_deficit, 'attrs'):
                spei_final.attrs = accumulated_deficit.attrs.copy()
            
            #print(f"DEBUG SPEI Seasonal: Fitting completed using {method_name}")
            #print(f"DEBUG SPEI Seasonal: Final NaN count: {int(spei_final.isnull().sum())}")
            #print(f"DEBUG SPEI Seasonal: Final valid count: {int((~spei_final.isnull()).sum())}")

        except Exception as e:
            print(f"Error in seasonal SPEI calculation: {e}")
            traceback.print_exc()
            return None

        # spei_final = spei_final.clip(-4, 4)

        # Set variable name and attributes
        if ts == 1:
            spei_final.name = 'spei-1'
            long_name_str = f'Standardized Precipitation-Evapotranspiration Index (1-month, {method_name} - Seasonal)'
        else:
            spei_final.name = f'spei-{ts}'
            long_name_str = f'Standardized Precipitation-Evapotranspiration Index (Time Scale: {ts} months, {method_name} - Seasonal)'

        spei_final.attrs.update({
            'time_scale_months': ts,
            'long_name': long_name_str,
            'calculation_method': f'Fitted separate {method_name} distributions for each week of year (seasonal), transformed to Z-score.',
            'fitting_method': self.spei_fitting_method
        })
        return spei_final

    def process_daymet_derived_indices(self,
                                    time_scale_spei: int = 1,
                                    time_scale_spi: int = 1,
                                    target_variables: Optional[List[str]] = None,
                                    force_recompute_vars: Optional[List[str]] = None,
                                    output_time_chunk_size: int = 4):
        """Process Daymet-derived indices (SPI and SPEI only) - single time scale version with time chunking"""
        method_type = "Seasonal (weekly)" if self.use_seasonal_method else "Non-seasonal"
        print(f"--- Processing Daymet-derived indices (Subfolder: '{self.actual_daymet_indices_subfolder}'). Target variables: {target_variables if target_variables is not None else 'ALL'} ---")
        print(f"INFO: Using {method_type} calculation method")
        print(f"INFO: SPEI fitting method: {self.spei_fitting_method}")
        print(f"INFO: SPI will be calculated with time scale ts={time_scale_spi}")
        print(f"INFO: SPEI will be calculated with time scale ts={time_scale_spei}")
        print(f"Output time chunk size: {output_time_chunk_size}")

        data_type_daymet = 'daymet'
        if data_type_daymet not in self.base_paths or not self.base_paths.get(data_type_daymet):
            print(f"Daymet raw data path not configured. Skipping Daymet-derived indices.")
            return
        
        # Determine generic target variables
        generic_target_variables = set()
        if target_variables is None: 
            generic_target_variables.update(['spi', 'spei'])
        elif isinstance(target_variables, list):
            for tv in target_variables:
                if tv in ['spi', 'spei']:
                    generic_target_variables.add(tv)
                elif tv.startswith('spi-'):
                    generic_target_variables.add('spi')
                elif tv.startswith('spei-'):
                    generic_target_variables.add('spei')
                else:
                    generic_target_variables.add(tv)
        
        vars_to_compute_internally = set(generic_target_variables)
        
        if not vars_to_compute_internally and target_variables is not None: 
            print("No Daymet variables specified or resolved. Skipping.")
            return
        
        num_cubes_daymet = self.num_cubes_map.get(data_type_daymet, 0)
        if num_cubes_daymet == 0 and vars_to_compute_internally: 
            print(f"No Daymet cubes, but vars {vars_to_compute_internally} requested. Skipping.")
            return
        
        s_name_daymet = os.path.basename(self.base_paths[data_type_daymet]).replace(".zarr", "")
        
        effective_out_zarr_store_path = self.daymet_indices_paths.get(data_type_daymet)
        if not effective_out_zarr_store_path:
            print(f"ERROR: Daymet output Zarr store path not determined. Skipping.")
            return

        hardcoded_daymet_chunks = {'time': 4, 'y': 12, 'x': 12}

        for i in tqdm(range(num_cubes_daymet), desc=f"Processing Daymet-idx for {s_name_daymet} (subfolder: {self.actual_daymet_indices_subfolder})"):
            abs_cn = self.cube_offsets[data_type_daymet] + i
            
            cube_res_to_save: Dict[str, xr.DataArray] = {}
            computed_this_run: Dict[str, xr.DataArray] = {}
            existing_derived_ds = None
            
            nan_stats = {}
            
            path_to_this_derived_cube = os.path.join(effective_out_zarr_store_path, str(abs_cn))
            if os.path.exists(path_to_this_derived_cube):
                try:
                    existing_derived_ds = xr.open_dataset(path_to_this_derived_cube, engine='zarr', chunks=hardcoded_daymet_chunks)
                    print(f"Loaded existing derived data for cube {abs_cn} from {path_to_this_derived_cube}")
                except Exception as e:
                    print(f"Warning: Could not load existing derived data from {path_to_this_derived_cube}: {e}")
                    existing_derived_ds = None

            max_retries = 1
            for attempt in range(max_retries):
                try:
                    input_cube_path = self._get_spi_spei_input_cube_path(i)
                    
                    if not os.path.exists(input_cube_path):
                        print(f"Warning: Input cube {input_cube_path} not found for SPI/SPEI calculation. Skipping cube {abs_cn}.")
                        break
                    
                    # Initialize spatial_ref storage
                    source_spatial_ref = None
                    
                    try:
                        input_ds = xr.open_dataset(input_cube_path, engine='zarr', chunks=hardcoded_daymet_chunks)
                        
                        # Extract spatial_ref if available for propagation
                        if 'spatial_ref' in input_ds:
                            source_spatial_ref = input_ds['spatial_ref'].copy(deep=True)
                            print(f"Extracted spatial_ref from input data for cube {abs_cn}")
                    except Exception as e:
                        print(f"Error loading input data from {input_cube_path}: {e}")
                        break

                    # Process SPI if requested
                    if 'spi' in vars_to_compute_internally:
                        spi_var_name = 'spi-1' if time_scale_spi == 1 else f'spi-{time_scale_spi}'
                        
                        if (existing_derived_ds is not None and 
                            spi_var_name in existing_derived_ds.data_vars and 
                            (force_recompute_vars is None or spi_var_name not in force_recompute_vars)):
                            print(f"SPI (ts={time_scale_spi}) already exists for cube {abs_cn}. Skipping calculation.")
                            cube_res_to_save[spi_var_name] = existing_derived_ds[spi_var_name]
                        else:
                            # Find precipitation variable
                            prcp_var_name = None
                            prcp_target = f'prcp-{time_scale_spi}'
                            if prcp_target in input_ds.data_vars:
                                prcp_var_name = prcp_target
                            elif 'prcp-1' in input_ds.data_vars:
                                prcp_var_name = 'prcp-1'
                            elif 'prcp' in input_ds.data_vars:
                                prcp_var_name = 'prcp'
                            
                            if prcp_var_name:
                                prcp_data = input_ds[prcp_var_name]
                                prcp_data = self._handle_nans(prcp_data, prcp_var_name, s_name_daymet, i)
                                
                                if prcp_data is not None:
                                    print(f"Computing SPI with time scale {time_scale_spi} for cube {abs_cn} using '{prcp_var_name}'...")
                                    spi_result = self._calculate_spi_from_prcp_arr(prcp_data, time_scale_spi, s_name_daymet, i)
                                    
                                    if spi_result is not None:
                                        spi_result.name = spi_var_name
                                        cube_res_to_save[spi_var_name] = spi_result
                                        computed_this_run[spi_var_name] = spi_result
                                        
                                        nan_percentage = self._calculate_nan_percentage(spi_result, spi_var_name)
                                        nan_stats[spi_var_name] = nan_percentage
                                        print(f"SPI (ts={time_scale_spi}) computed successfully for cube {abs_cn}. NaN percentage: {nan_percentage:.2f}%")
                                    else:
                                        print(f"Failed to compute SPI (ts={time_scale_spi}) for cube {abs_cn}")

                    # Process SPEI if requested
                    if 'spei' in vars_to_compute_internally:
                        spei_var_name = 'spei-1' if time_scale_spei == 1 else f'spei-{time_scale_spei}'
                        
                        if (existing_derived_ds is not None and 
                            spei_var_name in existing_derived_ds.data_vars and 
                            (force_recompute_vars is None or spei_var_name not in force_recompute_vars)):
                            print(f"SPEI (ts={time_scale_spei}) already exists for cube {abs_cn}. Skipping calculation.")
                            cube_res_to_save[spei_var_name] = existing_derived_ds[spei_var_name]
                        else:
                            # Find deficit variable
                            deficit_var_name = None
                            deficit_target = f'Deficit-{time_scale_spei}'
                            if deficit_target in input_ds.data_vars:
                                deficit_var_name = deficit_target
                            elif 'Deficit-1' in input_ds.data_vars:
                                deficit_var_name = 'Deficit-1'
                            elif 'Deficit' in input_ds.data_vars:
                                deficit_var_name = 'Deficit'
                            
                            if deficit_var_name:
                                deficit_data = input_ds[deficit_var_name]
                                deficit_data = self._handle_nans(deficit_data, deficit_var_name, s_name_daymet, i)
                                
                                if deficit_data is not None:
                                    print(f"Computing SPEI with time scale {time_scale_spei} for cube {abs_cn} using '{deficit_var_name}'...")
                                    spei_result = self._calculate_spei_from_deficit(deficit_data, time_scale_spei, s_name_daymet, i)
                                    
                                    if spei_result is not None:
                                        spei_result.name = spei_var_name
                                        cube_res_to_save[spei_var_name] = spei_result
                                        computed_this_run[spei_var_name] = spei_result
                                        
                                        nan_percentage = self._calculate_nan_percentage(spei_result, spei_var_name)
                                        nan_stats[spei_var_name] = nan_percentage
                                        print(f"SPEI (ts={time_scale_spei}) computed successfully for cube {abs_cn}. NaN percentage: {nan_percentage:.2f}%")
                                    else:
                                        print(f"Failed to compute SPEI (ts={time_scale_spei}) for cube {abs_cn}")

                    # Save results if any were computed - 使用简单的保存方法
                    if cube_res_to_save:
                        dataset_to_save = xr.Dataset(cube_res_to_save)
                        
                        # Add spatial_ref if extracted from input
                        if source_spatial_ref is not None:
                            dataset_to_save['spatial_ref'] = source_spatial_ref
                            # Force grid_mapping attribute for all data variables
                            for var_name in dataset_to_save.data_vars:
                                if var_name != 'spatial_ref':
                                    dataset_to_save[var_name].attrs['grid_mapping'] = 'spatial_ref'
                            print(f"Added spatial_ref to output dataset for cube {abs_cn}")
                        
                        # 使用简单的保存方法，参考帕尔默计算器
                        self._save_cube_dataset(
                            abs_cn, dataset_to_save, effective_out_zarr_store_path, 
                            f"Daymet-derived indices (ts_spi={time_scale_spi}, ts_spei={time_scale_spei})"
                        )
                        
                        if nan_stats:
                            self._write_nan_statistics(nan_stats, abs_cn, effective_out_zarr_store_path)

                    # Clean up
                    if 'input_ds' in locals():
                        input_ds.close()
                    if existing_derived_ds is not None:
                        existing_derived_ds.close()

                    break  # Success, exit retry loop

                except Exception as e:
                    print(f"Error in attempt {attempt + 1} for cube {abs_cn}: {e}")
                    if attempt == max_retries - 1:
                        print(f"All attempts failed for cube {abs_cn}")
                    else:
                        print(f"Retrying cube {abs_cn}...")
                        time.sleep(1)
                            
        print(f"Daymet-idx processing done for {s_name_daymet} (subfolder: {self.actual_daymet_indices_subfolder}).")

    def process_daymet_derived_indices_multiple_timescales(self,
                                                        spi_time_scales: List[int],
                                                        spei_time_scales: List[int],
                                                        target_variables: Optional[List[str]] = None,
                                                        force_recompute_vars: Optional[List[str]] = None,
                                                        output_time_chunk_size: int = 4):
        """Process Daymet-derived indices (SPI and SPEI) for multiple time scales with time chunking"""
        method_type = "Seasonal (weekly)" if self.use_seasonal_method else "Non-seasonal"
        print(f"--- Processing Daymet-derived indices for multiple time scales ---")
        print(f"INFO: Using {method_type} calculation method")
        print(f"INFO: SPEI fitting method: {self.spei_fitting_method}")
        print(f"SPI time scales: {spi_time_scales}")
        print(f"SPEI time scales: {spei_time_scales}")
        print(f"Target variables: {target_variables}")
        print(f"Output time chunk size: {output_time_chunk_size}")

        data_type_daymet = 'daymet'
        if data_type_daymet not in self.base_paths or not self.base_paths.get(data_type_daymet):
            print(f"Daymet raw data path not configured. Skipping Daymet-derived indices.")
            return
        
        # Determine which variables to compute
        compute_spi = target_variables is None or 'spi' in target_variables
        compute_spei = target_variables is None or 'spei' in target_variables
        
        if not compute_spi and not compute_spei:
            print("No SPI or SPEI variables to compute. Skipping.")
            return
        
        num_cubes_daymet = self.num_cubes_map.get(data_type_daymet, 0)
        if num_cubes_daymet == 0:
            print(f"No Daymet cubes available. Skipping.")
            return
        
        s_name_daymet = os.path.basename(self.base_paths[data_type_daymet]).replace(".zarr", "")
        
        # Use the standard output path (no time scale suffix)
        effective_out_zarr_store_path = self.daymet_indices_paths.get(data_type_daymet)
        if not effective_out_zarr_store_path:
            print(f"ERROR: Daymet output Zarr store path not determined. Skipping.")
            return

        # Default chunk settings
        hardcoded_daymet_chunks = {'time': 4, 'y': 12, 'x': 12}

        # Process each cube
        for i in tqdm(range(num_cubes_daymet), desc=f"Processing SPI/SPEI multi-timescales for {s_name_daymet}"):
            abs_cn = self.cube_offsets[data_type_daymet] + i
            
            cube_res_to_save: Dict[str, xr.DataArray] = {}
            computed_this_run: Dict[str, xr.DataArray] = {}
            existing_derived_ds = None
            
            # NaN statistics dictionary
            nan_stats = {}
            
            # Check for existing derived data (only check the base cube, since we'll be creating time chunks)
            path_to_this_derived_cube = os.path.join(effective_out_zarr_store_path, str(abs_cn))
            if os.path.exists(path_to_this_derived_cube):
                try:
                    existing_derived_ds = xr.open_dataset(path_to_this_derived_cube, engine='zarr', chunks=hardcoded_daymet_chunks)
                    print(f"Loaded existing derived data for cube {abs_cn} from {path_to_this_derived_cube}")
                except Exception as e:
                    print(f"Warning: Could not load existing derived data from {path_to_this_derived_cube}: {e}")
                    existing_derived_ds = None

            max_retries = 1
            for attempt in range(max_retries):
                try:
                    # Get input data for SPI/SPEI calculations
                    input_cube_path = self._get_spi_spei_input_cube_path(i)
                    
                    if not os.path.exists(input_cube_path):
                        print(f"Warning: Input cube {input_cube_path} not found for SPI/SPEI calculation. Skipping cube {abs_cn}.")
                        break
                    
                    # Initialize spatial_ref storage
                    source_spatial_ref = None
                    
                    try:
                        input_ds = xr.open_dataset(input_cube_path, engine='zarr', chunks=hardcoded_daymet_chunks)
                        print(f"Available variables in cube {abs_cn}: {list(input_ds.data_vars.keys())}")
                        
                        # Extract spatial_ref if available for propagation
                        if 'spatial_ref' in input_ds:
                            source_spatial_ref = input_ds['spatial_ref'].copy(deep=True)
                            print(f"Extracted spatial_ref from input data for cube {abs_cn}")
                    except Exception as e:
                        print(f"Error loading input data from {input_cube_path}: {e}")
                        break

                    # Process SPI for all time scales
                    if compute_spi:
                        for spi_ts in spi_time_scales:
                            spi_var_name = 'spi-1' if spi_ts == 1 else f'spi-{spi_ts}'
                            
                            # Check if already computed
                            if (existing_derived_ds is not None and 
                                spi_var_name in existing_derived_ds.data_vars and 
                                (force_recompute_vars is None or spi_var_name not in force_recompute_vars)):
                                print(f"SPI (ts={spi_ts}) already exists for cube {abs_cn}. Skipping calculation.")
                                cube_res_to_save[spi_var_name] = existing_derived_ds[spi_var_name]
                            else:
                                # Find the appropriate precipitation variable for this time scale
                                prcp_var_name = None
                                prcp_target = f'prcp-{spi_ts}'
                                if prcp_target in input_ds.data_vars:
                                    prcp_var_name = prcp_target
                                elif 'prcp-1' in input_ds.data_vars:
                                    prcp_var_name = 'prcp-1'
                                elif 'prcp' in input_ds.data_vars:
                                    prcp_var_name = 'prcp'
                                
                                if prcp_var_name:
                                    prcp_data = input_ds[prcp_var_name]
                                    prcp_data = self._handle_nans(prcp_data, prcp_var_name, s_name_daymet, i)
                                    
                                    if prcp_data is not None:
                                        print(f"Computing SPI with time scale {spi_ts} for cube {abs_cn} using '{prcp_var_name}'...")
                                        spi_result = self._calculate_spi_from_prcp_arr(prcp_data, spi_ts, s_name_daymet, i)
                                        
                                        if spi_result is not None:
                                            spi_result.name = spi_var_name
                                            cube_res_to_save[spi_var_name] = spi_result
                                            computed_this_run[spi_var_name] = spi_result
                                            
                                            nan_percentage = self._calculate_nan_percentage(spi_result, spi_var_name)
                                            nan_stats[spi_var_name] = nan_percentage
                                            print(f"SPI (ts={spi_ts}) computed successfully for cube {abs_cn}. NaN percentage: {nan_percentage:.2f}%")
                                        else:
                                            print(f"Failed to compute SPI (ts={spi_ts}) for cube {abs_cn}")
                                    else:
                                        print(f"No valid precipitation data for SPI (ts={spi_ts}) calculation in cube {abs_cn}")
                                else:
                                    print(f"No appropriate precipitation variable found for SPI (ts={spi_ts}) in cube {abs_cn}")

                    # Process SPEI for all time scales
                    if compute_spei:
                        for spei_ts in spei_time_scales:
                            spei_var_name = 'spei-1' if spei_ts == 1 else f'spei-{spei_ts}'
                            
                            # Check if already computed
                            if (existing_derived_ds is not None and 
                                spei_var_name in existing_derived_ds.data_vars and 
                                (force_recompute_vars is None or spei_var_name not in force_recompute_vars)):
                                print(f"SPEI (ts={spei_ts}) already exists for cube {abs_cn}. Skipping calculation.")
                                cube_res_to_save[spei_var_name] = existing_derived_ds[spei_var_name]
                            else:
                                # Find the appropriate deficit variable for this time scale
                                deficit_var_name = None
                                deficit_target = f'Deficit-{spei_ts}'
                                if deficit_target in input_ds.data_vars:
                                    deficit_var_name = deficit_target
                                elif 'Deficit-1' in input_ds.data_vars:
                                    deficit_var_name = 'Deficit-1'
                                elif 'Deficit' in input_ds.data_vars:
                                    deficit_var_name = 'Deficit'
                                
                                if deficit_var_name:
                                    deficit_data = input_ds[deficit_var_name]
                                    deficit_data = self._handle_nans(deficit_data, deficit_var_name, s_name_daymet, i)
                                    
                                    if deficit_data is not None:
                                        print(f"Computing SPEI with time scale {spei_ts} for cube {abs_cn} using '{deficit_var_name}'...")
                                        spei_result = self._calculate_spei_from_deficit(deficit_data, spei_ts, s_name_daymet, i)
                                        
                                        if spei_result is not None:
                                            spei_result.name = spei_var_name
                                            cube_res_to_save[spei_var_name] = spei_result
                                            computed_this_run[spei_var_name] = spei_result
                                            
                                            nan_percentage = self._calculate_nan_percentage(spei_result, spei_var_name)
                                            nan_stats[spei_var_name] = nan_percentage
                                            print(f"SPEI (ts={spei_ts}) computed successfully for cube {abs_cn}. NaN percentage: {nan_percentage:.2f}%")
                                        else:
                                            print(f"Failed to compute SPEI (ts={spei_ts}) for cube {abs_cn}")
                                    else:
                                        print(f"No valid deficit data for SPEI (ts={spei_ts}) calculation in cube {abs_cn}")
                                else:
                                    print(f"No appropriate deficit variable found for SPEI (ts={spei_ts}) in cube {abs_cn}")

                    # Save results if any were computed
                    if cube_res_to_save:
                        dataset_to_save = xr.Dataset(cube_res_to_save)
                        var_list = list(dataset_to_save.data_vars.keys())
                        
                        # Add spatial_ref if extracted from input
                        if source_spatial_ref is not None:
                            dataset_to_save['spatial_ref'] = source_spatial_ref
                            # Force grid_mapping attribute for all data variables
                            for var_name in dataset_to_save.data_vars:
                                if var_name != 'spatial_ref':
                                    dataset_to_save[var_name].attrs['grid_mapping'] = 'spatial_ref'
                            print(f"Added spatial_ref to output dataset for cube {abs_cn}")
                        
                        # save using simple method
                        self._save_cube_dataset(
                            abs_cn, dataset_to_save, effective_out_zarr_store_path, 
                            f"Daymet-derived indices (multiple timescales)"
                        )
                        
                        print(f"Saved variables for cube {abs_cn}: {var_list}")
                        
                        if nan_stats:
                            self._write_nan_statistics(nan_stats, abs_cn, effective_out_zarr_store_path)

                    # Clean up
                    if 'input_ds' in locals():
                        input_ds.close()
                    if existing_derived_ds is not None:
                        existing_derived_ds.close()

                    break  # Success, exit retry loop

                except Exception as e:
                    print(f"Error in attempt {attempt + 1} for cube {abs_cn}: {e}")
                    if attempt == max_retries - 1:
                        print(f"All attempts failed for cube {abs_cn}")
                    else:
                        print(f"Retrying cube {abs_cn}...")
                        time.sleep(1)
                            
        print(f"Multi-timescale SPI/SPEI processing done for {s_name_daymet}.")