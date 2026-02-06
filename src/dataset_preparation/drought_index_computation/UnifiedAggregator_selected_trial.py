# unified_temporal_aggregator.py
import os
import numpy as np
import re
from typing import Dict, List, Union, Tuple, Optional, Literal, Any
import xarray as xr
from tqdm import tqdm
import shutil
import warnings
import time
import traceback
from collections import defaultdict
import pandas as pd

class UnifiedTemporalAggregator:
    def __init__(self,
                output_aggregated_root_template: str,
                aggregation_period: str,
                variables_config: Dict[str, Dict[str, Any]],
                mode: str = "standard",
                output_resolution: Optional[str] = None,
                ):
        
        self.variables_config = variables_config
        self.aggregation_period = aggregation_period
        self.mode = mode.lower()
        self.output_resolution = output_resolution
        self._log_func = getattr(tqdm, 'write', print)

        # Validate mode and parameters
        if self.mode not in ["standard", "rolling"]:
            raise ValueError(f"Mode must be 'standard' or 'rolling', got: {mode}")
        
        if self.mode == "rolling" and output_resolution is None:
            raise ValueError("In rolling mode, output_resolution must be specified")

        # Use the template directly as the output root (no more placeholders)
        self.output_aggregated_root = output_aggregated_root_template
        
        self._log_func(f"UnifiedTemporalAggregator initialized.")
        self._log_func(f" Mode: {self.mode}")
        self._log_func(f" Aggregation period: {self.aggregation_period}")
        if self.mode == "rolling":
            self._log_func(f" Output resolution: {self.output_resolution}")
        self._log_func(f" Output aggregated indices to: {self.output_aggregated_root}")
        self._log_func(f" Variables configuration (first item shown if many): "
                    f"{list(self.variables_config.items())[0] if self.variables_config else 'None'}")

        os.makedirs(self.output_aggregated_root, exist_ok=True)

    def _parse_period_to_number(self, period: str) -> int:
        """Parse period string to get the number (e.g., '90D' -> 90, '3M' -> 3)"""
        import re
        if 'D' in period.upper():
            match = re.search(r'(\d+)D', period.upper())
            return int(match.group(1)) if match else 1
        elif 'W' in period.upper():
            match = re.search(r'(\d+)W', period.upper())
            return int(match.group(1)) if match else 1
        elif 'M' in period.upper():
            match = re.search(r'(\d+)M', period.upper())
            return int(match.group(1)) if match else 1
        else:
            return 1

    def _get_variable_output_name(self, var_name: str) -> str:
        """Generate output variable name with suffix for rolling mode"""
        if self.mode == "rolling":
            # Get the number from aggregation period for suffix
            period_number = self._parse_period_to_number(self.aggregation_period)
            return f"{var_name}-{period_number}"
        else:
            return var_name

    def _parse_period_to_days(self, period: str) -> int:
        """Convert time period string to number of days"""
        import re
        if 'D' in period.upper():
            match = re.search(r'(\d+)D', period.upper())
            return int(match.group(1)) if match else 1
        elif 'W' in period.upper():
            match = re.search(r'(\d+)W', period.upper())
            return (int(match.group(1)) if match else 1) * 7
        elif 'M' in period.upper():
            match = re.search(r'(\d+)M', period.upper())
            return (int(match.group(1)) if match else 1) * 30  # Approximate 30 days per month
        else:
            return 1

    def _aggregate_standard_mode(self, daily_da: xr.DataArray, method: str) -> xr.DataArray:
        """Standard mode: direct aggregation by period"""
        daily_da = daily_da.copy() 
        daily_da['time'] = daily_da.time.dt.floor('D')
        #resampler = daily_da.resample(time=self.aggregation_period)
        # resampler = daily_da.resample(time=self.aggregation_period, label='left', closed='left')
        # Use 'right' label and closed to align with end of period, if we have a time stamp at 1980-01-01, it belongs to the period ending at 1980-01-01
        resampler = daily_da.resample(time=self.aggregation_period, label='right', closed='right')
        if method == 'mean': 
            return resampler.mean(skipna=True)
        elif method == 'sum': 
            return resampler.sum(skipna=True,min_count=5)
        # After examination, all variables (in daily resolution) for summation does not have nan days (if this pixel is not a total "nan" pixeo).
        # The total nan pixel means all days are nan, so the sum should be nan as well.
        # For other pixels, the sum should be the sum of all valid days, so skipna=True is just a defensive setting.
        # We set min_count=5 to ensure that 1980-01-01 to 1980-01-05 can be summed to a valid value, 
        # because the first time stamp will be 1979-12-30 (Sunday), and we do not have data on 1979-12-30 and 1979-12-31.
        # Furthermore, daymet data does not have data on 12.31 of leap years, so for periods including 12.31, we may have one less day.

        elif method == 'min': 
            return resampler.min(skipna=True)
        elif method == 'max': 
            return resampler.max(skipna=True)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _aggregate_rolling_mode(self, daily_da: xr.DataArray, method: str) -> xr.DataArray:
        """Rolling mode: generate output points at specified resolution, each aggregating a window of data"""
        # Parse aggregation window size in days
        window_size = self._parse_period_to_days(self.aggregation_period)
        
        # Generate output time grid identical to standard mode (using output_resolution)
        # This ensures time alignment regardless of window size
        try:
            start_time = pd.to_datetime(daily_da.time.values[0])
            end_time = pd.to_datetime(daily_da.time.values[-1])
        except Exception as e:
            self._log_func(f"Error parsing time values: {e}")
            raise ValueError(f"Unable to parse time dimension: {e}")
        
        # Generate the same time grid that standard mode would produce
        # Create a dummy resampler to get the standard mode time points
        try:
            #dummy_resampler = daily_da.resample(time=self.output_resolution)
            dummy_resampler = daily_da.resample(time=self.output_resolution, label='right', closed='right')
            standard_time_points = []
            for label, group in dummy_resampler:
                standard_time_points.append(label)
            
            # Convert to pandas DatetimeIndex for easier manipulation
            output_times = pd.DatetimeIndex(standard_time_points)
            
            if len(output_times) == 0:
                self._log_func(f"Warning: No output time points generated for resolution {self.output_resolution}")
                return daily_da.isel(time=slice(0, 0))  # Return empty DataArray
                
        except Exception as e:
            self._log_func(f"Error creating time grid with resolution {self.output_resolution}: {e}")
            raise ValueError(f"Failed to create output time grid: {e}")
        
        self._log_func(f"Rolling mode: Using {len(output_times)} time points from {output_times[0]} to {output_times[-1]} "
                    f"(same as standard mode with {self.output_resolution} resolution)")
        self._log_func(f"Rolling window size: {window_size} days, data starts from: {start_time}")
        
        aggregated_data = []
        
        for i, target_time in enumerate(output_times):
            try:
                # Calculate aggregation window: from target_time backwards for window_size days
                window_start = target_time - pd.Timedelta(days=window_size-1)
                # window_end = target_time
                window_end = target_time + pd.Timedelta(hours=23, minutes=59, seconds=59)
                # Note: include the entire target day, the daily daymet data is 12:00:00, so we extend to the end of the day
                
                # Check if window_start is before data start time
                if window_start < start_time:
                    # Not enough historical data for a complete window
                    self._log_func(f"Time point {target_time}: Insufficient data (window starts {window_start}, data starts {start_time})")
                    ref_array = daily_da.isel(time=0)
                    if 'time' in ref_array.dims:
                        ref_array = ref_array.squeeze('time', drop=True)
                    agg_value = ref_array * np.nan
                    agg_value = agg_value.expand_dims(time=[target_time])
                    aggregated_data.append(agg_value)
                    continue
                
                # Select data within the window
                window_data = daily_da.sel(
                    time=slice(window_start, window_end)
                )
                
                if len(window_data.time) > 0:
                    # More strict requirement: need at least 90% of window size data points
                    if window_size >= 250:
                        upper_limit = window_size - 8
                    else:
                        upper_limit = window_size-3
                    min_data_points = max(window_size * 0.9, upper_limit)  # At least 90% or within upper_limit days of full window

                    if len(window_data.time) >= min_data_points:
                        # Apply aggregation method
                        if method == 'mean':
                            agg_value = window_data.mean(dim='time', skipna=True)
                        elif method == 'sum':
                            agg_value = window_data.sum(dim='time', skipna=True,min_count=min_data_points)
                        # same reason as in standard mode for skipna=True in sum
                        elif method == 'min':
                            agg_value = window_data.min(dim='time', skipna=True)
                        elif method == 'max':
                            agg_value = window_data.max(dim='time', skipna=True)
                        else:
                            raise ValueError(f"Unknown aggregation method: {method}")
                        
                        self._log_func(f"Time point {target_time}: Aggregated {len(window_data.time)}/{window_size} days")
                    else:
                        # Not enough data points, create NaN array with same spatial dimensions
                        self._log_func(f"Time point {target_time}: Insufficient data points ({len(window_data.time)}/{window_size} days, need {min_data_points})")
                        agg_value = window_data.isel(time=0) * np.nan
                else:
                    # No data in window, create NaN array with same spatial dimensions as input
                    self._log_func(f"Time point {target_time}: No data in window")
                    ref_array = daily_da.isel(time=0)
                    agg_value = ref_array * np.nan
                
                # Remove time dimension if it exists in agg_value
                if 'time' in agg_value.dims:
                    agg_value = agg_value.squeeze('time', drop=True)
                
                # Add time coordinate to the aggregated value
                agg_value = agg_value.expand_dims(time=[target_time])
                aggregated_data.append(agg_value)
                
            except Exception as e:
                self._log_func(f"Error processing time point {i+1}/{len(output_times)} ({target_time}): {e}")
                # Create NaN array for this time point
                try:
                    ref_array = daily_da.isel(time=0)
                    if 'time' in ref_array.dims:
                        ref_array = ref_array.squeeze('time', drop=True)
                    agg_value = ref_array * np.nan
                    agg_value = agg_value.expand_dims(time=[target_time])
                    aggregated_data.append(agg_value)
                except Exception as e2:
                    self._log_func(f"Failed to create NaN array for time point {target_time}: {e2}")
                    continue
        
        if aggregated_data:
            try:
                # Concatenate along time dimension
                result = xr.concat(aggregated_data, dim='time')
                # Ensure time coordinate is properly sorted
                result = result.sortby('time')
                return result
            except Exception as e:
                self._log_func(f"Error concatenating aggregated data: {e}")
                self._log_func(f"Number of data arrays to concatenate: {len(aggregated_data)}")
                if aggregated_data:
                    self._log_func(f"First array shape: {aggregated_data[0].shape}, dims: {aggregated_data[0].dims}")
                    self._log_func(f"First array time type: {type(aggregated_data[0].time.values[0])}")
                raise ValueError(f"Failed to concatenate rolling aggregation results: {e}")
        else:
            return daily_da.isel(time=slice(0, 0))  # Return empty DataArray

    def _iterate_source_cubes_generic_indices(self,
                                              source_root: str,
                                              target_county_ids_list: Optional[List[str]] = None,
                                              max_segments_per_county: Optional[int] = None):
        if not os.path.isdir(source_root):
            self._log_func(f"Warning: Source root for generic_indices not found: {source_root}. Skipping.")
            return

        all_segment_folders_raw = []
        try:
            all_segment_folders_raw = [
                d for d in os.listdir(source_root)
                if os.path.isdir(os.path.join(source_root, d)) and d.endswith(".zarr")
            ]
            all_segment_folders_raw.sort()
        except Exception as e:
            self._log_func(f"Error listing segment Zarr stores in '{source_root}': {e}")
            return

        segments_to_iterate_paths = []
        if target_county_ids_list:
            for zarr_folder_name in all_segment_folders_raw:
                if any(f"_{re.escape(county_id)}_" in zarr_folder_name for county_id in target_county_ids_list):
                    segments_to_iterate_paths.append(os.path.join(source_root, zarr_folder_name))
            if not segments_to_iterate_paths:
                return
        else:
            segments_to_iterate_paths = [os.path.join(source_root, name) for name in all_segment_folders_raw]

        if not segments_to_iterate_paths:
            return
        
        segments_by_county = defaultdict(list)
        for seg_path in segments_to_iterate_paths:
            match = re.search(r"_(\d{5})_", os.path.basename(seg_path))
            if match:
                county_code = match.group(1)
                segments_by_county[county_code].append(seg_path)
        
        final_segments_to_process_paths = []
        sorted_county_codes = sorted(list(segments_by_county.keys()))
        if target_county_ids_list:
             sorted_county_codes = [code for code in sorted_county_codes if code in target_county_ids_list]

        for county_code in sorted_county_codes:
            county_segments = sorted(segments_by_county[county_code])
            if max_segments_per_county is not None:
                final_segments_to_process_paths.extend(county_segments[:max_segments_per_county])
            else:
                final_segments_to_process_paths.extend(county_segments)
        
        for source_segment_path in final_segments_to_process_paths:
            segment_id = os.path.basename(source_segment_path)
            try:
                cube_names_in_segment = [
                    c for c in os.listdir(source_segment_path)
                    if c.isdigit() and os.path.isdir(os.path.join(source_segment_path, c))
                ]
                cube_names_in_segment.sort(key=int)
            except Exception as e:
                self._log_func(f"Warning: Error listing cubes in '{source_segment_path}': {str(e)}. Skipping segment.")
                continue
            if not cube_names_in_segment: continue
            for cube_id_str in cube_names_in_segment:
                source_cube_path = os.path.join(source_segment_path, cube_id_str)
                yield source_cube_path, segment_id, cube_id_str

    def _iterate_source_cubes_daymet_prcp(self,
                                        source_root: str,
                                        data_structure: str,
                                        target_county_ids_list: Optional[List[str]] = None,
                                        max_segments_per_county: Optional[int] = None):
        """Iterate through daymet data cubes - supports both nested and direct zarr structures"""
        
        for county_folder_name in os.listdir(source_root):
            county_folder_path = os.path.join(source_root, county_folder_name)
            
            if not os.path.isdir(county_folder_path):
                continue
                
            # Extract county code from folder name
            county_match = re.search(r"daymet_(\d{5})_", county_folder_name)
            if not county_match:
                continue
                
            county_code = county_match.group(1)
            
            # Skip if not in target list
            if target_county_ids_list is not None and county_code not in target_county_ids_list:
                continue
                
            print(f"Processing county: {county_code} (structure: {data_structure})")
            
            # Use data_structure parameter to determine structure
            if data_structure == 'direct_zarr' or county_folder_name.endswith('.zarr'):
                # Direct zarr: daymet_01003_0-9.zarr/0/variable
                zarr_base_path = county_folder_path
                segment_id = county_folder_name  # Use the zarr folder name as segment_id
                print(f"  Using direct zarr structure: {zarr_base_path}")
            else:
                # Nested zarr: daymet_01003_0-9/daymet_01003_0-9.zarr/0/variable
                zarr_base_path = os.path.join(county_folder_path, county_folder_name + '.zarr')
                segment_id = county_folder_name + '.zarr'  # Use the zarr folder name as segment_id
                print(f"  Using nested zarr structure: {zarr_base_path}")
                
            if not os.path.isdir(zarr_base_path):
                print(f"  Warning: Zarr directory not found: {zarr_base_path}")
                continue
                
            # Find cube folders (numbered directories like 0, 1, 2, etc.)
            try:
                cube_folders = [d for d in os.listdir(zarr_base_path) 
                            if d.isdigit() and os.path.isdir(os.path.join(zarr_base_path, d))]
                cube_folders.sort(key=int)
                
                if not cube_folders:
                    print(f"  Warning: No cube folders found in {zarr_base_path}")
                    continue
                    
                print(f"  Found {len(cube_folders)} cube folders: {cube_folders}")
                
                # Limit cubes if specified
                if max_segments_per_county is not None:
                    cube_folders = cube_folders[:max_segments_per_county]
                    
            except Exception as e:
                print(f"  Error listing cube folders: {e}")
                continue
                
            # Yield each cube in the original format (source_cube_path, segment_id, cube_id_str)
            for cube_folder in cube_folders:
                cube_path = os.path.join(zarr_base_path, cube_folder)
                yield cube_path, segment_id, cube_folder

    def _iterate_source_cubes(self,
                              variable_name_to_process: str,
                              target_county_ids_list: Optional[List[str]] = None,
                              max_segments_per_county: Optional[int] = None):
        var_config = self.variables_config[variable_name_to_process]
        source_root = var_config['source_root']
        source_type = var_config['source_type']
        data_structure = var_config.get('data_structure', 'direct_zarr')

        if source_type == 'generic_indices':
            yield from self._iterate_source_cubes_generic_indices(
                source_root, target_county_ids_list, max_segments_per_county
            )
        elif source_type == 'daymet_prcp':
            yield from self._iterate_source_cubes_daymet_prcp(
                source_root, data_structure, target_county_ids_list, max_segments_per_county
            )
        else:
            raise ValueError(f"Unknown source_type: {source_type} for variable {variable_name_to_process}")

    def aggregate_indices(self,
                          target_county_ids_list: Optional[List[str]] = None,
                          max_segments_per_county: Optional[int] = None,
                          output_time_chunk_size: Optional[int] = None):
        self._log_func(f"\nStarting unified temporal aggregation process for period '{self.aggregation_period}' in {self.mode} mode...")
        if self.mode == "rolling":
            self._log_func(f" Rolling window: {self.aggregation_period}, Output resolution: {self.output_resolution}")
        if target_county_ids_list: 
            self._log_func(f" Targeting County IDs: {target_county_ids_list}")
        if max_segments_per_county is not None: 
            self._log_func(f" Limiting to a maximum of {max_segments_per_county} segments per county.")

        for var_name, var_config in self.variables_config.items():
            self._log_func(f"\nProcessing variable: '{var_name}'")
            
            # Get output variable name (with suffix for rolling mode)
            output_var_name = self._get_variable_output_name(var_name)
            
            current_agg_method = var_config['method']
            consolidated_open = var_config.get('consolidated_open', True)
            output_mode = var_config.get('output_mode', 'w')
            skip_logic = var_config.get('skip_logic', 'check_zattrs' if output_mode == 'w' else 'check_var_exists')
            chunk_ref_var = var_config.get('chunk_ref_var', var_name)
            # If output_time_chunk_size not specified, will auto-detect from input data later
            current_output_time_chunk_size = var_config.get('output_time_chunk_size', output_time_chunk_size)
            target_spatial_chunks_config = var_config.get('target_spatial_chunks', None)

            self._log_func(f"  Config for '{var_name}' -> '{output_var_name}': agg_method='{current_agg_method}', output_mode='{output_mode}', "
                           f"skip_logic='{skip_logic}', time_chunks={current_output_time_chunk_size}, spatial_chunks={target_spatial_chunks_config}")

            iterable_cubes = list(self._iterate_source_cubes(var_name, target_county_ids_list, max_segments_per_county))
            total_cubes_to_process = len(iterable_cubes)
            self._log_func(f"  Found {total_cubes_to_process} source cubes for '{var_name}'.")

            if total_cubes_to_process == 0:
                self._log_func(f"  No cubes found for variable '{var_name}' with current filters. Source root: {var_config['source_root']}")
                continue

            for source_cube_path, segment_id, cube_id_str in tqdm(
                    iterable_cubes, total=total_cubes_to_process, desc=f"Aggregating {var_name}"):
                
                output_segment_dir = os.path.join(self.output_aggregated_root, segment_id)
                output_cube_path = os.path.join(output_segment_dir, cube_id_str)

                can_skip = False
                if skip_logic == 'check_var_exists':
                    if os.path.isdir(output_cube_path) and os.path.isdir(os.path.join(output_cube_path, output_var_name)):
                        can_skip = True
                elif skip_logic == 'check_zattrs':
                    if output_mode == 'w' and os.path.exists(os.path.join(output_cube_path, '.zattrs')):
                       can_skip = True
                
                if can_skip:
                    continue
                
                try:
                    os.makedirs(output_cube_path, exist_ok=True)
                    
                    with xr.open_zarr(source_cube_path, consolidated=consolidated_open) as daily_ds:
                        # Extract spatial reference information if present in source data
                        # This preserves CRS/projection metadata for GIS compatibility
                        source_spatial_ref = None
                        if 'spatial_ref' in daily_ds:
                            source_spatial_ref = daily_ds['spatial_ref'].copy(deep=True)
                            self._log_func(f"  Found spatial_ref in source data for {segment_id}/{cube_id_str}")
                        
                        if var_name not in daily_ds and chunk_ref_var not in daily_ds and not list(daily_ds.data_vars):
                             self._log_func(f"Warning: Neither var_name '{var_name}' nor chunk_ref_var '{chunk_ref_var}' found in {source_cube_path}, and no other vars. Skipping cube.")
                             continue
                        
                        spatial_chunks_template = {}
                        if target_spatial_chunks_config:
                            if 'y' in target_spatial_chunks_config and isinstance(target_spatial_chunks_config['y'], int):
                                spatial_chunks_template['y'] = target_spatial_chunks_config['y']
                            if 'x' in target_spatial_chunks_config and isinstance(target_spatial_chunks_config['x'], int):
                                spatial_chunks_template['x'] = target_spatial_chunks_config['x']
                        
                        if not ('y' in spatial_chunks_template and 'x' in spatial_chunks_template):
                            ref_da_for_chunks_calc = None
                            if chunk_ref_var in daily_ds: ref_da_for_chunks_calc = daily_ds[chunk_ref_var]
                            elif var_name in daily_ds : ref_da_for_chunks_calc = daily_ds[var_name]
                            elif daily_ds.data_vars: 
                                first_var_in_ds = next(iter(daily_ds.data_vars))
                                ref_da_for_chunks_calc = daily_ds[first_var_in_ds]
                                if not target_spatial_chunks_config:
                                     self._log_func(f"Debug: Using first var '{first_var_in_ds}' for chunk template for {var_name} in {source_cube_path}")
                            
                            if ref_da_for_chunks_calc is not None and ref_da_for_chunks_calc.chunks:
                                ref_chunks_tuples = ref_da_for_chunks_calc.chunks
                                for i, dim_name_src in enumerate(ref_da_for_chunks_calc.dims):
                                    if len(ref_chunks_tuples) > i and ref_chunks_tuples[i] and len(ref_chunks_tuples[i]) > 0:
                                        if dim_name_src == 'y' and 'y' not in spatial_chunks_template:
                                            spatial_chunks_template['y'] = ref_chunks_tuples[i][0]
                                        elif dim_name_src == 'x' and 'x' not in spatial_chunks_template:
                                            spatial_chunks_template['x'] = ref_chunks_tuples[i][0]
                        
                        if 'y' not in spatial_chunks_template: spatial_chunks_template['y'] = 'auto'
                        if 'x' not in spatial_chunks_template: spatial_chunks_template['x'] = 'auto'

                        if var_name not in daily_ds:
                            self._log_func(f"Warning: Variable '{var_name}' not found in source cube {source_cube_path}. Skipping aggregation for this var.")
                            continue
                            
                        daily_da = daily_ds[var_name]
                        if 'time' not in daily_da.dims or daily_da.sizes['time'] == 0:
                            self._log_func(f"Warning: Variable '{var_name}' in {source_cube_path} has no 'time' dim or is empty. Skipping.")
                            continue
                        
                        aggregated_da = None
                        try:
                            if self.mode == "standard":
                                aggregated_da = self._aggregate_standard_mode(daily_da, current_agg_method)
                            elif self.mode == "rolling":
                                aggregated_da = self._aggregate_rolling_mode(daily_da, current_agg_method)
                            else:
                                raise ValueError(f"Unknown mode: {self.mode}")
                            
                            aggregated_da.attrs = daily_da.attrs.copy()
                            # Force set grid_mapping attribute to ensure GIS software can locate the CRS
                            # This is critical even if the source had it, as resampling might drop it
                            aggregated_da.attrs['grid_mapping'] = 'spatial_ref'
                            
                            if self.mode == "rolling":
                                aggregated_da.attrs['aggregation_method'] = f"{current_agg_method} over {self.aggregation_period} window, every {self.output_resolution}"
                            else:
                                aggregated_da.attrs['aggregation_method'] = f"{current_agg_method} over {self.aggregation_period}"
                            
                            # Use output variable name (with suffix for rolling mode)
                            aggregated_da.name = output_var_name
                        except Exception as e_agg:
                            self._log_func(f"Error aggregating variable '{var_name}' in {source_cube_path}: {e_agg}")
                            continue
                        
                        if aggregated_da is not None:
                            # Auto-detect time chunk size from input data if not specified
                            if current_output_time_chunk_size is None:
                                # Try to get chunk size from input data
                                if daily_da.chunks and 'time' in daily_da.dims:
                                    time_dim_idx = list(daily_da.dims).index('time')
                                    if len(daily_da.chunks) > time_dim_idx and daily_da.chunks[time_dim_idx]:
                                        current_output_time_chunk_size = daily_da.chunks[time_dim_idx][0]
                                        self._log_func(f"  Auto-detected time chunk size: {current_output_time_chunk_size} from input data")
                                # Fallback to 4 if auto-detection fails
                                if current_output_time_chunk_size is None:
                                    current_output_time_chunk_size = 4
                                    self._log_func(f"  Using default time chunk size: {current_output_time_chunk_size}")
                            
                            target_chunks_dict = {}
                            if 'time' in aggregated_da.dims:
                                actual_time_size = aggregated_da.sizes['time']
                                target_chunks_dict['time'] = min(current_output_time_chunk_size, actual_time_size) if actual_time_size > 0 else 1

                            def get_spatial_chunk_val(dim_name_key, dim_size, template_val_from_dict):
                                if dim_size == 0: return 1
                                if template_val_from_dict == 'auto': return dim_size
                                if isinstance(template_val_from_dict, int): return min(template_val_from_dict, dim_size)
                                return dim_size

                            if 'y' in aggregated_da.dims:
                                target_chunks_dict['y'] = get_spatial_chunk_val('y', aggregated_da.sizes['y'], spatial_chunks_template.get('y'))
                            if 'x' in aggregated_da.dims:
                                target_chunks_dict['x'] = get_spatial_chunk_val('x', aggregated_da.sizes['x'], spatial_chunks_template.get('x'))
                            
                            final_chunks_for_var = {k: max(1, v) for k, v in target_chunks_dict.items() if k in aggregated_da.dims}
                            
                            dataset_for_saving = xr.Dataset()
                            encoding_out = {}
                            
                            # Add spatial reference to output dataset if it was present in source
                            # This ensures output files maintain geospatial metadata for GIS tools
                            if source_spatial_ref is not None:
                                dataset_for_saving['spatial_ref'] = source_spatial_ref
                                # Note: spatial_ref is not added to encoding_out, so it will use
                                # default zarr encoding (no chunking, which is correct for metadata)
                            
                            if final_chunks_for_var and aggregated_da.size > 0:
                                dataset_for_saving[output_var_name] = aggregated_da.chunk(final_chunks_for_var)
                                encoding_chunks_tuple = tuple(final_chunks_for_var[d] for d in aggregated_da.dims if d in final_chunks_for_var)
                                if len(encoding_chunks_tuple) == len(aggregated_da.dims):
                                    encoding_out[output_var_name] = {'chunks': encoding_chunks_tuple}
                                elif aggregated_da.chunks:
                                     encoding_out[output_var_name] = {'chunks': aggregated_da.chunksizes}
                            else:
                                dataset_for_saving[output_var_name] = aggregated_da
                                if aggregated_da.chunks:
                                    encoding_out[output_var_name] = {'chunks': aggregated_da.chunksizes}
                            
                            if dataset_for_saving.data_vars:
                                dataset_for_saving.to_zarr(output_cube_path,
                                                          mode=output_mode,
                                                          consolidated=True,
                                                          encoding=encoding_out if encoding_out else None)
                
                except Exception as e:
                    self._log_func(f"Error processing source cube {source_cube_path} for var '{var_name}': {e}")
                    if "Too many open files" in str(e) or "OSError: [Errno 24]" in str(e):
                        self._log_func("Pausing due to 'Too many open files' error. Will resume in 60 seconds.")
                        time.sleep(60)
                    else:
                        self._log_func(f"Traceback for error on {source_cube_path}:\n{traceback.format_exc()}")
            self._log_func(f"  Finished processing variable: '{var_name}' -> '{output_var_name}'")
        self._log_func(f"\nUnified temporal aggregation process finished for {self.output_aggregated_root}.")