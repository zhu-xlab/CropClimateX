#%%
"""
Docstring for Calculator_Week_Palmer
This module implements the PalmerAggregatedCalculator class for calculating Palmer Drought Indices and RWD/RSM
This code contains some experimental methods, but only the "psuedo_weekly" method is supported for actual use.
"""
# PalmerAggregatedCalculator.py
import os
import numpy as np
import re
from typing import Dict, List, Union, Tuple, Optional
import xarray as xr
import pandas as pd # Ensure pandas is imported for time handling in _create_empty_timeseries_like
import shutil
import warnings 
import time     
import traceback
# check the handout for understanding the code
class PalmerAggregatedCalculator:
    def __init__(self,
                aggregated_daymet_segment_path: str,
                soil_segment_path: Optional[str],
                num_cubes_agg_daymet: int,
                num_cubes_soil: Optional[int],
                palmer_output_root: Optional[str] = None,
                pdsi_persistence_factor: float = 0.897,
                phdi_persistence_factor: float = 0.897,
                z_index_divisor: float = 3.0,
                enable_nan_handling: bool = False,
                calibration_end_date: str = '2015-01-01'
                ):

        if not os.path.exists(aggregated_daymet_segment_path) or not os.path.isdir(aggregated_daymet_segment_path):
            raise FileNotFoundError(f"Aggregated Daymet segment path does not exist or is not a directory: {aggregated_daymet_segment_path}")
        self.aggregated_daymet_segment_path = aggregated_daymet_segment_path

        if soil_segment_path and (not os.path.exists(soil_segment_path) or not os.path.isdir(soil_segment_path)):
            print(f"Warning: Provided soil_segment_path '{soil_segment_path}' does not exist or is not a directory. PHDI/PDSI SMA step might fail if AWC is needed.")
            self.soil_segment_path = None
            self.num_cubes_soil = 0
        else:
            self.soil_segment_path = soil_segment_path
            self.num_cubes_soil = num_cubes_soil if num_cubes_soil is not None else 0

        self.num_cubes_agg_daymet = num_cubes_agg_daymet

        self.segment_name_agg_daymet = os.path.basename(self.aggregated_daymet_segment_path).replace(".zarr","")
        self.segment_name_soil = os.path.basename(self.soil_segment_path).replace(".zarr","") if self.soil_segment_path else "no_soil_segment_provided"

        # Ensure the segment names are valid for use in output paths
        if palmer_output_root:
            self.palmer_output_root = palmer_output_root
            self.palmer_output_segment_path = os.path.join(palmer_output_root, self.segment_name_agg_daymet + ".zarr")
            # make sure the output path exists
            if not os.path.exists(self.palmer_output_segment_path):
                os.makedirs(self.palmer_output_segment_path, exist_ok=True)
            print(f"Palmer results will be saved to: {self.palmer_output_segment_path}")
        else:
            # If no output root is provided, use the original aggregated segment path
            self.palmer_output_root = None
            self.palmer_output_segment_path = self.aggregated_daymet_segment_path
            print(f"Palmer results will be saved to original segment path: {self.palmer_output_segment_path}")

        match_agg = re.search(r"_(\d+)-(\d+)\.zarr$", self.aggregated_daymet_segment_path)
        if not match_agg: raise ValueError(f"Invalid Zarr path format for aggregated daymet segment: {self.aggregated_daymet_segment_path}")
        self.offset_agg_daymet = int(match_agg.group(1))

        self.offset_soil = 0
        if self.soil_segment_path:
            match_soil = re.search(r"_(\d+)-(\d+)\.zarr$", self.soil_segment_path)
            if not match_soil:
                print(f"Warning: Could not parse segment start from soil path {self.soil_segment_path}. Assuming offset 0 for soil cubes if used.")
            else:
                self.offset_soil = int(match_soil.group(1))

        self.pdsi_persistence_factor = pdsi_persistence_factor
        self.phdi_persistence_factor = phdi_persistence_factor
        self.z_index_divisor = z_index_divisor
        self.enable_nan_handling = enable_nan_handling
        self.calibration_end_date = calibration_end_date

    def _get_cube_path_from_segment(self, segment_path: str, segment_offset: int, cube_idx_in_segment: int) -> str:
        abs_cube_num = segment_offset + cube_idx_in_segment
        return os.path.join(segment_path, str(abs_cube_num))

    def _handle_nans(self, da: Optional[xr.DataArray], var_name: str, s_name: str, c_idx_abs: int) -> Optional[xr.DataArray]:
        if da is None: return None
        if da.size == 0: return da
        
        # If NaN handling is disabled, just compute dask arrays and return without NaN preprocessing
        if not self.enable_nan_handling:
            if hasattr(da, 'chunks') and da.chunks is not None:
                try:
                    return da.compute()
                except Exception:
                    return da
            return da

        loaded_da = da
        if hasattr(da, 'chunks') and da.chunks is not None:
            try:
                loaded_da = da.compute()
            except Exception as e_load:
                print(f"   Warning: Could not compute dask array {var_name} for NaN handling in cube {c_idx_abs}, seg {s_name}: {e_load}.")
                return da # Return original Dask array on failure if compute fails

        # Ensure loaded_da is not None after potential compute failure (though caught by return da)
        if loaded_da is None : return None 

        # Proceed with NaN checking on loaded_da
        # Make sure to handle cases where .all() or .sum() might return a DataArray (e.g. if loaded_da was scalar)
        if loaded_da.ndim == 0: # Scalar DataArray
             if loaded_da.isnull().item(): # Check if scalar is NaN
                  return loaded_da # Return as is if scalar NaN
        elif loaded_da.isnull().all().item(): # For multi-dimensional, check if all are NaN
             return loaded_da

        nan_count = loaded_da.isnull().sum()
        if isinstance(nan_count, xr.DataArray): # If sum returns a DataArray (e.g. sum over one dim)
            nan_count = nan_count.item() if nan_count.size == 1 else nan_count.sum().item() # Get total NaN count

        if nan_count > 0:
            mean_val = loaded_da.mean(skipna=True)
            # Handle mean_val if it's a DataArray (e.g. from a spatial mean) vs scalar
            if isinstance(mean_val, xr.DataArray):
                if mean_val.size == 1: # If it became a scalar DataArray
                    mean_val_item = mean_val.item()
                    if np.isnan(mean_val_item): return loaded_da
                    return loaded_da.fillna(mean_val_item)
                else: # mean_val is still an array, fillna will attempt to broadcast
                    return loaded_da.fillna(mean_val) 
            else: # mean_val is a scalar (e.g. float)
                if np.isnan(mean_val): return loaded_da
                return loaded_da.fillna(mean_val)
        return loaded_da

        # ============================================================
        # ORIGINAL VERSION - COMMENTED OUT FOR REFERENCE
        # ============================================================
        # def _palmer_state_machine_index_full(self, Z: xr.DataArray, persistence_factor: float = 0.897) -> Tuple[xr.DataArray, xr.DataArray]:
        # """
        # Palmer state machine full implementation following classic Palmer logic with probability calculations
        
        # Returns:
        #     Tuple[xr.DataArray, xr.DataArray]: (phdi_x3, pdsi_comprehensive)
        # """
        # dry_spell_threshold = 1.0  
        # wet_spell_threshold = 1.0  
        # drought_recovery_damping = 1.0
        # z_negative_multiplier = 1.0
        # z_positive_multiplier = 1.0
        # # Classic Palmer empirical coefficients
        # COEFF_2691 = 2.691
        # THRESHOLD_015 = 0.15  
        # THRESHOLD_15 = 1.5
        # Z_DIVISOR = 3.0  # Classic Palmer uses 3.0
        # establishment_threshold = 1.0 # Classic Palmer uses 1.0 for wet/drought establishment, but this can be adjusted

        # if Z is None or Z.ndim == 0 or 'time' not in Z.dims or Z.sizes.get('time', 0) == 0:
        #     # Return empty tuple
        #     empty_template = self._create_empty_template_from_z(Z)
        #     return empty_template, empty_template
        
        # Z_computed = Z.compute() if hasattr(Z, 'chunks') and Z.chunks is not None else Z
        
        # # Initialize spatial template
        # spatial_template = Z_computed.isel(time=0, drop=True, missing_dims='ignore')
        # if 'time' in spatial_template.dims:
        #     spatial_template = spatial_template.squeeze('time', drop=True)
        
        # zeros_spatial = xr.DataArray(np.zeros_like(spatial_template.data, dtype=Z_computed.dtype),
        #                             coords=spatial_template.coords, dims=spatial_template.dims)
        
        # # State variables initialization
        # x1 = zeros_spatial.copy(deep=True)  # Incipient wet spell
        # x2 = zeros_spatial.copy(deep=True)  # Incipient dry spell  
        # x3 = zeros_spatial.copy(deep=True)  # Established period (PHDI main component)
        # v = zeros_spatial.copy(deep=True)   # Accumulated departure
        # prob = zeros_spatial.copy(deep=True)  # Probability of ending
        
        # # Storage for results
        # phdi_x3_list = []  # PHDI = x3
        # pdsi_comprehensive_list = []  # PDSI = comprehensive logic result
        
        # for t_idx in range(Z_computed.sizes['time']):
        #     z_t = Z_computed.isel(time=t_idx).fillna(0)

        #     z_t = xr.where(z_t < 0, z_negative_multiplier * z_t, z_t * z_positive_multiplier)  # Apply negative multiplier

        #     time_coord_val = Z_computed.time.data[t_idx]
            
        #     # Classic Palmer state classification based on x3 only
        #     no_established_period = (x3 == 0)                    # No established period
        #     near_normal = (np.abs(x3) <= 0.5)                   # Close to normal conditions
        #     wet_spell = (x3 > 0.5)                              # Established wet spell
        #     drought_spell = (x3 < -0.5)                         # Established drought spell
            
        #     # Initialize current step outputs
        #     px1 = zeros_spatial.copy(deep=True)
        #     px2 = zeros_spatial.copy(deep=True) 
        #     px3 = zeros_spatial.copy(deep=True)
        #     pv = zeros_spatial.copy(deep=True)
        #     pprob = zeros_spatial.copy(deep=True)  # Current probability
            
        #     # Classic Palmer logic: Near normal state processing
        #     px1 = xr.where(near_normal, np.maximum(0, persistence_factor * x1 + z_t / Z_DIVISOR), px1)
        #     px2 = xr.where(near_normal, np.minimum(0, persistence_factor * x2 + z_t / Z_DIVISOR), px2)
        #     px3 = xr.where(near_normal & no_established_period, 0.0, px3)  # Reset x3 only when no established period
        #     pv = xr.where(near_normal, 0.0, pv)
        #     pprob = xr.where(near_normal, 0.0, pprob)
            
        #     # Classic Palmer trigger conditions - use threshold 1.0
        #     new_wet = no_established_period & (px1 >= wet_spell_threshold)
        #     px3 = xr.where(new_wet, px1, px3)
        #     px1 = xr.where(new_wet, 0.0, px1)
        #     pprob = xr.where(new_wet, 0.0, pprob)
            
        #     # Check new drought period start
        #     new_drought = no_established_period & (px2 <= (-1)*(dry_spell_threshold)) & ~new_wet
        #     px3 = xr.where(new_drought, px2, px3)
        #     px2 = xr.where(new_drought, 0.0, px2)
        #     pprob = xr.where(new_drought, 0.0, pprob)
            
        #     # Wet spell processing
        #     wet_intensifies = wet_spell & (z_t >= THRESHOLD_015)
        #     wet_abates = wet_spell & (z_t < THRESHOLD_015)
            
        #     # Wet spell intensification
        #     px3 = xr.where(wet_intensifies, persistence_factor * x3 + z_t / Z_DIVISOR, px3)
        #     px1 = xr.where(wet_intensifies, 0.0, px1)
        #     px2 = xr.where(wet_intensifies, 0.0, px2)
        #     pv = xr.where(wet_intensifies, 0.0, pv)
        #     pprob = xr.where(wet_intensifies, 0.0, pprob)
            
        #     # palmer.py: data["ud"] = data["z"][year, month] - 0.15
        #     # palmer.py: data["pv"] = data["ud"] + min(data["v"], 0.0)
        #     temp_v_wet_abate = z_t - THRESHOLD_015 + np.minimum(v, 0.0)  
            
        #     # Convert to classic Palmer formula for ending probability
        #     ze_wet = -COEFF_2691 * x3 + THRESHOLD_15  # For wet spells
        #     q_wet = xr.where(prob == 0, ze_wet, ze_wet + v)
        #     ending_prob_wet = xr.where(q_wet != 0, (temp_v_wet_abate / q_wet) * 100, 0.0)
        #     ending_prob_wet = ending_prob_wet.clip(0, 100)
            
        #     # Wet spell ending conditions: probability >= 100 or v condition met
        #     wet_ending = wet_abates & ((ending_prob_wet >= 100) | (temp_v_wet_abate >= 0))
        #     wet_continuing = wet_abates & ~wet_ending

        #     # Decay equations for x1 and x2
        #     px1_decay_calc = np.maximum(0, persistence_factor * x1 + z_t / Z_DIVISOR)
        #     px2_decay_calc = np.minimum(0, persistence_factor * x2 + z_t / Z_DIVISOR)

        #     # Wet spell ending - return to near normal
        #     px3 = xr.where(wet_ending, 0.0, px3)
        #     px1 = xr.where(wet_ending, px1_decay_calc, px1)
        #     px2 = xr.where(wet_ending, px2_decay_calc, px2)
        #     pv = xr.where(wet_ending, 0.0, pv)
        #     pprob = xr.where(wet_ending, 0.0, pprob)
            
        #     # Wet spell continuing but weakening
        #     px3 = xr.where(wet_continuing, 
        #                 xr.where(ending_prob_wet >= 100, 0.0, 
        #                         persistence_factor * x3 + z_t / Z_DIVISOR), px3)
        #     pv = xr.where(wet_continuing, temp_v_wet_abate, pv)
        #     pprob = xr.where(wet_continuing, ending_prob_wet.clip(0, 100), pprob)
            
        #     # Drought spell processing
        #     drought_intensifies = drought_spell & (z_t <= -THRESHOLD_015)
        #     drought_abates = drought_spell & (z_t > -THRESHOLD_015)
            
        #     # Drought spell intensification
        #     px3 = xr.where(drought_intensifies, persistence_factor * x3 + z_t / Z_DIVISOR, px3)
        #     px1 = xr.where(drought_intensifies, 0.0, px1)
        #     px2 = xr.where(drought_intensifies, 0.0, px2)
        #     pv = xr.where(drought_intensifies, 0.0, pv)
        #     pprob = xr.where(drought_intensifies, 0.0, pprob)

        #     # palmer.py: data["uw"] = data["z"][year, month] + 0.15
        #     # palmer.py: data["pv"] = data["uw"] + max(data["v"], 0.0)
        #     temp_v_drought_abate = z_t * drought_recovery_damping + THRESHOLD_015 + np.maximum(v, 0.0)

        #     # Calculate ending probability for drought spells
        #     ze_drought = -COEFF_2691 * x3 - THRESHOLD_15  # For drought spells
        #     q_drought = xr.where(prob == 0, ze_drought, ze_drought + v)
        #     ending_prob_drought = xr.where(q_drought != 0, (temp_v_drought_abate / q_drought) * 100, 0.0)
        #     ending_prob_drought = ending_prob_drought.clip(0, 100)

        #     # Drought spell ending conditions: probability >= 100 or v condition met
        #     drought_ending = drought_abates & ((ending_prob_drought >= 100) | (temp_v_drought_abate <= 0))
        #     drought_continuing = drought_abates & ~drought_ending
            
        #     # Drought spell ending - return to near normal
        #     px3 = xr.where(drought_ending, 0.0, px3)
        #     px1 = xr.where(drought_ending, px1_decay_calc, px1)
        #     px2 = xr.where(drought_ending, px2_decay_calc, px2)
        #     pv = xr.where(drought_ending, 0.0, pv)
        #     pprob = xr.where(drought_ending, 0.0, pprob)
            
        #     # Drought spell continuing but weakening
        #     px3 = xr.where(drought_continuing,
        #                 xr.where(ending_prob_drought >= 100, 0.0,
        #                         persistence_factor * x3 + z_t / Z_DIVISOR), px3)
        #     pv = xr.where(drought_continuing, temp_v_drought_abate, pv)
        #     pprob = xr.where(drought_continuing, ending_prob_drought.clip(0, 100), pprob)
            
        #     # PDSI comprehensive logic 
        #     pdsi_value = zeros_spatial.copy(deep=True)
            
        #     # Probability-based logic for established periods
        #     established_periods = (np.abs(px3) > 0.5)
        #     final_prob = pprob / 100.0  
            
        #     # Classic Palmer case logic with probability weighting
        #     # When established period exists, use x3 directly
        #     use_x3_directly = established_periods & ((pprob <= 0) | (pprob >= 100))
        #     pdsi_value = xr.where(use_x3_directly, px3, pdsi_value)

        #     # When probability weighting exists, use weighted combination
        #     has_probability_weight = established_periods & (pprob > 0) & (pprob < 100)

        #     # For drought periods (x3 <= 0), use (1-prob)*x3 + prob*x1
        #     drought_with_prob = has_probability_weight & (px3 <= 0)
        #     pdsi_value = xr.where(drought_with_prob, 
        #                         (1.0 - final_prob) * px3 + final_prob * px1, 
        #                         pdsi_value)

        #     # For wet periods (x3 > 0), use (1-prob)*x3 + prob*x2
        #     wet_with_prob = has_probability_weight & (px3 > 0)
        #     pdsi_value = xr.where(wet_with_prob,
        #                         (1.0 - final_prob) * px3 + final_prob * px2,
        #                         pdsi_value)

        #     # When near normal conditions exist, choose appropriate incipient values
        #     in_near_normal = (np.abs(px3) <= 0.5)

        #     # If x3=0 and both incipient values exist, choose the one with the larger absolute value
        #     both_exist = in_near_normal & (px1 != 0) & (px2 != 0)
        #     choose_x1 = both_exist & (np.abs(px1) > np.abs(px2))  
        #     choose_x2 = both_exist & (np.abs(px1) <= np.abs(px2))
            
        #     pdsi_value = xr.where(choose_x1, px1, pdsi_value)
        #     pdsi_value = xr.where(choose_x2, px2, pdsi_value)

        #     # If only one exists, use that one
        #     only_x1 = in_near_normal & (px1 != 0) & (px2 == 0)
        #     only_x2 = in_near_normal & (px1 == 0) & (px2 != 0)
            
        #     pdsi_value = xr.where(only_x1, px1, pdsi_value)
        #     pdsi_value = xr.where(only_x2, px2, pdsi_value)

        #     # Add debug output for first 15 and last 15 time steps
        #     if t_idx < 15 or t_idx >= (Z_computed.sizes['time'] - 15):
        #         z_mean = z_t.mean().item()
        #         x1_mean = px1.mean().item()
        #         x2_mean = px2.mean().item()
        #         x3_mean = px3.mean().item()
        #         pdsi_mean = pdsi_value.mean().item()
        #         prob_mean = pprob.mean().item()
        #         v_mean = pv.mean().item() 
                
        #         # Count trigger events
        #         trigger_wet_count = (px1 >= 1.0).sum().item() if (px1 >= 1.0).any() else 0
        #         trigger_drought_count = (px2 <= -1.0).sum().item() if (px2 <= -1.0).any() else 0
        #         prob_events = (pprob > 0).sum().item() if (pprob > 0).any() else 0
                
        #         print(f"DEBUG Palmer t={t_idx}: Z={z_mean:.4f}, "
        #             f"x1={x1_mean:.4f}, x2={x2_mean:.4f}, x3={x3_mean:.4f}, "
        #             f"pdsi={pdsi_mean:.4f}, prob={prob_mean:.2f}%, v={v_mean:.4f}, "
        #             f"wet_triggers={trigger_wet_count}, drought_triggers={trigger_drought_count}, "
        #             f"prob_events={prob_events}")
                
        #         if trigger_wet_count > 0 or trigger_drought_count > 0 or prob_events > 0:
        #             print(f"    *** EVENT at t={t_idx}: triggers={trigger_wet_count+trigger_drought_count}, prob_events={prob_events} ***")
            
        #     # Store results with time coordinate
        #     phdi_result = px3.copy(deep=True)
        #     pdsi_result = pdsi_value.copy(deep=True)
            
        #     # Assign time coordinate
        #     if 'time' in Z_computed.coords:
        #         phdi_result = phdi_result.assign_coords({'time': time_coord_val}).expand_dims('time')
        #         pdsi_result = pdsi_result.assign_coords({'time': time_coord_val}).expand_dims('time')
            
        #     phdi_x3_list.append(phdi_result)
        #     pdsi_comprehensive_list.append(pdsi_result)
            
        #     # Update state variables for next iteration
        #     x1 = px1.copy(deep=True)
        #     x2 = px2.copy(deep=True)
        #     x3 = px3.copy(deep=True)
        #     v = pv.copy(deep=True)
        #     prob = pprob.copy(deep=True)
        
        # # Concatenate all time steps
        # try:
        #     phdi_final = xr.concat(phdi_x3_list, dim='time')
        #     pdsi_final = xr.concat(pdsi_comprehensive_list, dim='time')
            
        #     # Set proper attributes
        #     phdi_final.name = 'phdi'
        #     phdi_final.attrs = {
        #         'long_name': 'Palmer Hydrological Drought Index',
        #         'description': 'x3 component from Palmer state machine with probability calculations - established drought/wet periods',
        #         'units': 'dimensionless',
        #         'calculation_method': 'Classic Palmer state machine with probability weighting and threshold 1.0'
        #     }
            
        #     pdsi_final.name = 'pdsi'
        #     pdsi_final.attrs = {
        #         'long_name': 'Palmer Drought Severity Index',
        #         'description': 'Comprehensive drought index using Palmer state machine with probability-weighted logic',
        #         'units': 'dimensionless',
        #         'calculation_method': 'Classic Palmer comprehensive logic with probability weighting combining x1, x2, x3 components'
        #     }
            
        #     return phdi_final, pdsi_final
            
        # except Exception as e:
        #     print(f"Error in Palmer state machine concatenation: {e}")
        #     empty_template = self._create_empty_template_from_z(Z)
        #     return empty_template, empty_template

    """
    In original script of Palmer (Palmer,1965), the definition of PDSI is not proposed clearly. Therefore, it is open to interpretation.
    Palmer introduced a backtracking idea to evaluate the drought condition in unestablished wet or dry periods.
    This method has a key limitation: The drought condition at time t is dependent on the future climate conditions (i.e., t+1, t+2, ...), which is not practical for real-time drought monitoring.
    This point has been critized by Alley (Alley, 1984) and others.
    Alley also mentioned a solution, when there is no established wet or dry period, the PDSI equals to X1 or X2 with larger absolute value.
    To solve this problem, the weighted probability approach has also been proposed (e.g., Heddinghaus and Sabol, 1991) and developed and verified by others (Adams, 2017) (Rhee, 2007). 
    Currently, the weighted probability approach is widely accepted and used in many operational drought monitoring systems (e.g., CPC)
    In this implementation, we follow the weighted probability approach (Adams, 2017) to calculate PDSI.
    """
    def _palmer_state_machine_index_full(self, Z: xr.DataArray, persistence_factor: float = 0.897, initial_state: dict = None) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        VECTORIZED Palmer State Machine - Fully compatible with standalone version
        Migrated from palmer_state_machine_standalone with complete logic preservation
        
        Args:
            Z: Climate anomaly index (3D)
            persistence_factor: Persistence coefficient (default 0.897 for monthly, but for weekly data it is 0.975)
            initial_state: Optional dict with 'x1', 'x2', 'x3' initial values
        
        Returns:
            Tuple[xr.DataArray, xr.DataArray]: (PHDI, PDSI)
        """
        # Handle initial_state parameter
        if initial_state is None:
            initial_state = {}
        
        # Constants from standalone version
        dry_spell_threshold = 1.0  
        wet_spell_threshold = 1.0  
        z_negative_multiplier = 1.0
        z_positive_multiplier = 1.0
        COEFF_2691 = 2.691
        THRESHOLD_015 = 0.15  
        THRESHOLD_15 = 1.5
        Z_DIVISOR = 3.0
        
        # Input validation
        if Z is None or Z.ndim == 0 or 'time' not in Z.dims or Z.sizes.get('time', 0) == 0:
            print(f"    WARNING: Invalid Z input, returning empty arrays")
            empty_template = self._create_empty_template_from_z(Z)
            return empty_template, empty_template
        
        Z_computed = Z.compute() if hasattr(Z, 'chunks') and Z.chunks is not None else Z
        
        # Initialize spatial template
        spatial_template = Z_computed.isel(time=0, drop=True, missing_dims='ignore')
        if 'time' in spatial_template.dims:
            spatial_template = spatial_template.squeeze('time', drop=True)
        
        zeros_spatial = xr.DataArray(np.zeros_like(spatial_template.data, dtype=Z_computed.dtype),
                                    coords=spatial_template.coords, dims=spatial_template.dims)
        
        # State variables initialization (support initial_state)
        val_x1 = initial_state.get('x1', 0.0)
        val_x2 = initial_state.get('x2', 0.0)
        val_x3 = initial_state.get('x3', 0.0)
        
        x1 = zeros_spatial + val_x1
        x2 = zeros_spatial + val_x2
        x3 = zeros_spatial + val_x3
        v = zeros_spatial.copy(deep=True)
        prob = zeros_spatial.copy(deep=True)
        
        # Storage for results
        phdi_x3_list = []
        pdsi_comprehensive_list = []
        
        # ============================================================
        # MAIN TIME LOOP - Vectorized Standalone Logic
        # ============================================================
        for t_idx in range(Z_computed.sizes['time']):
            z_t = Z_computed.isel(time=t_idx).fillna(0)
            time_coord_val = Z_computed.time.data[t_idx]
            
            # Apply Z multipliers
            z_t = xr.where(z_t < 0, z_negative_multiplier * z_t, z_t * z_positive_multiplier)
            
            # ============================================================
            # STEP 1: PRE-CALCULATE CANDIDATE VALUES (Vectorized)
            # ============================================================
            cand_x1 = np.maximum(0, persistence_factor * x1 + z_t / Z_DIVISOR)
            cand_x2 = np.minimum(0, persistence_factor * x2 + z_t / Z_DIVISOR)
            cand_x3 = persistence_factor * x3 + z_t / Z_DIVISOR
            
            # ============================================================
            # STEP 2: DETERMINE CURRENT STATE (Vectorized Masks)
            # ============================================================
            no_established_period = (x3 == 0)
            wet_spell = (x3 > 0.5)
            drought_spell = (x3 < -0.5)
            
            # ============================================================
            # STEP 3: INITIALIZE OUTPUT VARIABLES
            # ============================================================
            px1 = cand_x1.copy(deep=True)
            px2 = cand_x2.copy(deep=True)
            px3 = cand_x3.copy(deep=True)
            pv = zeros_spatial.copy(deep=True)
            pprob = zeros_spatial.copy(deep=True)
            
            # ============================================================
            # BRANCH 1: NO ESTABLISHED SPELL (Near Normal)
            # ============================================================
            # X3 stays at 0
            px3 = xr.where(no_established_period, 0.0, px3)
            
            # Check if X1 crosses wet threshold
            establish_wet = no_established_period & (cand_x1 >= wet_spell_threshold)
            px3 = xr.where(establish_wet, cand_x1, px3)
            px1 = xr.where(establish_wet, 0.0, px1)
            pprob = xr.where(establish_wet, 0.0, pprob)
            
            # Check if X2 crosses dry threshold
            establish_dry = no_established_period & (cand_x2 <= -dry_spell_threshold) & ~establish_wet
            px3 = xr.where(establish_dry, cand_x2, px3)
            px2 = xr.where(establish_dry, 0.0, px2)
            pprob = xr.where(establish_dry, 0.0, pprob)
            
            # ============================================================
            # BRANCH 2: WET SPELL PROCESSING
            # ============================================================
            wet_intensifies = wet_spell & (z_t >= THRESHOLD_015) & (prob == 0)
            wet_abates = wet_spell & ((z_t < THRESHOLD_015) | (prob > 0))
            
            # True Intensification (only when prob == 0)
            px3 = xr.where(wet_intensifies, cand_x3, px3)
            px1 = xr.where(wet_intensifies, 0.0, px1)
            px2 = xr.where(wet_intensifies, cand_x2, px2)  # KEY FIX: Opposite direction evolves
            
            # Wet Abatement Calculation
            temp_v_wet_abate = z_t - THRESHOLD_015 + np.minimum(v, 0.0)
            ze_wet = -COEFF_2691 * x3 + THRESHOLD_15
            q_wet = xr.where(prob == 0, ze_wet, ze_wet + v)
            ending_prob_wet = xr.where(q_wet != 0, (temp_v_wet_abate / q_wet) * 100, 0.0)
            ending_prob_wet = ending_prob_wet.clip(0, 100)
            
            # Wet Spell Ends (Prob >= 100%)
            wet_ends_by_prob = wet_abates & (ending_prob_wet >= 100)
            px3 = xr.where(wet_ends_by_prob, 0.0, px3)
            px1 = xr.where(wet_ends_by_prob, cand_x1, px1)
            px2 = xr.where(wet_ends_by_prob, cand_x2, px2)
            pv = xr.where(wet_ends_by_prob, 0.0, pv)
            pprob = xr.where(wet_ends_by_prob, ending_prob_wet, pprob)
            
            # Check immediate reversal to drought after wet spell ends
            immediate_dry_after_wet = wet_ends_by_prob & (cand_x2 <= -dry_spell_threshold)
            px3 = xr.where(immediate_dry_after_wet, cand_x2, px3)
            px2 = xr.where(immediate_dry_after_wet, 0.0, px2)
            
            # Wet Spell Ends (V* >= 0, Resurrection)
            wet_resurrects = wet_abates & (temp_v_wet_abate >= 0) & (ending_prob_wet < 100)
            px3 = xr.where(wet_resurrects, cand_x3, px3)
            px1 = xr.where(wet_resurrects, 0.0, px1)
            px2 = xr.where(wet_resurrects, cand_x2, px2)
            pv = xr.where(wet_resurrects, 0.0, pv)
            pprob = xr.where(wet_resurrects, 0.0, pprob)
            
            # Wet Spell Continues (Abatement continues)
            wet_continues = wet_abates & ~wet_ends_by_prob & ~wet_resurrects
            px3 = xr.where(wet_continues, cand_x3, px3)
            px1 = xr.where(wet_continues, cand_x1, px1)  # KEY FIX: Same direction evolves
            px2 = xr.where(wet_continues, cand_x2, px2)  # Opposite direction evolves
            pv = xr.where(wet_continues, temp_v_wet_abate, pv)
            pprob = xr.where(wet_continues, ending_prob_wet.clip(0, 100), pprob)
            
            # ============================================================
            # BRANCH 3: DROUGHT SPELL PROCESSING
            # ============================================================
            drought_intensifies = drought_spell & (z_t <= -THRESHOLD_015) & (prob == 0)
            drought_abates = drought_spell & ((z_t > -THRESHOLD_015) | (prob > 0))
            
            # True Intensification (only when prob == 0)
            px3 = xr.where(drought_intensifies, cand_x3, px3)
            px1 = xr.where(drought_intensifies, cand_x1, px1)  # KEY FIX: Opposite direction evolves
            px2 = xr.where(drought_intensifies, 0.0, px2)
            
            # Drought Abatement Calculation
            temp_v_drought_abate = z_t + THRESHOLD_015 + np.maximum(v, 0.0)
            ze_drought = -COEFF_2691 * x3 - THRESHOLD_15
            q_drought = xr.where(prob == 0, ze_drought, ze_drought + v)
            ending_prob_drought = xr.where(q_drought != 0, (temp_v_drought_abate / q_drought) * 100, 0.0)
            ending_prob_drought = ending_prob_drought.clip(0, 100)
            
            # Drought Spell Ends (Prob >= 100%)
            drought_ends_by_prob = drought_abates & (ending_prob_drought >= 100)
            px3 = xr.where(drought_ends_by_prob, 0.0, px3)
            px1 = xr.where(drought_ends_by_prob, cand_x1, px1)
            px2 = xr.where(drought_ends_by_prob, cand_x2, px2)
            pv = xr.where(drought_ends_by_prob, 0.0, pv)
            pprob = xr.where(drought_ends_by_prob, ending_prob_drought, pprob)
            
            # Check immediate reversal to wet after drought spell ends
            immediate_wet_after_dry = drought_ends_by_prob & (cand_x1 >= wet_spell_threshold)
            px3 = xr.where(immediate_wet_after_dry, cand_x1, px3)
            px1 = xr.where(immediate_wet_after_dry, 0.0, px1)
            
            # Drought Spell Ends (V* <= 0, Resurrection)
            drought_resurrects = drought_abates & (temp_v_drought_abate <= 0) & (ending_prob_drought < 100)
            px3 = xr.where(drought_resurrects, cand_x3, px3)
            px1 = xr.where(drought_resurrects, cand_x1, px1)
            px2 = xr.where(drought_resurrects, 0.0, px2)
            pv = xr.where(drought_resurrects, 0.0, pv)
            pprob = xr.where(drought_resurrects, 0.0, pprob)
            
            # Drought Spell Continues (Abatement continues)
            drought_continues = drought_abates & ~drought_ends_by_prob & ~drought_resurrects
            px3 = xr.where(drought_continues, cand_x3, px3)
            px1 = xr.where(drought_continues, cand_x1, px1)  # Opposite direction evolves
            px2 = xr.where(drought_continues, cand_x2, px2)  # KEY FIX: Same direction evolves
            pv = xr.where(drought_continues, temp_v_drought_abate, pv)
            pprob = xr.where(drought_continues, ending_prob_drought.clip(0, 100), pprob)
            
            # ============================================================
            # PDSI CALCULATION (Weighted Average Logic)
            # ============================================================
            pdsi_value = zeros_spatial.copy(deep=True)
            established_periods = (np.abs(px3) > 0.5)
            final_prob = pprob / 100.0
            
            # Use X3 directly when no probability weighting
            use_x3_directly = established_periods & ((pprob <= 0) | (pprob >= 100))
            pdsi_value = xr.where(use_x3_directly, px3, pdsi_value)
            
            # Probability weighting for established periods
            has_probability_weight = established_periods & (pprob > 0) & (pprob < 100)
            
            drought_with_prob = has_probability_weight & (px3 <= 0)
            pdsi_value = xr.where(drought_with_prob, 
                                (1.0 - final_prob) * px3 + final_prob * px1, 
                                pdsi_value)
            
            wet_with_prob = has_probability_weight & (px3 > 0)
            pdsi_value = xr.where(wet_with_prob,
                                (1.0 - final_prob) * px3 + final_prob * px2,
                                pdsi_value)
            
            # Near normal conditions
            in_near_normal = (np.abs(px3) <= 0.5)
            both_exist = in_near_normal & (px1 != 0) & (px2 != 0)
            choose_x1 = both_exist & (np.abs(px1) > np.abs(px2))
            choose_x2 = both_exist & (np.abs(px1) <= np.abs(px2))
            
            pdsi_value = xr.where(choose_x1, px1, pdsi_value)
            pdsi_value = xr.where(choose_x2, px2, pdsi_value)
            
            only_x1 = in_near_normal & (px1 != 0) & (px2 == 0)
            only_x2 = in_near_normal & (px1 == 0) & (px2 != 0)
            
            pdsi_value = xr.where(only_x1, px1, pdsi_value)
            pdsi_value = xr.where(only_x2, px2, pdsi_value)
            
            # ============================================================
            # DEBUG OUTPUT (Use mean values for brevity)
            # ============================================================
            if t_idx < 15 or t_idx >= (Z_computed.sizes['time'] - 15):
                z_mean = z_t.mean().item()
                x1_mean = px1.mean().item()
                x2_mean = px2.mean().item()
                x3_mean = px3.mean().item()
                pdsi_mean = pdsi_value.mean().item()
                prob_mean = pprob.mean().item()
                v_mean = pv.mean().item()
                
                trigger_wet_count = (px1 >= 1.0).sum().item() if (px1 >= 1.0).any() else 0
                trigger_drought_count = (px2 <= -1.0).sum().item() if (px2 <= -1.0).any() else 0
                prob_events = (pprob > 0).sum().item() if (pprob > 0).any() else 0
                
                print(f"DEBUG Palmer t={t_idx}: Z={z_mean:.4f}, "
                    f"x1={x1_mean:.4f}, x2={x2_mean:.4f}, x3={x3_mean:.4f}, "
                    f"pdsi={pdsi_mean:.4f}, prob={prob_mean:.2f}%, v={v_mean:.4f}, "
                    f"wet_triggers={trigger_wet_count}, drought_triggers={trigger_drought_count}, "
                    f"prob_events={prob_events}")
                
                if trigger_wet_count > 0 or trigger_drought_count > 0 or prob_events > 0:
                    print(f"EVENT at t={t_idx}: triggers={trigger_wet_count+trigger_drought_count}, prob_events={prob_events}")
            
            # Store results
            phdi_result = px3.copy(deep=True)
            pdsi_result = pdsi_value.copy(deep=True)
            
            if 'time' in Z_computed.coords:
                phdi_result = phdi_result.assign_coords({'time': time_coord_val}).expand_dims('time')
                pdsi_result = pdsi_result.assign_coords({'time': time_coord_val}).expand_dims('time')
            
            phdi_x3_list.append(phdi_result)
            pdsi_comprehensive_list.append(pdsi_result)
            
            # Update state variables for next iteration
            x1 = px1.copy(deep=True)
            x2 = px2.copy(deep=True)
            x3 = px3.copy(deep=True)
            v = pv.copy(deep=True)
            prob = pprob.copy(deep=True)
        
        # Concatenate all time steps
        try:
            phdi_final = xr.concat(phdi_x3_list, dim='time')
            pdsi_final = xr.concat(pdsi_comprehensive_list, dim='time')
            
            phdi_final.name = 'phdi'
            phdi_final.attrs = {
                'long_name': 'Palmer Hydrological Drought Index',
                'description': 'Indicator of established drought/wet periods (x3 component)',
                'units': 'dimensionless',
            }
            
            pdsi_final.name = 'pdsi'
            pdsi_final.attrs = {
                'long_name': 'Palmer Drought Severity Index',
                'description': 'Comprehensive PDSI with probability weighting',
                'units': 'dimensionless',
            }
            
            return phdi_final, pdsi_final
            
        except Exception as e:
            print(f"ERROR in Palmer state machine concatenation: {e}")
            empty_template = self._create_empty_template_from_z(Z)
            return empty_template, empty_template


    def _create_empty_template_from_z(self, Z: xr.DataArray) -> xr.DataArray:
        if Z is not None and Z.ndim > 0 and 'time' in Z.dims:
            spatial_template = Z.isel(time=0, drop=True, missing_dims='ignore')
            coords_dict = {'time': Z.coords['time'].data[:0]}
            target_dims_order = ['time']
            final_shape = [0]
            for dim_name in Z.dims:
                if dim_name != 'time' and dim_name in spatial_template.dims and spatial_template.sizes[dim_name] > 0:
                    target_dims_order.append(dim_name)
                    final_shape.append(spatial_template.sizes[dim_name])
                    coords_dict[dim_name] = spatial_template.coords[dim_name].data
            return xr.DataArray(np.empty(tuple(final_shape), dtype=Z.dtype), coords=coords_dict, dims=target_dims_order)
        else:
            raise ValueError("Cannot create empty template from invalid Z")

    def _save_cube_dataset(self, abs_cube_num: int, dataset_to_save: xr.Dataset, out_zarr_root_seg: str, log_tag: str):
        """Save cube dataset with proper chunking preserved"""
        cube_out_path = os.path.join(out_zarr_root_seg, str(abs_cube_num))
        
        try:
            if not os.path.exists(out_zarr_root_seg): 
                os.makedirs(out_zarr_root_seg, exist_ok=True)
            
            # Critical fix: Preserve chunking during compute and save
            computed_dataset_to_save = {}
            
            for var_name, data_array in dataset_to_save.data_vars.items():
                if hasattr(data_array.data, 'dask'):
                    # Data is dask-backed, need to compute but preserve chunking intent
                    print(f"    DEBUG: Computing dask array for {var_name}, original chunks: {data_array.chunks}")
                    computed_data = data_array.compute()
                    computed_dataset_to_save[var_name] = computed_data
                else:
                    # Data is already computed
                    computed_dataset_to_save[var_name] = data_array
            
            # Create dataset with computed data
            final_dataset = xr.Dataset(computed_dataset_to_save)
            
            # Apply chunking BEFORE saving to zarr
            chunked_vars = {}
            for var_name, data_array in final_dataset.data_vars.items():
                if 'time' in data_array.dims:
                    # For time-series data, apply the chunking we want
                    chunk_dict = {}
                    if 'time' in data_array.dims:
                        chunk_dict['time'] = min(256, data_array.sizes['time'])  # Force time chunks of 256
                    for dim in data_array.dims:
                        if dim != 'time':
                            chunk_dict[dim] = data_array.sizes[dim]  # Full size for spatial dims
                    
                    print(f"    DEBUG: Applying final chunks to {var_name}: {chunk_dict}")
                    chunked_vars[var_name] = data_array.chunk(chunk_dict)
                else:
                    chunked_vars[var_name] = data_array
            
            final_chunked_dataset = xr.Dataset(chunked_vars)
            
            # Save with encoding to preserve chunks
            encoding = {}
            for var_name in final_chunked_dataset.data_vars:
                if var_name in chunked_vars:
                    # Set zarr chunks to match dask chunks
                    chunks_list = []
                    for dim in final_chunked_dataset[var_name].dims:
                        chunk_size = final_chunked_dataset[var_name].chunks[final_chunked_dataset[var_name].get_axis_num(dim)][0]
                        chunks_list.append(chunk_size)
                    encoding[var_name] = {'chunks': chunks_list}
                    print(f"    DEBUG: Zarr encoding for {var_name}: {encoding[var_name]}")
            
            # Save to zarr with explicit encoding
            final_chunked_dataset.to_zarr(cube_out_path, mode='a', consolidated=False, encoding=encoding)
            
            print(f"   {log_tag} (vars: {list(dataset_to_save.data_vars.keys())}) for cube_abs {abs_cube_num} saved to: {cube_out_path}")
            
        except Exception as e:
            print(f"   Error saving {log_tag} to {cube_out_path}: {e}")
            print(f"   Error details: {traceback.format_exc()}")
            if os.path.exists(cube_out_path) and os.path.isdir(cube_out_path):
                try: 
                    import shutil
                    shutil.rmtree(cube_out_path)
                except Exception as e_rm: 
                    print(f"   Error during cleanup of {cube_out_path}: {e_rm}")
                
    def _calculate_z_score(self, data_arr: xr.DataArray, mean_arr: xr.DataArray, std_arr_orig: xr.DataArray) -> xr.DataArray:
        std_for_division = std_arr_orig.clip(min=1e-9) 
        z = xr.where(std_arr_orig.isnull(), np.nan,
                     xr.where(std_arr_orig == 0, 0.0, 
                              (data_arr - mean_arr) / std_for_division))
        return z

    def _calculate_cmi_from_aggregated_inputs(self,
                                            aggregated_cube_ds: xr.Dataset,
                                            awc_total_static_2d: xr.DataArray,
                                            s_name_for_log: str, 
                                            c_idx_abs: int
                                            ) -> Optional[xr.DataArray]:
        """
        Calculates the simplified Crop Moisture Index (CMI) from aggregated inputs.
        Calculating CMI is now abolished because of poor performance and lack of theoretical justification, but kept here for reference.
        This is a "memoryless" index based on the formula:
        CMI = 100 * (1 - MD/AWC), where MD = max(0, PET - Precipitation).
        """
        P_monthly = self._handle_nans(aggregated_cube_ds.get('prcp'), 'prcp_agg_cmi', s_name_for_log, c_idx_abs)
        PET_monthly = self._handle_nans(aggregated_cube_ds.get('pet'), 'pet_agg_cmi', s_name_for_log, c_idx_abs)

        if not all(v is not None for v in [P_monthly, PET_monthly, awc_total_static_2d]):
            print(f"  CMI Error: Missing base inputs (P, PET, or AWC) for cube {c_idx_abs} in {s_name_for_log}")
            return None

        try:
            # Debug AWC values first for context
            awc_stats = (f"AWC stats: mean={awc_total_static_2d.mean(skipna=True).item():.4f}, "
                        f"min={awc_total_static_2d.min(skipna=True).item():.4f}, "
                        f"max={awc_total_static_2d.max(skipna=True).item():.4f}, "
                        f"std={awc_total_static_2d.std(skipna=True).item():.4f}")
            print(f"DEBUG CMI Cube {c_idx_abs}: {awc_stats}")
            
            # Calculate the time-varying moisture deficit (MD)
            moisture_deficit_timeseries = (PET_monthly - P_monthly).clip(min=0)
            
            # Key refinement: Set a reasonable minimum threshold for AWC to prevent division by zero or
            # near-zero values, which can cause numerical instability.
            awc_min_threshold = 5.0  # Assumes areas with AWC < 5mm are not suitable for this analysis.
            awc_for_division = awc_total_static_2d.clip(min=awc_min_threshold)
            
            # Log how many pixels were affected by this clipping for transparency.
            low_awc_count = (awc_total_static_2d < awc_min_threshold).sum().item()
            if low_awc_count > 0:
                print(f"DEBUG CMI Cube {c_idx_abs}: {low_awc_count} pixels had AWC < {awc_min_threshold}mm and were clipped.")
            
            # Calculate the CMI using the clipped AWC value.
            # md_ratio_timeseries = np.minimum(1.0, 4.5 * moisture_deficit_timeseries / awc_for_division)
            md_ratio_timeseries = moisture_deficit_timeseries / awc_for_division

            cmi_timeseries = 100.0 * (1.0 - md_ratio_timeseries)  #
            
            # Clip the CMI values to a plausible range to handle potential outliers from extreme MD/AWC ratios.
            cmi_timeseries = cmi_timeseries.clip(-100, 100)
            
            # Handle any resulting NaNs from the calculations.
            cmi_timeseries = self._handle_nans(cmi_timeseries, 'cmi_timeseries', s_name_for_log, c_idx_abs)
            if cmi_timeseries is None:
                return None
            
            if P_monthly is not None:
                # ensure CMI has the same dimension order and coordinates as P_monthly
                if set(cmi_timeseries.dims) == set(P_monthly.dims):
                    # reorder dimensions to match P_monthly's dimension order
                    cmi_timeseries = cmi_timeseries.transpose(*P_monthly.dims)

                    # ensure coordinates match exactly
                    for coord_name in P_monthly.coords:
                        if coord_name in cmi_timeseries.coords:
                            cmi_timeseries.coords[coord_name] = P_monthly.coords[coord_name]
            
            # Set final variable name and metadata attributes.
            cmi_timeseries.name = 'cmi'
            cmi_timeseries.attrs = {
                'units': 'unitless',
                'long_name': 'Crop Moisture Index (Simplified)',
                'description': f'CMI = 100 * (1 - MD/AWC) where MD = max(0, PET - P), AWC_min_threshold = {awc_min_threshold}mm'
            }
            
            # Provide an enhanced debug output for diagnostics.
            if cmi_timeseries.size > 0:
                try:
                    # Check for extreme values in the final CMI product.
                    very_low = (cmi_timeseries < -50).sum().item()
                    very_high = (cmi_timeseries > 90).sum().item()
                    
                    print(f"DEBUG CMI Cube {c_idx_abs}: CMI stats: "
                        f"mean={cmi_timeseries.mean(skipna=True).item():.2f}, "
                        f"min={cmi_timeseries.min(skipna=True).item():.2f}, "
                        f"max={cmi_timeseries.max(skipna=True).item():.2f}, "
                        f"std={cmi_timeseries.std(skipna=True).item():.2f}, "
                        f"dims={cmi_timeseries.dims}, shape={cmi_timeseries.shape}, "
                        f"very_low(<-50)={very_low}, very_high(>90)={very_high}")
                    
                    # Sample some values to help understand the calculation logic.
                    sample_md = moisture_deficit_timeseries.isel(time=0).mean(skipna=True).item()
                    sample_awc = awc_for_division.mean(skipna=True).item()
                    if sample_awc > 0: # Avoid division by zero in logging
                        print(f"DEBUG CMI Cube {c_idx_abs}: Sample values for t=0: MD={sample_md:.2f}mm, AWC_mean={sample_awc:.2f}mm, ratio={sample_md/sample_awc:.3f}")
                    else:
                        print(f"DEBUG CMI Cube {c_idx_abs}: Sample values for t=0: MD={sample_md:.2f}mm, AWC_mean={sample_awc:.2f}mm")

                except Exception as e_stat:
                    print(f"DEBUG CMI Cube {c_idx_abs}: Error printing enhanced CMI stats: {e_stat}")
            
            return cmi_timeseries
            
        except Exception as e_cmi_calc:
            print(f"  Error during CMI calculation for cube_abs {c_idx_abs}, seg {s_name_for_log}: {e_cmi_calc}")
            return None

    # +++++++++++++++ The new added two-layer soil moisture accounting model +++++++++++++++
    def _perform_soil_moisture_accounting_two_layer(self,
                                                    P_t_series: xr.DataArray,
                                                    PET_t_series: xr.DataArray,
                                                    AWC_s_static: xr.DataArray,
                                                    AWC_u_static: xr.DataArray,
                                                    s_name_for_log: str, c_idx_abs: int
                                                    ) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray],
                                                               Optional[xr.DataArray], Optional[xr.DataArray],
                                                               Optional[xr.DataArray], Optional[xr.DataArray]]:
        if not all(isinstance(da, xr.DataArray) and da.size > 0 for da in [P_t_series, PET_t_series, AWC_s_static, AWC_u_static]):
            missing_vars = [name for name, da in zip(['P', 'PET', 'AWC_s', 'AWC_u'], [P_t_series, PET_t_series, AWC_s_static, AWC_u_static]) if not (isinstance(da, xr.DataArray) and da.size > 0)]
            print(f"   SMA-2L Error: Missing or empty inputs ({', '.join(missing_vars)}) for cube_abs {c_idx_abs}, seg {s_name_for_log}.")
            return None, None, None, None, None, None, None, None

        try:
            P_loaded = P_t_series.compute() if hasattr(P_t_series, 'chunks') and P_t_series.chunks is not None else P_t_series
            PET_loaded = PET_t_series.compute() if hasattr(PET_t_series, 'chunks') and PET_t_series.chunks is not None else PET_t_series
            AWC_s = AWC_s_static.compute() if hasattr(AWC_s_static, 'chunks') and AWC_s_static.chunks is not None else AWC_s_static
            AWC_u = AWC_u_static.compute() if hasattr(AWC_u_static, 'chunks') and AWC_u_static.chunks is not None else AWC_u_static

            P_loaded = P_loaded.clip(min=0)
            PET_loaded = PET_loaded.clip(min=0)
            AWC_s = AWC_s.clip(min=0)
            AWC_u = AWC_u.clip(min=0)

            num_time_steps = P_loaded.sizes.get('time', 0)
            if num_time_steps == 0:
                print(f"   SMA-2L Error: P_t_series has no time steps for cube {c_idx_abs}.")
                return None, None, None, None, None, None, None, None

            # --- Start of Corrected Logic ---
            # In the classical algorithm, 'awc' refers to the total available water capacity.
            AWC_total = AWC_s + AWC_u
            # Clip the total AWC to a small positive number to prevent division by zero.
            # It has been checked that the input AWC have no "zero" values or negative values. Here the clipping is just a safeguard.
            AWC_total_clipped = AWC_total.clip(min=1e-6)
            # --- End of Corrected Logic ---

            s_s_prev = AWC_s.copy(deep=True)
            s_u_prev = AWC_u.copy(deep=True)

            # Because initial soil moisture is set to field capacity, we need to mask out invalid pixels at the first time step which have NaN precipitation / PET.
            mask_valid_prcp = P_loaded.isel(time=0).notnull()
            mask_valid_pet = PET_loaded.isel(time=0).notnull()
            if mask_valid_pet.any() and mask_valid_prcp.any():
                print(f"   SMA-2L Info: Valid pixel counts at initial time step for cube {c_idx_abs} - Prcp: {mask_valid_prcp.sum().item()}, PET: {mask_valid_pet.sum().item()}")
            if mask_valid_prcp.equals(mask_valid_pet):
                print(f"   SMA-2L Info: Prcp and PET masks are identical for cube {c_idx_abs}. Applying initial masking.")
                mask_valid = mask_valid_prcp
                s_s_prev = s_s_prev.where(mask_valid)
                s_u_prev = s_u_prev.where(mask_valid)
            else:
                s_s_prev = s_s_prev
                s_u_prev = s_u_prev
                print(f"Warning: Prcp and PET masks are NOT identical for cube {c_idx_abs}. Skipping initial masking.")

            et_actual_list, r_total_list, l_total_list, ro_list, s_s_ts_list, s_u_ts_list = [], [], [], [], [], []
            pl_list, pr_list = [], []

            spatial_template = P_loaded.isel(time=0, drop=True, missing_dims='ignore')
            spatial_coords = spatial_template.coords
            spatial_dims = list(spatial_template.dims)

            zeros_spatial = xr.DataArray(np.zeros_like(spatial_template.data, dtype=P_loaded.dtype),
                                         coords=spatial_coords, dims=spatial_dims)

            for t_idx in range(num_time_steps):
                P_t = P_loaded.isel(time=t_idx, drop=True)
                PET_t = PET_loaded.isel(time=t_idx, drop=True)
                PR_t = (AWC_total_clipped - (s_s_prev + s_u_prev)).clip(min=0)
                PL_t = xr.where(s_s_prev >= PET_t,
                            PET_t,
                            np.minimum(s_s_prev + s_u_prev, 
                                      ((PET_t - s_s_prev) * s_u_prev) / AWC_total_clipped + s_s_prev))

                Ls_t = zeros_spatial.copy(deep=True)
                Lu_t = zeros_spatial.copy(deep=True)
                Rs_t = zeros_spatial.copy(deep=True)
                Ru_t = zeros_spatial.copy(deep=True)
                RO_t = zeros_spatial.copy(deep=True)

                s_s_current = s_s_prev.copy(deep=True)
                s_u_current = s_u_prev.copy(deep=True)

                ET_from_P_t = np.minimum(P_t, PET_t)
                remaining_pet_demand = PET_t - ET_from_P_t

                # Only proceed with soil loss if there is remaining PET demand
                if (remaining_pet_demand > 0).any():
                    # First, draw from the surface layer (sl)
                    can_lose_from_s = np.minimum(s_s_current, remaining_pet_demand)
                    Ls_t = xr.where(remaining_pet_demand > 0, can_lose_from_s, 0)
                    s_s_current = s_s_current - Ls_t
                    
                    # --- Start of Corrected Logic for Deep Layer Loss ---
                    # This corresponds to (pet - p - sl) in the classical formula
                    demand_after_surface_loss = remaining_pet_demand - Ls_t
                    demand_after_surface_loss = demand_after_surface_loss.clip(min=0)

                    if (demand_after_surface_loss > 0).any():
                        # This is the core of the classical formula: ul = (pet - p - sl) * su / awc
                        # Loss is proportional to the remaining demand and the ratio of deep soil moisture (su) to total AWC.
                        proportional_loss_from_u = demand_after_surface_loss * s_u_current / AWC_total_clipped
                        
                        # The actual loss cannot exceed the moisture available in the deep layer (su)
                        can_lose_from_u = np.minimum(s_u_current, proportional_loss_from_u)
                        
                        # Assign the calculated loss only where there is still demand
                        Lu_t = xr.where(demand_after_surface_loss > 0, can_lose_from_u, 0)
                        s_u_current = s_u_current - Lu_t
                    # --- End of Corrected Logic for Deep Layer Loss ---

                L_total_t = Ls_t + Lu_t
                ET_actual_t = ET_from_P_t + L_total_t
                ET_actual_t = np.minimum(ET_actual_t, PET_t).clip(min=0)

                water_available_for_recharge_runoff = (P_t - ET_from_P_t).clip(min=0)

                if (water_available_for_recharge_runoff > 0).any():
                    can_recharge_s = (AWC_s - s_s_current).clip(min=0)
                    Rs_t = np.minimum(water_available_for_recharge_runoff, can_recharge_s)
                    s_s_current = s_s_current + Rs_t
                    water_available_for_recharge_runoff = water_available_for_recharge_runoff - Rs_t
                    water_available_for_recharge_runoff = water_available_for_recharge_runoff.clip(min=0)

                    if (water_available_for_recharge_runoff > 0).any():
                        can_recharge_u = (AWC_u - s_u_current).clip(min=0)
                        Ru_t = np.minimum(water_available_for_recharge_runoff, can_recharge_u)
                        s_u_current = s_u_current + Ru_t
                        water_available_for_recharge_runoff = water_available_for_recharge_runoff - Ru_t
                        water_available_for_recharge_runoff = water_available_for_recharge_runoff.clip(min=0)
                    
                    if (water_available_for_recharge_runoff > 0).any():
                        # RO_t = water_available_for_recharge_runoff
                        RO_t = water_available_for_recharge_runoff.clip(min=0)
                # Recharge and runoff logic to deal with the initial value of soil moisture for "whole time nan" pixels have different results.
                # If at the first time step the water_available_for_recharge_runoff < 0, the recharge will not be considered.
                # The soil moisture will be at the initial value.
                # If at the first time step the water_available_for_recharge_runoff > 0, the recharge will be considered.
                # The soil moisture of "whole time nan" pixels will be NaN at the first time step.
                # Hence, the final RSM results for "whole time nan" pixels will be different.

                R_total_t = Rs_t + Ru_t

                s_s_current = s_s_current.clip(min=0, max=AWC_s)
                s_u_current = s_u_current.clip(min=0, max=AWC_u)

                time_coord_val = P_loaded.time.data[t_idx]
                et_actual_list.append(ET_actual_t.expand_dims(time=[time_coord_val]))
                r_total_list.append(R_total_t.expand_dims(time=[time_coord_val]))
                l_total_list.append(L_total_t.expand_dims(time=[time_coord_val]))
                ro_list.append(RO_t.expand_dims(time=[time_coord_val]))
                s_s_ts_list.append(s_s_current.expand_dims(time=[time_coord_val]))
                s_u_ts_list.append(s_u_current.expand_dims(time=[time_coord_val]))

                pl_list.append(PL_t.expand_dims(time=[time_coord_val]))
                pr_list.append(PR_t.expand_dims(time=[time_coord_val]))

                s_s_prev = s_s_current
                s_u_prev = s_u_current
            
            concat_dim_coord = P_loaded.coords['time']
            ET_actual_series = xr.concat(et_actual_list, dim=concat_dim_coord) if et_actual_list else self._create_empty_timeseries_like(P_loaded, num_time_steps, spatial_coords, spatial_dims)
            R_total_series = xr.concat(r_total_list, dim=concat_dim_coord) if r_total_list else self._create_empty_timeseries_like(P_loaded, num_time_steps, spatial_coords, spatial_dims)
            L_total_series = xr.concat(l_total_list, dim=concat_dim_coord) if l_total_list else self._create_empty_timeseries_like(P_loaded, num_time_steps, spatial_coords, spatial_dims)
            RO_total_series = xr.concat(ro_list, dim=concat_dim_coord) if ro_list else self._create_empty_timeseries_like(P_loaded, num_time_steps, spatial_coords, spatial_dims)
            S_s_series = xr.concat(s_s_ts_list, dim=concat_dim_coord) if s_s_ts_list else self._create_empty_timeseries_like(P_loaded, num_time_steps, spatial_coords, spatial_dims)
            S_u_series = xr.concat(s_u_ts_list, dim=concat_dim_coord) if s_u_ts_list else self._create_empty_timeseries_like(P_loaded, num_time_steps, spatial_coords, spatial_dims)
            PL_total_series = xr.concat(pl_list, dim=concat_dim_coord) if pl_list else self._create_empty_timeseries_like(P_loaded, num_time_steps, spatial_coords, spatial_dims)
            PR_total_series = xr.concat(pr_list, dim=concat_dim_coord) if pr_list else self._create_empty_timeseries_like(P_loaded, num_time_steps, spatial_coords, spatial_dims)

            return ET_actual_series, R_total_series, L_total_series, RO_total_series, S_s_series, S_u_series, PL_total_series, PR_total_series

        except Exception as e_sma_2l:
            print(f"   SMA-2L Error during core loop for cube_abs {c_idx_abs}, seg {s_name_for_log}: {e_sma_2l}")
            traceback.print_exc()
            return None, None, None, None, None, None, None, None

    def _create_moving_window_data(self, weekly_da: xr.DataArray, window_size_weeks: int = 4, window_pattern: str = "") -> xr.DataArray:
        """
        Create moving window sum data from weekly time series data.
        This converts weekly data into pseudo-monthly data using sliding window approach.
        
        Args:
            weekly_da: Input weekly DataArray with time dimension
            window_size_weeks: Size of sliding window in weeks (default: 4)
            window_pattern: Description or pattern of the window (default: "")
            
        Returns:
            DataArray with moving window sums, maintaining weekly temporal resolution
        """
        print(f"    DEBUG Moving Window: Creating {window_size_weeks}-week moving window data with pattern '{window_pattern}'...")
        print(f"    DEBUG Moving Window: Input data shape: {weekly_da.shape}, "
              f"time range: {weekly_da.time.dt.strftime('%Y-%m-%d').data[0]} to "
              f"{weekly_da.time.dt.strftime('%Y-%m-%d').data[-1]}")
        
        # Apply rolling window operation
        if window_pattern == "centered":
            rolling_window = weekly_da.rolling(time=window_size_weeks, center=True)
            windowed_data = rolling_window.sum()
        else:
            rolling_window = weekly_da.rolling(time=window_size_weeks, center=False, min_periods=1)
            windowed_data = rolling_window.mean() * window_size_weeks
        
        print(f"    DEBUG Moving Window: Created windowed data with shape: {windowed_data.shape}")
        print(f"    DEBUG Moving Window: Sample values (first 5 time steps): "
              f"{[f'{x:.2f}' for x in windowed_data.isel(x=0, y=0).values[:5]]}")
        
        # Set attributes
        windowed_data.attrs = weekly_da.attrs.copy()
        windowed_data.attrs['long_name'] = f"{window_size_weeks}-week moving window sum of " + windowed_data.attrs.get('long_name', 'data')
        windowed_data.attrs['description'] = f"Moving window sum with {window_size_weeks} week window"
        
        return windowed_data

    def _interpolate_monthly_coefficients(self, target_times: pd.DatetimeIndex, monthly_coeffs: np.ndarray) -> xr.DataArray:
        """
        Interpolate monthly coefficients to weekly time points using linear interpolation.
        
        Args:
            target_times: DatetimeIndex with target time points (usually weekly)
            monthly_coeffs: Array of 12 monthly coefficient values
            
        Returns:
            DataArray with interpolated coefficients aligned to target_times
        """
        print(f"    DEBUG Coefficient Interpolation: Interpolating 12 monthly coefficients to {len(target_times)} time points")
        print(f"    DEBUG Coefficient Interpolation: Monthly coeffs: {[f'{x:.3f}' for x in monthly_coeffs[:6]]}... (first 6)")
        
        # Create day-of-year centers for 12 months (approximately middle of each month)
        month_centers_doy = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
        
        # Extend arrays to handle year-boundary interpolation
        # Add previous December at the beginning and next January at the end
        extended_coeffs = np.concatenate([monthly_coeffs[-1:], monthly_coeffs, monthly_coeffs[:1]])
        extended_doy = np.concatenate([month_centers_doy[-1:] - 365, month_centers_doy, month_centers_doy[:1] + 365])
        
        # Get day-of-year for target times
        target_doy = target_times.dayofyear
        
        # Perform linear interpolation
        interpolated_coeffs = np.interp(target_doy, extended_doy, extended_coeffs)
        
        print(f"    DEBUG Coefficient Interpolation: Sample interpolated values (first 5): "
              f"{[f'{x:.3f}' for x in interpolated_coeffs[:5]]}")
        print(f"    DEBUG Coefficient Interpolation: Target DOY range: {target_doy.min()}-{target_doy.max()}")
        
        # Create DataArray with proper coordinates
        interp_da = xr.DataArray(
            interpolated_coeffs,
            coords={'time': target_times},
            dims=['time'],
            name='interpolated_coefficients'
        )
        
        interp_da.attrs = {
            'units': 'unitless',
            'long_name': 'Linearly interpolated monthly coefficients',
            'description': 'Monthly coefficients interpolated to weekly time points using linear interpolation'
        }
        
        return interp_da

    def _create_empty_timeseries_like(self, 
                                      template_da_with_time: xr.DataArray, 
                                      num_time_steps: int, 
                                      spatial_coords_dict: Optional[Dict[str, any]] = None, # Made type hint more specific
                                      spatial_dims_list: Optional[List[str]] = None    # Made type hint more specific
                                      ) -> xr.DataArray:
        if spatial_coords_dict is None or spatial_dims_list is None:
            template_spatial = template_da_with_time.isel(time=0, drop=True, missing_dims='ignore')
            spatial_coords_dict = dict(template_spatial.coords) # Convert Coords object to dict
            spatial_dims_list = list(template_spatial.dims) 

        target_dtype = template_da_with_time.dtype
        
        coords = {**spatial_coords_dict} # Make a copy
        if num_time_steps == 0 or not template_da_with_time.coords['time'].size:
            coords['time'] = np.array([], dtype=template_da_with_time.coords['time'].dtype)
            time_dim_size = 0
        elif template_da_with_time.sizes['time'] >= num_time_steps:
             coords['time'] = template_da_with_time.coords['time'].data[:num_time_steps]
             time_dim_size = num_time_steps
        else: 
            try:
                start_time = template_da_with_time.coords['time'].data[0] if template_da_with_time.sizes['time'] > 0 else pd.Timestamp('1900-01-01')
                # Try to infer frequency, default to 'MS' (Month Start) if not inferable
                freq = pd.infer_freq(template_da_with_time.coords['time'].data)
                if freq is None: # If still None, try common monthly frequencies
                    if template_da_with_time.time.dt.day.nunique() == 1 and template_da_with_time.time.dt.day[0] == 1 :
                        freq = 'MS' 
                    else: # Fallback if frequency is truly unknown or irregular
                        freq = 'D' # Daily, then select first of month, or handle differently
                        temp_index = pd.date_range(start=start_time, periods=num_time_steps * 31, freq=freq) # Overestimate
                        coords['time'] = temp_index[temp_index.is_month_start][:num_time_steps].values
                        if len(coords['time']) < num_time_steps: # Fallback if still not enough
                            coords['time'] = pd.date_range(start=start_time, periods=num_time_steps, freq='MS').values

                if 'time_index' not in locals() or len(coords['time']) < num_time_steps : # if pd.infer_freq path was not taken or failed
                     time_index = pd.date_range(start=start_time, periods=num_time_steps, freq=freq or 'MS') # use inferred or default MS
                     coords['time'] = time_index.values

            except Exception: # Fallback for any error in time generation
                 coords['time'] = np.arange(num_time_steps, dtype='datetime64[ns]') # Last resort: simple sequence
            time_dim_size = len(coords['time']) # num_time_steps

        dims = ['time'] + spatial_dims_list
        # Ensure shape matches the actual length of the time coordinate generated
        shape = [time_dim_size] + [template_da_with_time.sizes[d] for d in spatial_dims_list] 
        
        return xr.DataArray(np.full(tuple(shape), np.nan, dtype=target_dtype), coords=coords, dims=dims)
        
    def _calculate_phdi_and_pdsi_from_aggregated_inputs(self,
                                                    aggregated_cube_ds: xr.Dataset,
                                                    awc_s_static_2d: xr.DataArray, 
                                                    awc_u_static_2d: xr.DataArray, 
                                                    s_name_for_log: str, c_idx_abs: int,
                                                    method: str = 'cafec'
                                                    ) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray], Optional[xr.DataArray], Optional[xr.DataArray]]:
        """
        Calculates the Palmer Hydrological Drought Index (PHDI), Palmer Drought Severity Index (PDSI),
        Relative Soil Moisture (RSM), and Relative Water Deficit (RWD)
        
        Args:
            aggregated_cube_ds: Dataset containing precipitation and PET data
            awc_s_static_2d: Available water capacity for surface layer
            awc_u_static_2d: Available water capacity for unsaturated layer
            s_name_for_log: Segment name for logging
            c_idx_abs: Cube index for logging
            method: Calculation method - 'cafec' for classic CAFEC calibration or 'z-score' for Z-score method
        
        Returns: (phdi, pdsi, rsm, rwd)
        """
        #if method not in ['cafec', 'z-score', 'terragon']:
        #    raise ValueError("Method must be 'cafec', 'z-score', or 'terragon'")
        if method not in ['cafec', 'z-score', 'terragon', 'cafec_monthly', 'pseudo_weekly']:
            raise ValueError("Method must be 'cafec', 'z-score', 'terragon', 'cafec_monthly', or 'pseudo_weekly'")

        
        P_monthly = self._handle_nans(aggregated_cube_ds.get('prcp'), 'prcp_agg_palmer', s_name_for_log, c_idx_abs)
        PET_monthly = self._handle_nans(aggregated_cube_ds.get('pet'), 'pet_agg_palmer', s_name_for_log, c_idx_abs) 

        # Initialize RSM and RWD results
        rsm_result = None
        rwd_result = None

        if not all(v is not None for v in [P_monthly, PET_monthly, awc_s_static_2d, awc_u_static_2d]):
            print(f"   Palmer Error: Missing base inputs (P, PET, or split AWC) for cube {c_idx_abs} in {s_name_for_log}")
            return None, None, None, None

        # This block will handle the data aggregation if the monthly method is chosen.
        if method == 'cafec_monthly':
            print(f"  INFO: Using HYBRID 'cafec_monthly' method for cube {c_idx_abs}.")
            # Step 1: Resample the weekly data to monthly sums for calculation.
            P_for_calc = P_monthly.resample(time='1M').sum()
            PET_for_calc = PET_monthly.resample(time='1M').sum()
        else:
            # For all other methods, use the original weekly data.
            # The name is monthly, but it depends on the input. The name is given because the it initially works for monthly data.
            P_for_calc = P_monthly
            PET_for_calc = PET_monthly
        # ---------------------------

        P_monthly = P_for_calc
        PET_monthly = PET_for_calc
        ET_actual_monthly = None
        R_total_monthly = None
        L_total_monthly = None
        RO_monthly = None
        S_s_series = None
        S_u_series = None

        # ONLY use the two-layer SMA for Palmer calculation
        if method != 'pseudo_weekly':
            ET_actual_monthly, R_total_monthly, L_total_monthly, RO_monthly, S_s_series, S_u_series, PL_series, PR_series= \
                self._perform_soil_moisture_accounting_two_layer(
                    P_monthly, PET_monthly, 
                    awc_s_static_2d, awc_u_static_2d,
                    s_name_for_log, c_idx_abs
                )

            if ET_actual_monthly is None or RO_monthly is None: 
                print(f"   Palmer Error: Two-Layer SMA failed for cube {c_idx_abs} in {s_name_for_log}")
                return None, None, None, None, None, None, None, None
        
        try:
            # Initialize default persistence factor
            effective_persistence_factor = 0.897  # Default monthly persistence factor
            
            # Choose Z-index calculation method
            if method == 'z-score':
                # Original Z-score method, the k factor is approximated from Palmer (1965), not using CAFEC calibration
                # The z-score method is not used anymore, but kept here for reference.
                # calculate actual ET from the two-layer SMA
                D_calc = P_monthly - ET_actual_monthly - RO_monthly 
                
                # Palmer k factor calculation
                denominator_K = ET_actual_monthly + P_monthly + RO_monthly 
                K_numerator = PET_monthly + P_monthly 
                K_factor = xr.where(denominator_K != 0, K_numerator / denominator_K, 1.0) 
                K_factor = K_factor.fillna(1.0) 

                # kd calculation
                KD = K_factor * D_calc
                KD_mean = KD.mean(dim='time', skipna=True)
                KD_std_original = KD.std(dim='time', skipna=True)

                # Z-score calculation
                Z_unclipped = self._calculate_z_score(KD, KD_mean, KD_std_original)
                Z_unclipped = self._handle_nans(Z_unclipped, 'Z_unclipped', s_name_for_log, c_idx_abs)
                
                if Z_unclipped is None: 
                    return None, None, None, None
                
                Z_index = Z_unclipped.clip(-12, 12)
                
            
            elif method == 'cafec':
                ## 1. (Auto-detect time frequency) ##
                # Check the time intervals in the monthly data to determine if it's monthly or weekly
                # This assumes that the time dimension is uniformly spaced.
                time_deltas = np.diff(P_monthly.time.values)
                median_delta_ns = np.median(time_deltas)
                timespan = pd.to_timedelta(median_delta_ns)
                
                print(f"  INFO CAFEC: Auto-detected data frequency. Median timespan is ~{timespan.days} days.")

                ## 2.  (Set decision threshold) ##
                # if the median timespan is greater than 15 days, we assume it's monthly data.
                IS_MONTHLY_DATA = timespan.days > 15

                if IS_MONTHLY_DATA:
                    effective_persistence_factor = 0.897  # Monthly persistence factor from the paper
                    print(f"  --> Frequency appears to be monthly. Using persistence factor: {effective_persistence_factor}")
                else:
                    effective_persistence_factor = 0.975  # Weekly persistence factor from the paper
                    print(f"  --> Frequency appears to be weekly. Using persistence factor: {effective_persistence_factor}")


                ## 3. (Choose appropriate CAFEC method based on frequency) ##
                if IS_MONTHLY_DATA:
                    print(f"  --> Frequency appears to be monthly. Using MONTHLY CAFEC calibration (12 coefficients).")
                    # if it's monthly data, call the new monthly CAFEC Z-index calculation
                    Z_index = self._calculate_z_index_cafec_monthly(
                        P_monthly, PET_monthly, ET_actual_monthly, R_total_monthly, 
                        L_total_monthly, RO_monthly, S_s_series, S_u_series,
                        awc_s_static_2d, awc_u_static_2d, s_name_for_log, c_idx_abs
                    )
                else:
                    print(f"  --> Frequency appears to be weekly. Using WEEKLY CAFEC calibration (53 coefficients).")
                    # if it's weekly data, call the original CAFEC Z-index calculation
                    Z_index = self._calculate_z_index_cafec(
                        P_monthly, PET_monthly, ET_actual_monthly, R_total_monthly, 
                        L_total_monthly, RO_monthly, S_s_series, S_u_series,
                        awc_s_static_2d, awc_u_static_2d, s_name_for_log, c_idx_abs
                    )
            
                if Z_index is None:
                    return None, None, None, None
                # Clip Z-index to reasonable range
                Z_index = Z_index.clip(-50, 50)

            elif method == 'terragon':
            # This method is just an experiment, not used in production.
                print(f"  Using Terragon-style Z-index calculation for cube {c_idx_abs}")
                Z_index = self._calculate_z_index_terragon_style(
                    P_monthly, ET_actual_monthly, RO_monthly, s_name_for_log, c_idx_abs
                )
                effective_persistence_factor = 0.897  # Use monthly persistence factor for Terragon method

            elif method == 'pseudo_weekly':
            # This is the most important new method added recently, it serves for weekly data input but uses moving window approach.
            # The 'pseudo_weekly' method is designed to handle weekly input data by applying a moving window approach, aiming to reflect monthly dynamics at a weekly level.
                print(f"  Using Pseudo-Weekly (moving window) method for cube {c_idx_abs}")
                
                # Check if input data is weekly
                time_deltas = np.diff(P_monthly.time.values)
                median_delta_ns = np.median(time_deltas)
                timespan = pd.to_timedelta(median_delta_ns)
                
                if timespan.days > 15:
                    # Actually what we input is the weekly data, this is just a warning in case the user inputs monthly data by mistake.
                    print(f"  WARNING Pseudo-Weekly: Input data appears to be monthly (timespan: {timespan.days} days). "
                          "Pseudo-weekly method is designed for weekly input data.")
                    if ET_actual_monthly is None:
                        print(f"  INFO: Pseudo-weekly fallback triggered, computing missing SMA variables...")
                        ET_actual_monthly, R_total_monthly, L_total_monthly, RO_monthly, S_s_series, S_u_series, PL_series, PR_series = \
                            self._perform_soil_moisture_accounting_two_layer(
                                P_monthly, PET_monthly, 
                                awc_s_static_2d, awc_u_static_2d,
                                s_name_for_log, c_idx_abs
                            )
                        # Check if SMA calculation was successful
                    if ET_actual_monthly is None:
                        return None, None, None, None
                    # For monthly data, fall back to standard monthly method
                    Z_index = self._calculate_z_index_cafec_monthly(
                        P_monthly, PET_monthly, ET_actual_monthly, R_total_monthly, 
                        L_total_monthly, RO_monthly, S_s_series, S_u_series,
                        awc_s_static_2d, awc_u_static_2d, s_name_for_log, c_idx_abs
                    )
                    effective_persistence_factor = 0.897  # Monthly persistence factor
                else:
                    print(f"  DEBUG Pseudo-Weekly: Input data is weekly (timespan: {timespan.days} days), "
                          "proceeding with pseudo-weekly analysis")
                    
                    # a. Get base coefficients from standard monthly calibration
                    print("  DEBUG Pseudo-Weekly: Step a - Getting base monthly coefficients")
                    cafec_coeffs, k_factors = self._get_standard_monthly_calibration_coeffs(
                        P_monthly, PET_monthly, awc_s_static_2d, awc_u_static_2d, s_name_for_log, c_idx_abs
                    )
                    
                    if cafec_coeffs is None or k_factors is None:
                        print(f"  ERROR Pseudo-Weekly: Failed to get base coefficients for cube {c_idx_abs}")
                        return None, None, None, None
                    
                    # b. Create moving window data (4-week sliding windows)
                    print("  DEBUG Pseudo-Weekly: Step b - Creating moving window data")
                    window_size_weeks = 4
                    P_windowed = self._create_moving_window_data(P_monthly, window_size_weeks, window_pattern="backward")
                    PET_windowed = self._create_moving_window_data(PET_monthly, window_size_weeks, window_pattern="backward")
                    
                    # c. Run water balance model on windowed data
                    print("  DEBUG Pseudo-Weekly: Step c - Running water balance on windowed data")
                    ET_windowed, R_windowed, L_windowed, RO_windowed, Ss_windowed, Su_windowed, PL_windowed, PR_windowed = \
                        self._perform_soil_moisture_accounting_two_layer(
                            P_windowed, PET_windowed, awc_s_static_2d, awc_u_static_2d, s_name_for_log, c_idx_abs
                        )
                    # Here ET_windowed and Ss_windowed, Su_windowed will be used for computing RSM and RWD
                    # R_windowed and L_windowed are not used further in this method, but could be logged or analyzed if needed, so they are returned.
                    
                    if ET_windowed is None or RO_windowed is None:
                        print(f"  ERROR Pseudo-Weekly: Water balance on windowed data failed for cube {c_idx_abs}")
                        return None, None, None, None
                    
                    # --- Start RSM and RWD calculation ---
                    # Calculate total available water capacity AWC, and clip to prevent division by zero
                    AWC_total = (awc_s_static_2d + awc_u_static_2d).clip(min=1e-6)
                    
                    # Calculate total soil water storage SWS
                    SWS_windowed = Ss_windowed + Su_windowed
                    
                    # Calculate RSM
                    rsm_result = 100 * (SWS_windowed / AWC_total).clip(0, 1)
                    rsm_result = rsm_result.clip(0, 100)
                    rsm_result.name = 'rsm'
                    rsm_result.attrs = {
                        'units': '%',
                        'long_name': 'Relative Soil Moisture',
                        'description': 'Soil Water Storage / AWC, from pseudo-weekly water balance.'
                    }
                    
                    # Calculate RWD, and clip to prevent division by zero
                    rwd_raw = 100 * (1 - (ET_windowed / PET_windowed))
                    rwd_result = xr.where(PET_windowed > 1e-6, rwd_raw, 0.0)
                    rwd_result = rwd_result.where(PET_windowed.notnull())
                    rwd_result = rwd_result.clip(0, 100)
                    rwd_result.name = 'rwd'
                    rwd_result.attrs = {
                        'units': '%',
                        'long_name': 'Relative Water Deficit',
                        'description': '(1 - AET/PET) * 100, from pseudo-weekly water balance.'
                    }
                    print(f"  DEBUG Pseudo-Weekly: RSM and RWD calculated successfully")
                    # --- End RSM and RWD calculation ---
                    
                    # d. Interpolate all coefficients to weekly time points
                    print("  DEBUG Pseudo-Weekly: Step d - Interpolating coefficients to weekly time points")
                    target_times = pd.DatetimeIndex(P_monthly.time.values)
                    
                    alpha_interp = self._interpolate_monthly_coefficients(target_times, cafec_coeffs['alpha'])
                    beta_interp = self._interpolate_monthly_coefficients(target_times, cafec_coeffs['beta'])
                    gamma_interp = self._interpolate_monthly_coefficients(target_times, cafec_coeffs['gamma'])
                    delta_interp = self._interpolate_monthly_coefficients(target_times, cafec_coeffs['delta'])
                    k_factors_interp = self._interpolate_monthly_coefficients(target_times, k_factors)
                    
                    # e. Calculate Z-index time series
                    print("  DEBUG Pseudo-Weekly: Step e - Calculating Z-index time series")
                    AWC_total = awc_s_static_2d + awc_u_static_2d
                    SP_full = AWC_total - PR_windowed
                    PR_full = PR_windowed
                    PL_full = PL_windowed
                    #SP_full = Ss_windowed + Su_windowed
                    # PR_full = AWC_total - SP_full
                    
                    # Calculate potential loss for windowed data
                    # PL_full = xr.where(Ss_windowed >= PET_windowed,
                    #                   PET_windowed,
                    #                   np.minimum(Ss_windowed + Su_windowed,
                    #                             ((PET_windowed - Ss_windowed) * Su_windowed) / 
                    #                             (awc_u_static_2d + awc_s_static_2d) + Ss_windowed))
                    
                    # Calculate moisture departure d = P - P_hat
                    p_hat = (alpha_interp * PET_windowed +
                             beta_interp * PR_full +
                             gamma_interp * SP_full -
                             delta_interp * PL_full)
                    
                    d = P_windowed - p_hat
                    Z_index = k_factors_interp * d

                    # f. Normalize and smooth Z-index
                    # Because our purpose is reflecting monthly drought condition at weekly level, we need to normalize and smooth the Z-index time series.
                    # Considering one month usually has 4-5 week time stamps, we choose a normalization factor of 4.4 (average weeks per month) here.
                    # Similar reason for the window size of the rolling mean, we choose 5-week rolling mean to smooth the Z-index.
                    # This helps to reduce high frequency noise and better represent monthly drought dynamics.
                    # This value can be adjusted based on further calibration or validation studies.
    
                    Z_normalizer = 4.4
                    Z_index = Z_index / Z_normalizer
                    print(f"  DEBUG Pseudo-Weekly: Z-index calculated with normalization factor {Z_normalizer}")
                    window_for_zindex = 5
                    center_rolling_for_zindex = False  
                    Z_index = Z_index.rolling(time=window_for_zindex, center=center_rolling_for_zindex, min_periods=1).mean()
                    print(f"  DEBUG Pseudo-Weekly: Applied {window_for_zindex}-week rolling mean to Z-index, center = {center_rolling_for_zindex}")
                    
                    print(f"  DEBUG Pseudo-Weekly: Z-index calculation completed. "
                          f"Z statistics - mean: {Z_index.mean().item():.3f}, "
                          f"std: {Z_index.std().item():.3f}, "
                          f"min: {Z_index.min().item():.3f}, "
                          f"max: {Z_index.max().item():.3f}")
                    
                    # Use monthly persistence factor since calculation units are "moving months"
                    effective_persistence_factor = 0.975
                    print(f"  DEBUG Pseudo-Weekly: Using persistence factor {effective_persistence_factor} for pseudo-weekly method")
                
                # Set attributes
                Z_index.name = 'z_index_pseudo_weekly'
                Z_index.attrs = {
                    'units': 'unitless',
                    'long_name': 'Palmer Z-Index (Pseudo-Weekly Method)',
                    'description': 'Z-Index calculated using Palmer 1965 pseudo-weekly method with moving windows and interpolated monthly coefficients'
                }
                
                # Clip Z-index to reasonable range
                Z_index = Z_index.clip(-50, 50)

            elif method == 'cafec_monthly':
                print(f"  Using CAFEC Monthly-style Z-index calculation for cube {c_idx_abs}")
                Z_index = self._calculate_z_index_cafec(
                    P_monthly, PET_monthly, ET_actual_monthly, R_total_monthly, 
                    L_total_monthly, RO_monthly, S_s_series, S_u_series,
                    awc_s_static_2d, awc_u_static_2d, s_name_for_log, c_idx_abs
                )
                Z_index = Z_index.clip(-50, 50)
                effective_persistence_factor = 0.897  # Use monthly persistence factor for cafec_monthly method
            
            # Apply Palmer state machine to calculated Z-index
            phdi_result, pdsi_result = self._palmer_state_machine_index_full(Z_index, effective_persistence_factor)

            if phdi_result is None or pdsi_result is None:
                return None, None, None, None
            
            # Clip the final results to a reasonable range for PHDI and PDSI
            phdi_final = phdi_result.clip(-12, 12)
            pdsi_final = pdsi_result.clip(-12, 12)

            # #Refill the nan values that were in the original input data
            valid_mask_da = P_monthly.notnull().any(dim='time')
            mask_values = valid_mask_da.values  # Shape: (y, x)

            # check shape compatibility before applying mask
            if mask_values.shape == pdsi_final.shape[-2:]:
                # Apply the mask to final results
                phdi_final = phdi_final.where(mask_values)
                pdsi_final = pdsi_final.where(mask_values)
                print(f"   DEBUG: Applied NaN mask using numpy values. Mask shape: {mask_values.shape}")
            else:
                # If shapes do not match, log a warning and skip masking
                print(f"   WARNING: Mask shape mismatch! PDSI: {pdsi_final.shape}, Mask: {mask_values.shape}. Skipping mask application.")
            
            # Set attributes based on method
            method_suffix = 'CAFEC' if method == 'cafec' else 'Z-Score'
            if method == 'cafec':
                print(f"   Palmer calculation using CAFEC method for cube_abs {c_idx_abs}, seg {s_name_for_log}")
            
            phdi_final.name = f'phdi_palmer_state_machine_{method.lower()}'
            phdi_final.attrs = {
                'units': 'unitless', 
                'long_name': f'Palmer Hydrological Drought Index (Palmer State Machine, {method_suffix})',
                'description': f'PHDI calculated using Palmer state machine with {method_suffix} Z-index calculation, x3 component representing established drought/wet spell intensity'
            }
            
            pdsi_final.name = f'pdsi_palmer_state_machine_{method.lower()}'
            pdsi_final.attrs = {
                'units': 'unitless', 
                'long_name': f'Palmer Drought Severity Index (Palmer State Machine, {method_suffix})',
                'description': f'PDSI calculated using Palmer state machine with {method_suffix} Z-index calculation, comprehensive logic including x1, x2, x3 components'
            }
            
            return phdi_final, pdsi_final, rsm_result, rwd_result
            
        except Exception as e_calc:
            print(f"   Error during Palmer calculation for cube_abs {c_idx_abs}, seg {s_name_for_log}: {e_calc}")
            return None, None, None, None

    def process_palmer_indices_for_current_segment(self,
                                                    target_palmer_variables: List[str],
                                                    output_time_chunk_size: int = 4,
                                                    method: str = 'cafec'):
            """
            Process Palmer drought indices for the current segment.
            This version discovers cube directories dynamically instead of relying on a count.
            """
            ### MODIFICATION START ###
            # 1. Discover the actual cube directories that exist in the segment path.
            try:
                # This creates a list of subdirectories whose names are digits, e.g., ['3', '10', '11']
                all_dirs = os.listdir(self.aggregated_daymet_segment_path)
                cube_names_to_process = [d for d in all_dirs
                                        if d.isdigit() and os.path.isdir(os.path.join(self.aggregated_daymet_segment_path, d))]
                # Sort the list numerically to ensure processing in the correct order.
                cube_names_to_process.sort(key=int)
            except OSError as e:
                print(f"  CRITICAL: Could not read directories from segment path {self.aggregated_daymet_segment_path}: {e}")
                return # Exit if the segment directory is unreadable.

            if not cube_names_to_process:
                print(f"  INFO: No valid cube directories found in {self.aggregated_daymet_segment_path}. Skipping Palmer processing for this segment.")
                return
            
            print(f"  INFO: Found {len(cube_names_to_process)} cubes to process in segment: {cube_names_to_process if len(cube_names_to_process) < 10 else str(len(cube_names_to_process))+' cubes'}")

            # 2. Loop over the discovered list of cube names, not a range of numbers.
            for cube_name in cube_names_to_process:
                # 3. Build paths directly from the cube_name.
                abs_cn_for_log_and_save = int(cube_name)
                agg_daymet_cube_path = os.path.join(self.aggregated_daymet_segment_path, cube_name)
                
                # The 'if not os.path.exists' check is now technically redundant because we just listed
                # the directories, but it's kept as a safeguard against race conditions.
                if not os.path.exists(agg_daymet_cube_path):
                    print(f"  WARNING: Cube path {agg_daymet_cube_path} disappeared after being listed. Skipping.")
                    continue
                
                ### MODIFICATION END ###

                # The rest of the function logic remains largely the same.
                cube_res_to_save: Dict[str, xr.DataArray] = {}

                try:
                    with xr.open_zarr(agg_daymet_cube_path, consolidated=False, chunks='auto') as ds_aggregated_daymet:
                        # Step 1: Check if 'prcp' variable exists and is not empty
                        if 'prcp' not in ds_aggregated_daymet or ds_aggregated_daymet['prcp'].size == 0:
                            print(f"  ERROR: 'prcp' not found or is empty in {agg_daymet_cube_path}. Skipping cube {abs_cn_for_log_and_save}.")
                            continue
                        
                        # Obtain the target time coordinate and spatial template from the 'prcp' variable
                        target_time_coord_da = ds_aggregated_daymet['prcp'].coords['time']
                        
                        # Create a time coordinate array with the specified output chunk size
                        target_spatial_template = ds_aggregated_daymet['prcp'].isel(
                            time=0, drop=True, missing_dims='ignore'
                        ).compute() 

                        # initialize AWC variables (for surface and unsaturated layers)
                        awc_s_static_2d = None
                        awc_u_static_2d = None
                        awc_total_static_2d = None  # Added for CMI calculation
                        awc_input_ok_for_sma = False # signal for whether AWC is ready for SMA

                        if 'phdi' in target_palmer_variables or 'pdsi' in target_palmer_variables or 'cmi' in target_palmer_variables:
                            # Logic to find the corresponding soil cube
                            if self.soil_segment_path:
                                # We assume the soil cube has the same name as the daymet cube
                                awc_cube_source_path = os.path.join(self.soil_segment_path, cube_name)
                                if not os.path.exists(awc_cube_source_path):
                                    print(f"  WARNING: Corresponding soil cube path does not exist: {awc_cube_source_path}. AWC will be effectively None.")
                                else:
                                    try:
                                        with xr.open_zarr(awc_cube_source_path, consolidated=False, chunks='auto') as ds_awc_source:
                                            if 'awc' in ds_awc_source:
                                                awc_for_cube_total_raw = 1.5 * ds_awc_source['awc']/(25.4) # convert from mm to inches
                                                # The coefficient 1.5 is used to amplying the AWC values to better fit the suggestions of Palmer.
                                                # As pointed by Palmer himself, using 8 inches and 4 inches do not make a big difference in general result.
                                                # But using awc values which are small (e.g. 2 inches) may lead to disability to capture extreme drought conditions.
                                                # Actually, PDSI and PHDI are not sensitive to the exact AWC values, but rather the relative differences across space.
                                                # That is why, in experiments of palmer, a proper value can be assigned to be used across a state.
                                                # Our awc_fraction is properly estimated, the awc_mm or awc_inch are decided by "how deep the soil layer is considered".
                                                # In the suggestions of Palmer, it is helpful to consider the soil layer deeper.
                                                # Therefore the input awc calculated by considering 1500mm is amplified here, this can be adjusted in the future if needed.
                                                if hasattr(awc_for_cube_total_raw, 'chunks') and awc_for_cube_total_raw.chunks is not None:
                                                    awc_for_cube_total_raw = awc_for_cube_total_raw.compute() #load original awc
                                                
                                                awc_for_cube_total_processed = self._handle_nans(awc_for_cube_total_raw,
                                                                                                'awc_total_raw_nan_handled',
                                                                                                self.segment_name_soil or "soil",
                                                                                                abs_cn_for_log_and_save)

                                                if awc_for_cube_total_processed is not None and awc_for_cube_total_processed.size > 0:
                                                    awc_to_resample = awc_for_cube_total_processed
                                                    if 'time' in awc_to_resample.dims and awc_to_resample.sizes['time'] == 1:
                                                        awc_to_resample = awc_to_resample.isel(time=0, drop=True)
                                                    elif 'time' in awc_to_resample.dims and awc_to_resample.sizes['time'] > 1:
                                                        print(f"  WARNING: AWC data for cube {abs_cn_for_log_and_save} has multiple time steps. Using the first time step for AWC.")
                                                        awc_to_resample = awc_to_resample.isel(time=0, drop=True)

                                                    # Step 2: Checking and resampling AWC
                                                    resample_needed = False
                                                    current_awc_coords = awc_to_resample.coords
                                                    if awc_to_resample.shape != target_spatial_template.shape or \
                                                    not current_awc_coords.get('x', pd.Index([])).equals(target_spatial_template.coords['x']) or \
                                                    not current_awc_coords.get('y', pd.Index([])).equals(target_spatial_template.coords['y']):
                                                        resample_needed = True
                                                    
                                                    if resample_needed:
                                                        print(f"  Resampling AWC data for cube {abs_cn_for_log_and_save} from grid with shape {awc_to_resample.shape} to target grid shape {target_spatial_template.shape}")
                                                        awc_for_cube_total_resampled = awc_to_resample.interp_like(
                                                            target_spatial_template,
                                                            method="nearest",
                                                            kwargs={"fill_value": "extrapolate"} 
                                                        )
                                                    else:
                                                        awc_for_cube_total_resampled = awc_to_resample

                                                    # Step 3: Using resampled AWC for stratification and CMI
                                                    if awc_for_cube_total_resampled is not None and awc_for_cube_total_resampled.size > 0 :
                                                        awc_total_static_2d = awc_for_cube_total_resampled
                                                        
                                                        AWC_s_fixed_mm = 1.0 #The name of variable is in mm, but value is in inches, for the purpose of using the code, the name remains
                                                        awc_s_static_2d = xr.full_like(target_spatial_template, AWC_s_fixed_mm, dtype=awc_for_cube_total_resampled.dtype)
                                                        awc_u_static_2d = (awc_for_cube_total_resampled - AWC_s_fixed_mm).clip(min=0)
                                                        
                                                        awc_s_static_2d = xr.where(awc_for_cube_total_resampled < AWC_s_fixed_mm,
                                                                                awc_for_cube_total_resampled,
                                                                                awc_s_static_2d)
                                                        awc_u_static_2d = xr.where(awc_for_cube_total_resampled < AWC_s_fixed_mm,
                                                                                xr.zeros_like(awc_for_cube_total_resampled, dtype=awc_for_cube_total_resampled.dtype),
                                                                                awc_u_static_2d)
                                                        awc_input_ok_for_sma = True
                                                    else:
                                                        print(f"  WARNING: Resampled AWC is None or empty for cube {abs_cn_for_log_and_save}. Cannot split AWC.")
                                                else:
                                                    print(f"  WARNING: Raw AWC is None or empty after NaN handling for cube {abs_cn_for_log_and_save}. Cannot resample or split AWC.")
                                            else:
                                                print(f"  WARNING: 'awc' not found in {awc_cube_source_path} for cube {abs_cn_for_log_and_save}")
                                    except Exception as e_awc_load:
                                        print(f"  ERROR loading/processing AWC from {awc_cube_source_path} for cube {abs_cn_for_log_and_save}: {e_awc_load}")
                            else:
                                print(f"  INFO: Soil data path not provided. Proceeding without AWC for PHDI/PDSI/CMI (will likely fail or skip in calc functions).")
                        
                        # Step 4: Loading and aligning Prcp, PET 
                        P_monthly_orig = ds_aggregated_daymet.get('prcp')/25.4 #The name of variable is monthly, because it was initially designed for monthly data, now it serves for weekly data, but the name remains.
                        # Deficit_monthly_orig = ds_aggregated_daymet.get('Deficit-1')/25.4 
                        PET_monthly_orig = ds_aggregated_daymet.get('pet')/25.4 #Similar as above comment.
                        
                        P_monthly_aligned = None
                        PET_monthly_aligned = None
                        daymet_vars_ok_for_sma = False

                        if P_monthly_orig is not None and PET_monthly_orig is not None:
                            P_monthly_loaded = P_monthly_orig.compute() if hasattr(P_monthly_orig, 'chunks') else P_monthly_orig
                            PET_monthly_loaded = PET_monthly_orig.compute() if hasattr(PET_monthly_orig, 'chunks') else PET_monthly_orig

                            P_monthly_spatial_part = P_monthly_loaded.isel(time=0, drop=True, missing_dims='ignore')
                            if P_monthly_spatial_part.shape != target_spatial_template.shape or \
                            not P_monthly_spatial_part.coords.get('x', pd.Index([])).equals(target_spatial_template.coords['x']) or \
                            not P_monthly_spatial_part.coords.get('y', pd.Index([])).equals(target_spatial_template.coords['y']):
                                print(f"  Aligning P_monthly for cube {abs_cn_for_log_and_save} to target grid.")
                                P_monthly_aligned = P_monthly_loaded.interp_like(target_spatial_template.expand_dims(time=target_time_coord_da), method="linear")
                            else:
                                P_monthly_aligned = P_monthly_loaded
                            P_monthly_aligned = self._handle_nans(P_monthly_aligned, 'prcp_aligned', self.segment_name_agg_daymet, abs_cn_for_log_and_save)

                            PET_monthly_spatial_part = PET_monthly_loaded.isel(time=0, drop=True, missing_dims='ignore')
                            if PET_monthly_spatial_part.shape != target_spatial_template.shape or \
                            not PET_monthly_spatial_part.coords.get('x', pd.Index([])).equals(target_spatial_template.coords['x']) or \
                            not PET_monthly_spatial_part.coords.get('y', pd.Index([])).equals(target_spatial_template.coords['y']):
                                print(f"  Aligning PET_monthly for cube {abs_cn_for_log_and_save} to target grid.")
                                PET_monthly_aligned = PET_monthly_loaded.interp_like(target_spatial_template.expand_dims(time=target_time_coord_da), method="linear")
                            else:
                                PET_monthly_aligned = PET_monthly_loaded
                            PET_monthly_aligned = self._handle_nans(PET_monthly_aligned, 'pet_aligned', self.segment_name_agg_daymet, abs_cn_for_log_and_save)
                            
                            if P_monthly_aligned is not None and PET_monthly_aligned is not None:
                                daymet_vars_ok_for_sma = True
                            else:
                                print(f"  ERROR: Prcp or PET is None after alignment/NaN handling for cube {abs_cn_for_log_and_save}.")
                        else:
                            print(f"  ERROR: Original Prcp or PET not found in ds_aggregated_daymet for cube {abs_cn_for_log_and_save}.")

                        if ('phdi' in target_palmer_variables or 'pdsi' in target_palmer_variables or 'rsm' in target_palmer_variables or 'rwd' in target_palmer_variables):
                            if awc_input_ok_for_sma and daymet_vars_ok_for_sma:
                                ds_for_calc = xr.Dataset({'prcp': P_monthly_aligned, 'pet': PET_monthly_aligned})
                                
                                phdi_val, pdsi_val, rsm_val, rwd_val = self._calculate_phdi_and_pdsi_from_aggregated_inputs(
                                    ds_for_calc, awc_s_static_2d, awc_u_static_2d,
                                    self.segment_name_agg_daymet, abs_cn_for_log_and_save, method
                                )
                                
                                if 'phdi' in target_palmer_variables and phdi_val is not None: 
                                    cube_res_to_save['phdi'] = phdi_val
                                if 'pdsi' in target_palmer_variables and pdsi_val is not None: 
                                    cube_res_to_save['pdsi'] = pdsi_val
                                if 'rsm' in target_palmer_variables and rsm_val is not None:
                                    cube_res_to_save['rsm'] = rsm_val
                                if 'rwd' in target_palmer_variables and rwd_val is not None:
                                    cube_res_to_save['rwd'] = rwd_val
                            else:
                                missing_components = []
                                if not awc_input_ok_for_sma: missing_components.append("AWC")
                                if not daymet_vars_ok_for_sma: missing_components.append("Daymet P/PET")
                                print(f"  Skipping PHDI/PDSI/RSM/RWD for cube {abs_cn_for_log_and_save} due to missing: {', '.join(missing_components)}")

                        if 'cmi' in target_palmer_variables:
                            if awc_input_ok_for_sma and daymet_vars_ok_for_sma and awc_total_static_2d is not None:
                                ds_for_calc = xr.Dataset({'prcp': P_monthly_aligned, 'pet': PET_monthly_aligned})
                                cmi_val = self._calculate_cmi_from_aggregated_inputs(
                                    ds_for_calc, awc_total_static_2d,
                                    self.segment_name_agg_daymet, abs_cn_for_log_and_save
                                )
                                if cmi_val is not None: cube_res_to_save['cmi'] = cmi_val
                            else:
                                missing_components = []
                                if not awc_input_ok_for_sma: missing_components.append("AWC")
                                if not daymet_vars_ok_for_sma: missing_components.append("Daymet P/PET")
                                if awc_total_static_2d is None: missing_components.append("total AWC")
                                print(f"  Skipping CMI for cube {abs_cn_for_log_and_save} due to missing: {', '.join(missing_components)}")

                        if cube_res_to_save:
                            chunked_palmer_results = {}
                            for name, da_val in cube_res_to_save.items():
                                if da_val is None or da_val.size == 0: 
                                    print(f"  Skipping empty DataArray for variable {name} in cube {abs_cn_for_log_and_save}")
                                    continue

                                target_chunks_palmer = {}
                                current_dims = da_val.dims
                                
                                if 'time' in current_dims and da_val.sizes['time'] > 0: 
                                    target_chunks_palmer['time'] = min(output_time_chunk_size, da_val.sizes['time'])
                                
                                for dim_name in current_dims: 
                                    if dim_name != 'time' and da_val.sizes[dim_name] > 0:
                                        target_chunks_palmer[dim_name] = da_val.sizes[dim_name]
                                
                                final_chunks = {}
                                for dim_name in current_dims:
                                    if dim_name in target_chunks_palmer and da_val.sizes.get(dim_name, 0) > 0:
                                        final_chunks[dim_name] = target_chunks_palmer[dim_name]
                                
                                print(f"  DEBUG: Chunk config for {name} in cube {abs_cn_for_log_and_save}: {final_chunks} (original shape: {da_val.shape})")
                                
                                if (final_chunks and 
                                    len(final_chunks) == len(current_dims) and 
                                    all(final_chunks.get(d, 0) > 0 for d in current_dims) and 
                                    da_val.size > 0):
                                    chunked_palmer_results[name] = da_val.chunk(final_chunks)
                                else: 
                                    print(f"  WARNING: Using original data array for {name} due to chunking config issues")
                                    chunked_palmer_results[name] = da_val 
                                
                            if chunked_palmer_results:
                                ds_to_save = xr.Dataset(chunked_palmer_results)
                                self._save_cube_dataset(abs_cn_for_log_and_save, ds_to_save, self.palmer_output_segment_path,
                                                        f"Palmer_FromAgg_2L_SMA_Resampled({','.join(list(cube_res_to_save.keys()))})")

                except FileNotFoundError: 
                    print(f"  WARNING: Zarr store not found for cube {abs_cn_for_log_and_save} at {agg_daymet_cube_path}, skipping.")
                except Exception as e_outer_cube_loop:
                    print(f"  ERROR processing Palmer for aggregated cube_abs {abs_cn_for_log_and_save} of {self.segment_name_agg_daymet}: {str(e_outer_cube_loop)}")
                    traceback.print_exc()
    
    def _calculate_cafec_coefficients_monthly(self,
                                              monthly_sums: Dict[str, np.ndarray],
                                              calibration_years: int) -> Dict[str, np.ndarray]:
        """
        Calculate CAFEC coefficients (alpha, beta, gamma, delta) from monthly sums.
        This version is specifically for 12 monthly periods.

        Args:
            monthly_sums: Dictionary containing 12-element arrays for each variable sum over the calibration period.
            calibration_years: Number of years in the calibration period.

        Returns:
            Dictionary containing the four CAFEC coefficients as 12-element arrays.
        """
        print(f"    DEBUG CAFEC Monthly: Starting CAFEC coefficient calculation for "
              f"{calibration_years} calibration years (12 periods)")

        # Average the sums over the calibration period
        avg_sums = {key: values / calibration_years for key, values in monthly_sums.items()}
        
        # Print monthly averages for debugging
        for key, values in avg_sums.items():
            print(f"    DEBUG CAFEC Monthly: {key} monthly averages: "
                  f"{[f'{x:.2f}' for x in values[:3]]}... "
                  f"(showing first 3 months)")

        num_periods = 12  # Explicitly for 12 months
        alpha = np.zeros(num_periods)
        beta = np.zeros(num_periods)
        gamma = np.zeros(num_periods)
        delta = np.zeros(num_periods)

        print("    DEBUG CAFEC Monthly: Calculating alpha coefficients (ET/PET ratio)...")
        # Calculate alpha (ET_actual / PET)
        for i in range(num_periods):
            if avg_sums['petsum'][i] > 0:
                alpha[i] = avg_sums['etsum'][i] / avg_sums['petsum'][i]
            elif avg_sums['etsum'][i] == 0:
                alpha[i] = 1.0
            else:
                alpha[i] = 0.0

        print(f"    DEBUG CAFEC Monthly: Alpha monthly coefficients (first 6): "
              f"{[f'{x:.3f}' for x in alpha[:6]]}")
        print(f"    DEBUG CAFEC Monthly: Alpha monthly coefficients (last 6): "
              f"{[f'{x:.3f}' for x in alpha[-6:]]}")

        print("    DEBUG CAFEC Monthly: Calculating beta coefficients (R/PR ratio)...")
        # Calculate beta (R / PR)
        for i in range(num_periods):
            if avg_sums['prsum'][i] > 0:
                beta[i] = avg_sums['rsum'][i] / avg_sums['prsum'][i]
            elif avg_sums['rsum'][i] == 0:
                beta[i] = 1.0
            else:
                beta[i] = 0.0

        print(f"    DEBUG CAFEC Monthly: Beta monthly coefficients (first 6): "
              f"{[f'{x:.3f}' for x in beta[:6]]}")
        print(f"    DEBUG CAFEC Monthly: Beta monthly coefficients (last 6): "
              f"{[f'{x:.3f}' for x in beta[-6:]]}")

        print("    DEBUG CAFEC Monthly: Calculating gamma coefficients (RO/SP ratio)...")
        # Calculate gamma (RO / SP)
        for i in range(num_periods):
            if avg_sums['spsum'][i] > 0:
                gamma[i] = avg_sums['rosum'][i] / avg_sums['spsum'][i]
            elif avg_sums['rosum'][i] == 0:
                gamma[i] = 1.0
            else:
                gamma[i] = 0.0
        
        print(f"    DEBUG CAFEC Monthly: Gamma monthly coefficients (first 6): "
              f"{[f'{x:.3f}' for x in gamma[:6]]}")
        print(f"    DEBUG CAFEC Monthly: Gamma monthly coefficients (last 6): "
              f"{[f'{x:.3f}' for x in gamma[-6:]]}")

        print("    DEBUG CAFEC Monthly: Calculating delta coefficients (TL/PL ratio)...")
        # Calculate delta (TL / PL)
        for i in range(num_periods):
            if avg_sums['plsum'][i] > 0:
                delta[i] = avg_sums['tlsum'][i] / avg_sums['plsum'][i]
            else:
                delta[i] = 0.0

        print(f"    DEBUG CAFEC Monthly: Delta monthly coefficients (first 6): "
              f"{[f'{x:.3f}' for x in delta[:6]]}")
        print(f"    DEBUG CAFEC Monthly: Delta monthly coefficients (last 6): "
              f"{[f'{x:.3f}' for x in delta[-6:]]}")

        return {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'delta': delta
        }

    def _calculate_cafec_coefficients(self, 
                                     weekly_sums: Dict[str, np.ndarray], 
                                     calibration_years: int) -> Dict[str, np.ndarray]:
        """
        Calculate CAFEC coefficients (alpha, beta, gamma, delta) from weekly sums
        
        Args:
            weekly_sums: Dictionary containing 53-element arrays for each variable
            calibration_years: Number of years in calibration period
        
        Returns:
            Dictionary containing the four CAFEC coefficients as 53-element arrays
        """
        print(f"   DEBUG CAFEC: Starting CAFEC coefficient calculation for "
              f"{calibration_years} calibration years")
        
        # Average the sums over the calibration period
        avg_sums = {}
        for key, values in weekly_sums.items():
            avg_sums[key] = values / calibration_years
            print(f"   DEBUG CAFEC: {key} weekly averages: "
                  f"{[f'{x:.2f}' for x in avg_sums[key][:3]]}... "
                  f"(showing first 3 weeks)")
        
        # Initialize coefficient arrays
        alpha = np.zeros(53)
        beta = np.zeros(53) 
        gamma = np.zeros(53)
        delta = np.zeros(53)
        
        print("   DEBUG CAFEC: Calculating alpha coefficients (ET/PET ratio)...")
        # Calculate alpha (ET_actual / PET)
        for week in range(53):
            if avg_sums['petsum'][week] != 0:
                alpha[week] = avg_sums['etsum'][week] / avg_sums['petsum'][week]
            elif avg_sums['etsum'][week] == 0:
                alpha[week] = 1.0
            else:
                alpha[week] = 0.0
        
        print(f"   DEBUG CAFEC: Alpha weekly coefficients (first 5): "
              f"{[f'{x:.3f}' for x in alpha[:5]]}")
        print(f"   DEBUG CAFEC: Alpha weekly coefficients (last 5): "
              f"{[f'{x:.3f}' for x in alpha[-5:]]}")
        
        print("   DEBUG CAFEC: Calculating beta coefficients (R/PR ratio)...")
        # Calculate beta (R / PR) 
        for week in range(53):
            if avg_sums['prsum'][week] != 0:
                beta[week] = avg_sums['rsum'][week] / avg_sums['prsum'][week]
            elif avg_sums['rsum'][week] == 0:
                beta[week] = 1.0
            else:
                beta[week] = 0.0
        
        print(f"   DEBUG CAFEC: Beta weekly coefficients (first 5): "
              f"{[f'{x:.3f}' for x in beta[:5]]}")
        print(f"   DEBUG CAFEC: Beta weekly coefficients (last 5): "
              f"{[f'{x:.3f}' for x in beta[-5:]]}")
        
        print("   DEBUG CAFEC: Calculating gamma coefficients (RO/SP ratio)...")
        # Calculate gamma (RO / SP)
        for week in range(53):
            if avg_sums['spsum'][week] != 0:
                gamma[week] = avg_sums['rosum'][week] / avg_sums['spsum'][week]
            elif avg_sums['rosum'][week] == 0:
                gamma[week] = 1.0
            else:
                gamma[week] = 0.0
        
        print(f"   DEBUG CAFEC: Gamma weekly coefficients (first 5): "
              f"{[f'{x:.3f}' for x in gamma[:5]]}")
        print(f"   DEBUG CAFEC: Gamma weekly coefficients (last 5): "
              f"{[f'{x:.3f}' for x in gamma[-5:]]}")
        
        print("   DEBUG CAFEC: Calculating delta coefficients (TL/PL ratio)...")
        # Calculate delta (TL / PL)
        for week in range(53):
            if avg_sums['plsum'][week] != 0:
                delta[week] = avg_sums['tlsum'][week] / avg_sums['plsum'][week]
            else:
                delta[week] = 0.0
        
        print(f"   DEBUG CAFEC: Delta weekly coefficients (first 5): "
              f"{[f'{x:.3f}' for x in delta[:5]]}")
        print(f"   DEBUG CAFEC: Delta weekly coefficients (last 5): "
              f"{[f'{x:.3f}' for x in delta[-5:]]}")
        
        return {
            'alpha': alpha,
            'beta': beta/4.35,
            'gamma': gamma/4.35,
            'delta': delta
        }

    def _calculate_k_factors_monthly(self,
                                     monthly_sums: Dict[str, np.ndarray],
                                     cafec_coeffs: Dict[str, np.ndarray],
                                     calibration_data: Dict[str, xr.DataArray],
                                     calibration_years: int) -> np.ndarray:
        """
        Calculate K factors (monthly weighting factors) for Z-index calculation.
        This version is specifically for 12 monthly periods.

        Args:
            monthly_sums: Dictionary containing 12-element monthly sum arrays.
            cafec_coeffs: Dictionary containing 12-element CAFEC coefficients.
            calibration_data: Dictionary containing calibration period data as xarray DataArrays.
            calibration_years: Number of years in the calibration period.

        Returns:
            An array of 12 monthly K factors.
        """
        print(f"    DEBUG CAFEC Monthly: Starting K factor calculation for "
              f"{calibration_years} years (12 periods)")
        
        num_periods = 12  # Explicitly for 12 months

        # Average the sums over the calibration period
        avg_sums = {key: values / calibration_years for key, values in monthly_sums.items()}

        # Calculate T ratio for each month
        trat = np.zeros(num_periods)
        for i in range(num_periods):
            numerator = (avg_sums['petsum'][i] + avg_sums['rsum'][i] + avg_sums['rosum'][i])
            denominator = avg_sums['psum'][i] + avg_sums['tlsum'][i]
            trat[i] = numerator / denominator if denominator != 0 else 1.0

        print(f"    DEBUG CAFEC Monthly: Monthly T-ratios (first 6): "
              f"{[f'{x:.3f}' for x in trat[:6]]}")
        print(f"    DEBUG CAFEC Monthly: Monthly T-ratios (last 6): "
              f"{[f'{x:.3f}' for x in trat[-6:]]}")

        # Calculate absolute deviations for each month
        sabsd = np.zeros(num_periods)
        
        P_cal, PET_cal, PR_cal, SP_cal, PL_cal = [calibration_data[var] for var in ['prcp', 'pet', 'pr', 'sp', 'pl']]

        print(f"    DEBUG CAFEC Monthly: Calculating absolute deviations over "
              f"{P_cal.sizes['time']} time steps...")

        # Calculate deviations for each time step in the calibration period
        for t_idx in range(P_cal.sizes['time']):
            # CRITICAL CHANGE: Get month index (0-11) instead of week index
            time_val = P_cal.time.data[t_idx]
            month_idx = pd.Timestamp(time_val).month - 1  # .month is 1-12, so subtract 1 for 0-11 index
            
            # Ensure index is valid (should always be, but as a safeguard)
            month_idx = max(0, min(num_periods - 1, month_idx))

            p_hat = (cafec_coeffs['alpha'][month_idx] * PET_cal.isel(time=t_idx) +
                     cafec_coeffs['beta'][month_idx] * PR_cal.isel(time=t_idx) +
                     cafec_coeffs['gamma'][month_idx] * SP_cal.isel(time=t_idx) -
                     cafec_coeffs['delta'][month_idx] * PL_cal.isel(time=t_idx))
            
            d = P_cal.isel(time=t_idx) - p_hat
            
            # Accumulate absolute deviation for this month
            sabsd[month_idx] += np.abs(d).mean().item()

        # Average absolute deviations
        dbar = sabsd / calibration_years

        print(f"    DEBUG CAFEC Monthly: Average absolute deviations (first 6 months): "
              f"{[f'{x:.2f}' for x in dbar[:6]]}")
        print(f"    DEBUG CAFEC Monthly: Average absolute deviations (last 6 months): "
              f"{[f'{x:.2f}' for x in dbar[-6:]]}")

        # Calculate preliminary K factors
        akhat = np.zeros(num_periods)
        for i in range(num_periods):
            if dbar[i] > 0:
                akhat[i] = (1.5 * np.log10((trat[i] + 2.8) / dbar[i]) + 0.5)
            else:
                akhat[i] = 1.0
        
        print(f"    DEBUG CAFEC Monthly: Preliminary monthly K factors (first 6): "
              f"{[f'{x:.3f}' for x in akhat[:6]]}")
        print(f"    DEBUG CAFEC Monthly: Preliminary monthly K factors (last 6): "
              f"{[f'{x:.3f}' for x in akhat[-6:]]}")
        
        # Normalize K factors
        swtd = np.sum(dbar * akhat)
        k_factors = 17.67 * akhat / swtd if swtd > 0 else np.ones(num_periods)
        
        print(f"    DEBUG CAFEC Monthly: Final normalized monthly K-factors (first 6): "
              f"{[f'{x:.3f}' for x in k_factors[:6]]}")
        print(f"    DEBUG CAFEC Monthly: Final normalized monthly K-factors (last 6): "
              f"{[f'{x:.3f}' for x in k_factors[-6:]]}")
        print(f"    DEBUG CAFEC Monthly: K factor sum: {np.sum(k_factors):.3f}, "
              f"normalization factor: {17.67/swtd:.3f}")

        return k_factors

    def _calculate_k_factors(self,
                            weekly_sums: Dict[str, np.ndarray],
                            cafec_coeffs: Dict[str, np.ndarray],
                            calibration_data: Dict[str, xr.DataArray],
                            calibration_years: int) -> np.ndarray:
        """
        Calculate K factors (weekly weighting factors) for Z-index calculation
        
        Args:
            weekly_sums: Dictionary containing weekly sum arrays
            cafec_coeffs: Dictionary containing CAFEC coefficients
            calibration_data: Dictionary containing calibration period data
            calibration_years: Number of years in calibration period
        
        Returns:
            Array of 53 K factors
        """
        print(f"   DEBUG CAFEC: Starting K factor calculation for "
              f"{calibration_years} years")
        
        # Average the sums over calibration period
        avg_sums = {}
        for key, values in weekly_sums.items():
            avg_sums[key] = values / calibration_years
        
        # Calculate T ratio for each week
        trat = np.zeros(53)
        for week in range(53):
            numerator = (avg_sums['petsum'][week] + avg_sums['rsum'][week] + 
                        avg_sums['rosum'][week])
            denominator = avg_sums['psum'][week] + avg_sums['tlsum'][week]
            if denominator != 0:
                trat[week] = numerator / denominator
            else:
                trat[week] = 1.0
        
        print(f"   DEBUG CAFEC: Weekly T-ratios (first 5): "
              f"{[f'{x:.3f}' for x in trat[:5]]}")
        print(f"   DEBUG CAFEC: Weekly T-ratios (last 5): "
              f"{[f'{x:.3f}' for x in trat[-5:]]}")
        
        # Calculate absolute deviations for each week during calibration period
        sabsd = np.zeros(53)
        
        # Load and compute the calibration data
        data_vars = ['prcp', 'pet', 'pr', 'sp', 'pl']
        loaded_data = {}
        for var in data_vars:
            cal_data = calibration_data[var]
            loaded_data[var] = (cal_data.compute() if hasattr(cal_data, 'chunks') 
                               else cal_data)
        
        P_cal, PET_cal, PR_cal, SP_cal, PL_cal = [loaded_data[var] 
                                                 for var in data_vars]
        
        print(f"   DEBUG CAFEC: Calculating absolute deviations for "
              f"{P_cal.sizes['time']} time steps...")
        
        # Calculate deviations for each time step
        for t_idx in range(P_cal.sizes['time']):
            # Get week index (0-52) - robust time handling
            time_val = P_cal.time.data[t_idx]
            
            # Enhanced robustness check for time_val type
            try:
                if isinstance(time_val, tuple):
                    # If time_val is a tuple, try to extract the first element
                    if len(time_val) > 0:
                        time_val = time_val[0]
                    else:
                        # Fallback to using the time coordinate directly
                        time_val = P_cal.time.values[t_idx]
                
                # Convert to pandas Timestamp for reliable week extraction
                if hasattr(time_val, 'isocalendar'):
                    week = time_val.isocalendar().week - 1
                else:
                    # Handle numpy datetime64 or other time formats
                    try:
                        week = pd.Timestamp(time_val).isocalendar().week - 1
                    except (TypeError, ValueError):
                        # Final fallback - use pandas time index directly
                        time_pd = pd.Timestamp(P_cal.time.values[t_idx])
                        week = time_pd.isocalendar().week - 1
                        
            except Exception as e_time:
                # print(f"   Warning: Issue processing time value at index "
                #      f("{t_idx}: {e_time}")
                # print(f"   time_val type: {type(time_val)}, "
                #      f("value: {time_val}")
                # Use a simple modulo approach as absolute fallback
                week = t_idx % 53
            
            # Ensure week index is within valid range (0-52)
            week = max(0, min(52, week))
            
            # Calculate P_hat for this time step
            p_hat = (cafec_coeffs['alpha'][week] * PET_cal.isel(time=t_idx) +
                     cafec_coeffs['beta'][week] * PR_cal.isel(time=t_idx) +
                     cafec_coeffs['gamma'][week] * SP_cal.isel(time=t_idx) -
                     cafec_coeffs['delta'][week] * PL_cal.isel(time=t_idx))
            
            # Calculate deviation (d = P - P_hat)
            d = P_cal.isel(time=t_idx) - p_hat
            
            # Accumulate absolute deviation for this week
            sabsd[week] += np.abs(d).mean().item()
        
        # Average absolute deviations
        dbar = sabsd / calibration_years
        
        print(f"   DEBUG CAFEC: Average absolute deviations (first 5 weeks): "
              f"{[f'{x:.2f}' for x in dbar[:5]]}")
        print(f"   DEBUG CAFEC: Average absolute deviations (last 5 weeks): "
              f"{[f'{x:.2f}' for x in dbar[-5:]]}")
        
        # Calculate preliminary K factors
        akhat = np.zeros(53)
        for week in range(53):
            if dbar[week] > 0:
                akhat[week] = (1.5 * np.log10((trat[week] + 2.8) / 
                              dbar[week]) + 0.5)
            else:
                akhat[week] = 1.0
        
        print(f"   DEBUG CAFEC: Preliminary weekly K factors (first 5): "
              f"{[f'{x:.3f}' for x in akhat[:5]]}")
        print(f"   DEBUG CAFEC: Preliminary weekly K factors (last 5): "
              f"{[f'{x:.3f}' for x in akhat[-5:]]}")
        
        # Normalize K factors  
        swtd = np.sum(dbar * akhat)
        if swtd > 0:
            k_factors = 17.67 * akhat / swtd
        else:
            k_factors = np.ones(53)
        
        print(f"   DEBUG CAFEC: Final normalized weekly K-factors (first 5): "
              f"{[f'{x:.3f}' for x in k_factors[:5]]}")
        print(f"   DEBUG CAFEC: Final normalized weekly K-factors (last 5): "
              f"{[f'{x:.3f}' for x in k_factors[-5:]]}")
        print(f"   DEBUG CAFEC: K factor sum: {np.sum(k_factors):.3f}, "
              f"normalization factor: {17.67/swtd:.3f}")
        
        return k_factors

    def _accumulate_monthly_data_groupby(self,
                                        P_monthly: xr.DataArray,
                                        PET_monthly: xr.DataArray, 
                                        ET_actual: xr.DataArray,
                                        R_total: xr.DataArray,
                                        L_total: xr.DataArray,
                                        RO_monthly: xr.DataArray,
                                        S_s_series: xr.DataArray,
                                        S_u_series: xr.DataArray,
                                        awc_s_static: xr.DataArray,
                                        awc_u_static: xr.DataArray,
                                        calibration_end_date: str = '2015-01-01') -> Tuple[Dict[str, np.ndarray], Dict[str, xr.DataArray], int]:
        """
        Accumulate weekly data for calibration period using groupby method (memory efficient)
        
        Args:
            Various input DataArrays from water balance calculations
            calibration_end_date: End date for calibration period (exclusive)
        
        Returns:
            Tuple of (weekly_sums, calibration_data, calibration_years)
        """
        print(f"   DEBUG CAFEC: Starting weekly data accumulation "
              f"(groupby method) with calibration_end_date={calibration_end_date}")
        
        # Ensure time coordinate is datetime64 for stable processing
        P_monthly['time'] = P_monthly.time.astype('datetime64[ns]')
        
        # Filter data for calibration period (before 2018-01-01)
        cal_mask = P_monthly.time < pd.Timestamp(calibration_end_date)
        cal_count = cal_mask.sum().item()
        total_count = len(P_monthly.time)
        print(f"   DEBUG CAFEC: Found {cal_count}/{total_count} calibration "
              f"time steps before {calibration_end_date}")
        
        if not cal_mask.any():
            raise ValueError(f"No calibration data found before "
                           f"{calibration_end_date}")
        
        # Extract calibration period data
        P_cal = P_monthly.where(cal_mask, drop=True)
        PET_cal = PET_monthly.where(cal_mask, drop=True)
        ET_cal = ET_actual.where(cal_mask, drop=True)
        R_cal = R_total.where(cal_mask, drop=True)
        L_cal = L_total.where(cal_mask, drop=True) 
        RO_cal = RO_monthly.where(cal_mask, drop=True)
        Ss_cal = S_s_series.where(cal_mask, drop=True)
        Su_cal = S_u_series.where(cal_mask, drop=True)
        
        print(f"   DEBUG CAFEC: Calibration data extracted - P_cal shape: "
              f"{P_cal.shape}, time range: "
              f"{P_cal.time.dt.strftime('%Y-%m').data[0]} to "
              f"{P_cal.time.dt.strftime('%Y-%m').data[-1]}")
        
        # Calculate derived variables for calibration period
        AWC_total = awc_s_static + awc_u_static
        SP_cal = Ss_cal + Su_cal  # Soil moisture at beginning of period
        PR_cal = AWC_total - SP_cal  # Potential recharge
        
        # Calculate potential loss using the two-layer logic
        PL_cal = xr.where(Ss_cal >= PET_cal, 
                         PET_cal,
                         np.minimum(Ss_cal + Su_cal, 
                                   ((PET_cal - Ss_cal) * Su_cal) / 
                                   (awc_u_static + 1.0) + Ss_cal))
        
        print(f"   DEBUG CAFEC: Calculated derived variables - SP_cal mean: "
              f"{SP_cal.mean().item():.2f}, PR_cal mean: "
              f"{PR_cal.mean().item():.2f}, PL_cal mean: "
              f"{PL_cal.mean().item():.2f}")
        
        # Calculate number of calibration years
        start_year = P_cal.time.dt.year.min().item()
        end_year = P_cal.time.dt.year.max().item()
        calibration_years = end_year - start_year + 1
        
        print(f"   DEBUG CAFEC: Calibration period spans {calibration_years} "
              f"years ({start_year}-{end_year})")
        
        try:
            # Use groupby to accumulate weekly sums (memory efficient)
            print("   DEBUG CAFEC: Using groupby method for weekly accumulation...")
            weekly_sums = {}
            
            # Group by ISO calendar week and sum spatially averaged values
            week_group = P_cal.time.dt.isocalendar().week
            weekly_sums['psum'] = P_cal.groupby(week_group).sum().mean(dim=['x', 'y']).values
            weekly_sums['petsum'] = PET_cal.groupby(PET_cal.time.dt.isocalendar().week).sum().mean(dim=['x', 'y']).values
            weekly_sums['etsum'] = ET_cal.groupby(ET_cal.time.dt.isocalendar().week).sum().mean(dim=['x', 'y']).values
            weekly_sums['rsum'] = R_cal.groupby(R_cal.time.dt.isocalendar().week).sum().mean(dim=['x', 'y']).values
            weekly_sums['tlsum'] = L_cal.groupby(L_cal.time.dt.isocalendar().week).sum().mean(dim=['x', 'y']).values
            weekly_sums['rosum'] = RO_cal.groupby(RO_cal.time.dt.isocalendar().week).sum().mean(dim=['x', 'y']).values
            weekly_sums['spsum'] = SP_cal.groupby(SP_cal.time.dt.isocalendar().week).sum().mean(dim=['x', 'y']).values
            weekly_sums['prsum'] = PR_cal.groupby(PR_cal.time.dt.isocalendar().week).sum().mean(dim=['x', 'y']).values
            weekly_sums['plsum'] = PL_cal.groupby(PL_cal.time.dt.isocalendar().week).sum().mean(dim=['x', 'y']).values
            
            print("   DEBUG CAFEC: Weekly sums calculated successfully via groupby")
            print(f"  [DEBUG] Validation: Aggregation resulted in "
                  f"{len(weekly_sums['psum'])} weekly periods.")
            print("   DEBUG CAFEC: Sample weekly sums (groupby method):")
            print(f"   DEBUG CAFEC:   psum (first 5 weeks): "
                  f"{[f'{x:.1f}' for x in weekly_sums['psum'][:5]]}")
            print(f"   DEBUG CAFEC:   petsum (first 5 weeks): "
                  f"{[f'{x:.1f}' for x in weekly_sums['petsum'][:5]]}")
            
            # Handle case where some weeks might be missing
            for key in weekly_sums:
                if len(weekly_sums[key]) < 53:
                    print(f"   DEBUG CAFEC: Filling missing weeks for {key} "
                          f"(found {len(weekly_sums[key])}/53 weeks)")
                    full_array = np.zeros(53)
                    available_weeks = P_cal.groupby(P_cal.time.dt.isocalendar().week).groups.keys()
                    for i, week in enumerate(available_weeks):
                        if week <= 53:  # ISO weeks are 1-53
                            full_array[week-1] = weekly_sums[key][i]
                    weekly_sums[key] = full_array
                elif len(weekly_sums[key]) == 53:
                    weekly_sums[key] = weekly_sums[key]
                else:
                    # Truncate if more than 53 weeks
                    weekly_sums[key] = weekly_sums[key][:53]
                    
        except Exception as e:
            print(f"   Warning: groupby method failed ({e}), "
                  "falling back to load method")
            return self._accumulate_monthly_data_load(
                P_monthly, PET_monthly, ET_actual, R_total, L_total, 
                RO_monthly, S_s_series, S_u_series, awc_s_static, 
                awc_u_static, calibration_end_date
            )
        
        # Store calibration data for K factor calculation
        calibration_data = {
            'prcp': P_cal,
            'pet': PET_cal, 
            'pr': PR_cal,
            'sp': SP_cal,
            'pl': PL_cal
        }
        
        print(f"   DEBUG CAFEC: Returning weekly_sums, calibration_data, "
              f"and calibration_years={calibration_years} (groupby method)")
        
        return weekly_sums, calibration_data, calibration_years

    def _accumulate_monthly_calibration_data(self,
                                            P_monthly: xr.DataArray,
                                            PET_monthly: xr.DataArray,
                                            ET_actual_monthly: xr.DataArray,
                                            R_total_monthly: xr.DataArray,
                                            L_total_monthly: xr.DataArray,
                                            PL_monthly: xr.DataArray,   
                                            PR_monthly: xr.DataArray,
                                            RO_monthly: xr.DataArray,
                                            S_s_series_monthly: xr.DataArray,
                                            S_u_series_monthly: xr.DataArray,
                                            awc_s_static: xr.DataArray,
                                            awc_u_static: xr.DataArray,
                                            calibration_end_date: str = '2015-01-01') -> Tuple[Dict[str, np.ndarray], Dict[str, xr.DataArray], int]:
        """
        Accumulate monthly data for calibration period to compute monthly sums for standard Palmer calibration.
        This function extracts and processes the calibration period data needed for monthly coefficient calculation.
        
        Args:
            P_monthly, PET_monthly: Monthly precipitation and PET data
            ET_actual_monthly, R_total_monthly, L_total_monthly, RO_monthly: Monthly water balance components
            S_s_series_monthly, S_u_series_monthly: Monthly soil moisture states
            awc_s_static, awc_u_static: Static available water capacity layers
            calibration_end_date: End date for calibration period (exclusive)
            
        Returns:
            Tuple of (monthly_sums, calibration_data, calibration_years)
        """
        print(f"    DEBUG Monthly Calibration: Starting monthly calibration data accumulation "
              f"with calibration_end_date={calibration_end_date}")
        
        # Ensure time coordinate is datetime64 for stable processing
        P_monthly['time'] = P_monthly.time.astype('datetime64[ns]')
        
        # Filter data for calibration period
        cal_mask = P_monthly.time < pd.Timestamp(calibration_end_date)
        cal_count = cal_mask.sum().item()
        total_count = len(P_monthly.time)
        print(f"    DEBUG Monthly Calibration: Found {cal_count}/{total_count} calibration "
              f"time steps before {calibration_end_date}")
        
        if not cal_mask.any():
            raise ValueError(f"No calibration data found before {calibration_end_date}")
        
        # Extract calibration period data
        P_cal = P_monthly.where(cal_mask, drop=True)
        PET_cal = PET_monthly.where(cal_mask, drop=True)
        ET_cal = ET_actual_monthly.where(cal_mask, drop=True)
        R_cal = R_total_monthly.where(cal_mask, drop=True)
        L_cal = L_total_monthly.where(cal_mask, drop=True)
        RO_cal = RO_monthly.where(cal_mask, drop=True)
        PL_cal = PL_monthly.where(cal_mask, drop=True)  
        PR_cal = PR_monthly.where(cal_mask, drop=True)
        Ss_cal = S_s_series_monthly.where(cal_mask, drop=True)
        Su_cal = S_u_series_monthly.where(cal_mask, drop=True)
        
        print(f"    DEBUG Monthly Calibration: Calibration data extracted - P_cal shape: "
              f"{P_cal.shape}, time range: "
              f"{P_cal.time.dt.strftime('%Y-%m').data[0]} to "
              f"{P_cal.time.dt.strftime('%Y-%m').data[-1]}")
        
        # Calculate derived variables for calibration period
        AWC_total = awc_s_static + awc_u_static
        SP_cal = AWC_total - PR_cal  # Soil moisture at beginning of period
        # SP_cal = Ss_cal + Su_cal  # Soil moisture at beginning of period
        # PR_cal = AWC_total - SP_cal  # Potential recharge
        
        # Calculate potential loss using the two-layer logic
        # PL_cal = xr.where(Ss_cal >= PET_cal, 
        #                  PET_cal,
        #                  np.minimum(Ss_cal + Su_cal, 
        #                            ((PET_cal - Ss_cal) * Su_cal) / 
        #                            (awc_u_static + 1.0) + Ss_cal))
        
        print(f"    DEBUG Monthly Calibration: Calculated derived variables - SP_cal mean: "
              f"{SP_cal.mean().item():.2f}, PR_cal mean: "
              f"{PR_cal.mean().item():.2f}, PL_cal mean: "
              f"{PL_cal.mean().item():.2f}")
        
        # Calculate number of calibration years
        start_year = P_cal.time.dt.year.min().item()
        end_year = P_cal.time.dt.year.max().item()
        calibration_years = end_year - start_year + 1
        
        print(f"    DEBUG Monthly Calibration: Calibration period spans {calibration_years} "
              f"years ({start_year}-{end_year})")
        
        # Group by month (1-12) and calculate monthly sums
        month_group = P_cal.time.dt.month
        
        # Calculate monthly sums using groupby and reindex to ensure all 12 months are present
        # METHODOLOGY NOTE: Regional Calibration with Pixel-Level Application
        # -------------------------------------------------------------------------
        # In this implementation, CAFEC coefficients (alpha, beta, gamma, delta) and 
        # K-factors are derived from spatially aggregated sums over the entire 
        # 12x12 km cube (Regional Calibration), rather than calculated per-pixel.
        #
        # JUSTIFICATION:
        # 1. Numerical Stability: Spatial aggregation prevents mathematical singularities 
        #    (e.g., division by zero) that may occur at the single-pixel level 
        #    due to local extremes (e.g., zero cumulative precipitation/PET on water/rock).
        # 2. Climatological Consistency: It assumes a consistent climate regime within 
        #    the local window (~144 km²), acting as a physical regularization filter 
        #    to remove high-frequency noise while preserving valid local trends.
        #    We would not analyze that this pixel is at drought while the surrounding area is not.
        # 3. High-Resolution Output: These stable regional coefficients are then applied 
        #    to pixel-level inputs (P, PET), ensuring the final PDSI maintains the 
        #    native 1km spatial resolution.
        # More Information:
        # In the original Palmer (1965) methodology, Palmer said that "Some may wonder why areas have been chosen for study rather than points.
        # Of course point data could have been used, but for developmental purposes it was easier to deal with areal averages, thereby avoiding the extreme variability of point weather."
        # Therefore, the area he utilized usually contains 10-30 counties. In this modern adaptation, we use a 12x12 km area to balance local detail with stability.
        # -------------------------------------------------------------------------
        monthly_sums = {}
        monthly_sums['psum'] = P_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
        monthly_sums['petsum'] = PET_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
        monthly_sums['etsum'] = ET_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
        monthly_sums['rsum'] = R_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
        monthly_sums['tlsum'] = L_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
        monthly_sums['rosum'] = RO_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
        monthly_sums['spsum'] = SP_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
        monthly_sums['prsum'] = PR_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
        monthly_sums['plsum'] = PL_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
        # Here we do not use skipna=True because missing data should be treated as zero contribution in sums, and nan pixels are constant in all variables.
        # Even though the computed mean value will be smaller due to nan pixels, but this is not a problem for CAFEC calibration since all variables are affected equally.
        # E.g. ET / PET ratio remains valid even if both ET and PET are reduced proportionally due to nan pixels.
        
        print("    DEBUG Monthly Calibration: Monthly sums calculated successfully")
        print(f"    DEBUG Monthly Calibration: Sample monthly sums:")
        print(f"    DEBUG Monthly Calibration:   psum (12 months): "
              f"{[f'{x:.1f}' for x in monthly_sums['psum']]}")
        print(f"    DEBUG Monthly Calibration:   petsum (first 6 months): "
              f"{[f'{x:.1f}' for x in monthly_sums['petsum'][:6]]}")
        
        # Store calibration data for K factor calculation
        calibration_data = {
            'prcp': P_cal,
            'pet': PET_cal,
            'pr': PR_cal,
            'sp': SP_cal,
            'pl': PL_cal
        }
        
        print(f"    DEBUG Monthly Calibration: Returning monthly_sums, calibration_data, "
              f"and calibration_years={calibration_years}")
        
        return monthly_sums, calibration_data, calibration_years

    def _accumulate_monthly_data_load(self,
                                     P_monthly: xr.DataArray,
                                     PET_monthly: xr.DataArray,
                                     ET_actual: xr.DataArray,
                                     R_total: xr.DataArray,
                                     L_total: xr.DataArray,
                                     RO_monthly: xr.DataArray,
                                     S_s_series: xr.DataArray,
                                     S_u_series: xr.DataArray,
                                     awc_s_static: xr.DataArray,
                                     awc_u_static: xr.DataArray,
                                     calibration_end_date: str = '2018-01-01') -> Tuple[Dict[str, np.ndarray], Dict[str, xr.DataArray], int]:
        """
        Accumulate weekly data for calibration period using load method (fallback)
        This method loads calibration data into memory and performs weekly accumulation.
        Used as fallback when groupby method fails due to complex dependencies.
        """
        print(f"   DEBUG CAFEC: Starting weekly data accumulation "
              f"(load method) with calibration_end_date={calibration_end_date}")
        
        # Ensure time coordinate is datetime64 for stable processing
        P_monthly['time'] = P_monthly.time.astype('datetime64[ns]')
        
        # Filter data for calibration period
        cal_mask = P_monthly.time < pd.Timestamp(calibration_end_date)
        cal_count = cal_mask.sum().item()
        total_count = len(P_monthly.time)
        print(f"   DEBUG CAFEC: Found {cal_count}/{total_count} calibration "
              f"time steps before {calibration_end_date}")
        
        if not cal_mask.any():
            raise ValueError(f"No calibration data found before "
                           f"{calibration_end_date}")
        
        # Extract and load calibration period data into memory
        print("   DEBUG CAFEC: Loading calibration data into memory...")
        P_cal = P_monthly.where(cal_mask, drop=True).compute()
        PET_cal = PET_monthly.where(cal_mask, drop=True).compute()
        ET_cal = ET_actual.where(cal_mask, drop=True).compute()
        R_cal = R_total.where(cal_mask, drop=True).compute()
        L_cal = L_total.where(cal_mask, drop=True).compute()
        RO_cal = RO_monthly.where(cal_mask, drop=True).compute()
        Ss_cal = S_s_series.where(cal_mask, drop=True).compute()
        Su_cal = S_u_series.where(cal_mask, drop=True).compute()
        
        print(f"   DEBUG CAFEC: Calibration data loaded - P_cal shape: "
              f"{P_cal.shape}, time range: "
              f"{P_cal.time.dt.strftime('%Y-%m').data[0]} to "
              f"{P_cal.time.dt.strftime('%Y-%m').data[-1]}")
        
        # Calculate derived variables
        AWC_total = awc_s_static + awc_u_static
        SP_cal = Ss_cal + Su_cal
        PR_cal = AWC_total - SP_cal
        
        print(f"   DEBUG CAFEC: Calculated derived variables - "
              f"AWC_total mean: {AWC_total.mean().item():.2f}, "
              f"SP_cal mean: {SP_cal.mean().item():.2f}, "
              f"PR_cal mean: {PR_cal.mean().item():.2f}")
        
        # Calculate potential loss
        PL_cal = xr.where(Ss_cal >= PET_cal,
                         PET_cal,
                         np.minimum(Ss_cal + Su_cal, 
                                   ((PET_cal - Ss_cal) * Su_cal) / 
                                   (awc_u_static + 1.0) + Ss_cal))
        
        print(f"   DEBUG CAFEC: Calculated PL_cal - mean: "
              f"{PL_cal.mean().item():.2f}, min: {PL_cal.min().item():.2f}, "
              f"max: {PL_cal.max().item():.2f}")
        
        # Initialize weekly sum arrays
        weekly_sums = {key: np.zeros(53) for key in 
                       ['psum', 'petsum', 'etsum', 'rsum', 'tlsum', 'rosum', 
                        'spsum', 'prsum', 'plsum']}
        
        # Calculate number of calibration years
        start_year = P_cal.time.dt.year.min().item()
        end_year = P_cal.time.dt.year.max().item()
        calibration_years = end_year - start_year + 1
        
        print(f"   DEBUG CAFEC: Calibration period spans {calibration_years} "
              f"years ({start_year}-{end_year})")
        
        # Accumulate data by looping through time steps
        print(f"   DEBUG CAFEC: Starting weekly accumulation for "
              f"{P_cal.sizes['time']} time steps...")
        for t_idx in range(P_cal.sizes['time']):
            # Get week index (0-52)
            time_val = P_cal.time.data[t_idx]
            if hasattr(time_val, 'isocalendar'):
                week_idx = time_val.isocalendar().week - 1
            else:
                week_idx = pd.Timestamp(time_val).isocalendar().week - 1
            
            # Ensure week index is within valid range (0-52)
            week_idx = max(0, min(52, week_idx))
            
            # Accumulate spatially averaged values for this week
            weekly_sums['psum'][week_idx] += P_cal.isel(time=t_idx).mean().item()
            weekly_sums['petsum'][week_idx] += PET_cal.isel(time=t_idx).mean().item()
            weekly_sums['etsum'][week_idx] += ET_cal.isel(time=t_idx).mean().item()
            weekly_sums['rsum'][week_idx] += R_cal.isel(time=t_idx).mean().item()
            weekly_sums['tlsum'][week_idx] += L_cal.isel(time=t_idx).mean().item()
            weekly_sums['rosum'][week_idx] += RO_cal.isel(time=t_idx).mean().item()
            weekly_sums['spsum'][week_idx] += SP_cal.isel(time=t_idx).mean().item()
            weekly_sums['prsum'][week_idx] += PR_cal.isel(time=t_idx).mean().item()
            weekly_sums['plsum'][week_idx] += PL_cal.isel(time=t_idx).mean().item()
        
        print("   DEBUG CAFEC: Weekly accumulation completed. Sample weekly sums:")
        print(f"   [DEBUG] Validation: Aggregation resulted in "
              f"{len([x for x in weekly_sums['psum'] if x > 0])} "
              f"non-zero weekly periods.")
        print(f"   DEBUG CAFEC:   psum (first 5 weeks): "
              f"{[f'{x:.1f}' for x in weekly_sums['psum'][:5]]}")
        print(f"   DEBUG CAFEC:   petsum (first 5 weeks): "
              f"{[f'{x:.1f}' for x in weekly_sums['petsum'][:5]]}")
        print(f"   DEBUG CAFEC:   etsum (first 5 weeks): "
              f"{[f'{x:.1f}' for x in weekly_sums['etsum'][:5]]}")
        
        # Store calibration data for K factor calculation
        calibration_data = {
            'prcp': P_cal,
            'pet': PET_cal,
            'pr': PR_cal,
            'sp': SP_cal,
            'pl': PL_cal
        }
        
        print(f"   DEBUG CAFEC: Returning weekly_sums, calibration_data, "
              f"and calibration_years={calibration_years}")
        
        return weekly_sums, calibration_data, calibration_years

    def _get_standard_monthly_calibration_coeffs(self,
                                                P_da: xr.DataArray,
                                                PET_da: xr.DataArray,
                                                awc_s_static_2d: xr.DataArray,
                                                awc_u_static_2d: xr.DataArray,
                                                s_name_for_log: str,
                                                c_idx_abs: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Factory function to perform complete standard monthly Palmer self-calibration workflow.
        This function executes the complete monthly calibration process and returns 12 base climate coefficients.
        NO hardcoded coefficient values are allowed in this function.
        
        Args:
            P_da: Precipitation DataArray (can be weekly or monthly)
            PET_da: Potential evapotranspiration DataArray
            awc_s_static_2d: Static available water capacity for surface layer
            awc_u_static_2d: Static available water capacity for upper layer
            s_name_for_log: Segment name for logging
            c_idx_abs: Cube index for logging
            
        Returns:
            Tuple of (cafec_coeffs, k_factors) where:
            - cafec_coeffs: Dict with 12-element arrays for alpha, beta, gamma, delta
            - k_factors: Array of 12 K-factor values
        """
        print(f"    DEBUG Standard Monthly Calibration: Starting complete monthly calibration "
              f"workflow for cube {c_idx_abs}, segment {s_name_for_log}")
        
        # Step 1: Data frequency processing
        print("    DEBUG Standard Monthly Calibration: Step 1 - Data frequency processing")
        
        # Check if data is weekly (time interval < 15 days) and resample to monthly if needed
        time_diff = (P_da.time[1] - P_da.time[0]).values
        time_diff_days = pd.Timedelta(time_diff).days
        
        if time_diff_days < 15:
            print(f"    DEBUG Standard Monthly Calibration: Detected weekly data (interval: {time_diff_days} days), "
                  "resampling to monthly")
            P_monthly = P_da.resample(time='1M').sum()
            PET_monthly = PET_da.resample(time='1M').sum()
        else:
            print(f"    DEBUG Standard Monthly Calibration: Data already monthly (interval: {time_diff_days} days)")
            P_monthly = P_da
            PET_monthly = PET_da
        
        print(f"    DEBUG Standard Monthly Calibration: Monthly data shape: {P_monthly.shape}, "
              f"time range: {P_monthly.time.dt.strftime('%Y-%m').data[0]} to "
              f"{P_monthly.time.dt.strftime('%Y-%m').data[-1]}")
        
        # Step A: Run monthly water balance model
        print("    DEBUG Standard Monthly Calibration: Step A - Running monthly water balance model")
        water_balance_results = self._perform_soil_moisture_accounting_two_layer(
            P_monthly, PET_monthly, awc_s_static_2d, awc_u_static_2d, s_name_for_log, c_idx_abs
        )
        
        if any(result is None for result in water_balance_results):
            print(f"    ERROR Standard Monthly Calibration: Water balance model failed for cube {c_idx_abs}")
            return None, None
        
        ET_actual_monthly, R_total_monthly, L_total_monthly, RO_monthly, S_s_series_monthly, S_u_series_monthly, PL_series_monthly, PR_series_monthly = water_balance_results
        
        print(f"    DEBUG Standard Monthly Calibration: Water balance completed - "
              f"ET mean: {ET_actual_monthly.mean().item():.2f}, "
              f"R mean: {R_total_monthly.mean().item():.2f}, "
              f"RO mean: {RO_monthly.mean().item():.2f}")
        
        # Step B: Prepare monthly calibration data
        print("    DEBUG Standard Monthly Calibration: Step B - Preparing monthly calibration data")
        monthly_sums, calibration_data, calibration_years = self._accumulate_monthly_calibration_data(
            P_monthly, PET_monthly, ET_actual_monthly, R_total_monthly, L_total_monthly,PL_series_monthly, PR_series_monthly,
            RO_monthly, S_s_series_monthly, S_u_series_monthly, awc_s_static_2d, awc_u_static_2d,
            calibration_end_date=self.calibration_end_date
        )
        
        print(f"    DEBUG Standard Monthly Calibration: Calibration data prepared for {calibration_years} years")
        
        # Step C: Calculate 12 base coefficients
        print("    DEBUG Standard Monthly Calibration: Step C - Calculating 12 base coefficients")
        
        # Calculate CAFEC coefficients
        print("    DEBUG Standard Monthly Calibration: Step C1 - Calculating CAFEC coefficients")
        cafec_coeffs = self._calculate_cafec_coefficients_monthly(monthly_sums, calibration_years)
        
        # Calculate K factors
        print("    DEBUG Standard Monthly Calibration: Step C2 - Calculating K factors")
        k_factors = self._calculate_k_factors_monthly(monthly_sums, cafec_coeffs, calibration_data, calibration_years)
        
        # Step D: Return final results
        print(f"    DEBUG Standard Monthly Calibration: Step D - Returning final results")
        print(f"    DEBUG Standard Monthly Calibration: CAFEC coeffs summary - "
              f"alpha: {cafec_coeffs['alpha'][:3]} (first 3), "
              f"beta: {cafec_coeffs['beta'][:3]} (first 3)")
        print(f"    DEBUG Standard Monthly Calibration: K factors summary - "
              f"mean: {k_factors.mean():.3f}, sum: {k_factors.sum():.3f}")
        
        return cafec_coeffs, k_factors

    def _calculate_z_index_terragon_style(self,
                                        P_monthly: xr.DataArray,
                                        ET_actual_monthly: xr.DataArray,
                                        RO_monthly: xr.DataArray,
                                        s_name_for_log: str, c_idx_abs: int
                                        ) -> Optional[xr.DataArray]:
        """
        Calculates Z-index by mimicking the Terragon methodology.
        1. Resample data into ~30-day blocks.
        2. Calculate moisture departure 'D' for each block.
        3. Standardize 'D' over a calibration period to get a monthly-like Z-index.
        4. Upsample the Z-index back to the original time frequency.
        5. Scale the Z-index by dividing by 6.
        """
        try:
            print(f"  DEBUG Terragon Style: Starting Z-index calculation for cube {c_idx_abs}")
            resample_period = '30D'
            P_30day = P_monthly.resample(time=resample_period).sum()
            ET_30day = ET_actual_monthly.resample(time=resample_period).sum()
            RO_30day = RO_monthly.resample(time=resample_period).sum()
            
            D_30day = P_30day - ET_30day - RO_30day
            mean_D_30day = D_30day.mean(dim='time', skipna=True)
            std_D_30day = D_30day.std(dim='time', skipna=True)

            std_for_division = std_D_30day.clip(min=1e-9)
            Z_30day = xr.where(std_D_30day == 0, 0.0, (D_30day - mean_D_30day) / std_for_division)
            print(f"  DEBUG Terragon Style: Calculated 30-day Z-index. Mean={Z_30day.mean().item():.2f}, Std={Z_30day.std().item():.2f}")

            Z_upsampled = Z_30day.reindex_like(P_monthly, method='backfill')
            
            Z_scaled = Z_upsampled / 4.3
            print(f"  DEBUG Terragon Style: Scaled Z-index by /6. New Mean={Z_scaled.mean().item():.2f}")
            
            Z_scaled.name = 'z_index_terragon_style'
            Z_scaled.attrs = {
                'units': 'unitless',
                'long_name': 'Palmer Z-Index (Terragon Emulation)',
                'description': 'Z-Index calculated using 30-day blocks, standardized, and scaled by 1/6.'
            }
            
            return Z_scaled

        except Exception as e:
            print(f"  ERROR during Terragon-style Z-index calculation for cube {c_idx_abs}: {e}")
            traceback.print_exc()
            return None

    def _calculate_z_index_cafec_monthly(self,
                                         P_monthly: xr.DataArray,
                                         PET_monthly: xr.DataArray,
                                         ET_actual: xr.DataArray,
                                         R_total: xr.DataArray,
                                         L_total: xr.DataArray,
                                         RO_monthly: xr.DataArray,
                                         S_s_series: xr.DataArray,
                                         S_u_series: xr.DataArray,
                                         awc_s_static: xr.DataArray,
                                         awc_u_static: xr.DataArray,
                                         s_name_for_log: str,
                                         c_idx_abs: int) -> Optional[xr.DataArray]:
        """
        Calculates a MONTHLY Z-index using the CAFEC method.
        This function assumes all inputs are already at a monthly resolution.
        """
        try:
            # Step 1: Accumulate monthly data for calibration period using 'time.month'
            print(f"  DEBUG CAFEC Monthly: Starting monthly CAFEC Z-index calculation "
                  f"for cube {c_idx_abs}")
            print("  DEBUG CAFEC Monthly: Starting monthly data accumulation...")
            
            # --- This section is adapted from _accumulate_monthly_data_groupby ---
            P_monthly['time'] = P_monthly.time.astype('datetime64[ns]')
            calibration_end_date = '2018-01-01'
            cal_mask = P_monthly.time < pd.Timestamp(calibration_end_date)
            
            if not cal_mask.any():
                raise ValueError(f"No calibration data found before {calibration_end_date}")
            
            cal_count = cal_mask.sum().item()
            total_count = len(P_monthly.time)
            print(f"  DEBUG CAFEC Monthly: Found {cal_count}/{total_count} calibration "
                  f"time steps before {calibration_end_date}")
            
            # Extract calibration data
            P_cal = P_monthly.where(cal_mask, drop=True)
            PET_cal = PET_monthly.where(cal_mask, drop=True)
            ET_cal = ET_actual.where(cal_mask, drop=True)
            R_cal = R_total.where(cal_mask, drop=True)
            L_cal = L_total.where(cal_mask, drop=True)
            RO_cal = RO_monthly.where(cal_mask, drop=True)
            Ss_cal = S_s_series.where(cal_mask, drop=True)
            Su_cal = S_u_series.where(cal_mask, drop=True)

            start_year = P_cal.time.dt.year.min().item()
            end_year = P_cal.time.dt.year.max().item()
            calibration_years = end_year - start_year + 1
            print(f"  DEBUG CAFEC Monthly: Calibration period: {start_year}-{end_year} "
                  f"({calibration_years} years)")

            # Calculate derived variables for calibration
            AWC_total = awc_s_static + awc_u_static
            SP_cal = Ss_cal + Su_cal
            PR_cal = AWC_total - SP_cal
            PL_cal = xr.where(Ss_cal >= PET_cal, PET_cal,
                              np.minimum(SP_cal, ((PET_cal - Ss_cal) * Su_cal) / (awc_u_static + 1.0) + Ss_cal))
            
            # --- This section is adapted from the groupby logic ---
            monthly_sums = {}
            month_group = P_cal.time.dt.month
            
            monthly_sums['psum'] = P_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
            monthly_sums['petsum'] = PET_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
            monthly_sums['etsum'] = ET_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
            monthly_sums['rsum'] = R_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
            monthly_sums['tlsum'] = L_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
            monthly_sums['rosum'] = RO_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
            monthly_sums['spsum'] = SP_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
            monthly_sums['prsum'] = PR_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
            monthly_sums['plsum'] = PL_cal.groupby(month_group).sum().mean(dim=['x', 'y']).reindex({'month': range(1, 13)}, fill_value=0).values
            
            calibration_data = {'prcp': P_cal, 'pet': PET_cal, 'pr': PR_cal, 'sp': SP_cal, 'pl': PL_cal}

            # Step 2: Calculate MONTHLY CAFEC coefficients (12 values)
            print("  DEBUG CAFEC Monthly: Calculating CAFEC coefficients...")
            cafec_coeffs = self._calculate_cafec_coefficients_monthly(monthly_sums, calibration_years)
            
            # Step 3: Calculate MONTHLY K factors (12 values)
            print("  DEBUG CAFEC Monthly: Calculating K factors...")
            k_factors = self._calculate_k_factors_monthly(monthly_sums, cafec_coeffs, calibration_data, calibration_years)
            
            # Step 4: Calculate Z-index for the entire MONTHLY time series
            print(f"  DEBUG CAFEC Monthly: Calculating Z-index for entire "
                  f"time series ({P_monthly.sizes['time']} time steps)...")
            SP_full = S_s_series + S_u_series
            PR_full = (awc_s_static + awc_u_static) - SP_full
            PL_full = xr.where(S_s_series >= PET_monthly, PET_monthly,
                               np.minimum(SP_full, ((PET_monthly - S_s_series) * S_u_series) / (awc_u_static + 1.0) + S_s_series))
            
            # Apply monthly coefficients to calculate monthly Z-index
            month_indices = P_monthly.time.dt.month - 1 # Get month index (0-11)
            
            alpha_ts = xr.DataArray(cafec_coeffs['alpha'][month_indices], coords={'time': P_monthly.time}, dims=['time'])
            beta_ts = xr.DataArray(cafec_coeffs['beta'][month_indices], coords={'time': P_monthly.time}, dims=['time'])
            gamma_ts = xr.DataArray(cafec_coeffs['gamma'][month_indices], coords={'time': P_monthly.time}, dims=['time'])
            delta_ts = xr.DataArray(cafec_coeffs['delta'][month_indices], coords={'time': P_monthly.time}, dims=['time'])
            k_factors_ts = xr.DataArray(k_factors[month_indices], coords={'time': P_monthly.time}, dims=['time'])
            
            p_hat = (alpha_ts * PET_monthly +
                     beta_ts * PR_full +
                     gamma_ts * SP_full -
                     delta_ts * PL_full)
            
            d = P_monthly - p_hat
            Z_index = k_factors_ts * d
            
            # Print monthly Z-index statistics for debugging (sample)
            z_stats_by_month = {i: [] for i in range(12)}  # Track Z values by month for debugging
            
            for t_idx in range(P_monthly.sizes['time']):
                time_val = P_monthly.time.data[t_idx]
                month_idx = pd.Timestamp(time_val).month - 1  # 0-11 index
                z_val = Z_index.isel(time=t_idx)
                z_stats_by_month[month_idx].append(z_val.mean().item())
            
            # Print monthly Z-index statistics for debugging (sample)
            for month in range(0, 12, 3):  # Sample every 3rd month
                if z_stats_by_month[month]:
                    month_mean = np.mean(z_stats_by_month[month])
                    month_std = np.std(z_stats_by_month[month])
                    month_count = len(z_stats_by_month[month])
                    print(f"  DEBUG CAFEC Monthly: Month {month+1:2d} - "
                          f"{month_count:3d} values, mean Z: {month_mean:6.2f}, "
                          f"std: {month_std:6.2f}")
            
            print("  DEBUG CAFEC Monthly: Z-index calculation completed.")
            return Z_index

        except Exception as e_cafec:
            print(f"  Error during MONTHLY CAFEC Z-index calculation for cube {c_idx_abs}: {e_cafec}")
            traceback.print_exc()
            return None


    def _calculate_z_index_cafec(self,
                                P_monthly: xr.DataArray,
                                PET_monthly: xr.DataArray,
                                ET_actual: xr.DataArray,
                                R_total: xr.DataArray,
                                L_total: xr.DataArray,
                                RO_monthly: xr.DataArray,
                                S_s_series: xr.DataArray,
                                S_u_series: xr.DataArray,
                                awc_s_static: xr.DataArray,
                                awc_u_static: xr.DataArray,
                                s_name_for_log: str,
                                c_idx_abs: int) -> Optional[xr.DataArray]:
        """
        Calculate Z-index using classic CAFEC method
        
        Args:
            Various input DataArrays from water balance calculations
            s_name_for_log: Segment name for logging
            c_idx_abs: Cube index for logging
        
        Returns:
            Z-index DataArray or None if calculation fails
        """
        try:
            # Step 1: Accumulate weekly data for calibration period
            # print(f"   DEBUG CAFEC Z-index: Starting CAFEC Z-index calculation "
            #       f"for cube {c_idx_abs}")
            try:
                weekly_sums, calibration_data, calibration_years = self._accumulate_monthly_data_groupby(
                    P_monthly, PET_monthly, ET_actual, R_total, L_total, 
                    RO_monthly, S_s_series, S_u_series, awc_s_static, 
                    awc_u_static,
                    calibration_end_date=self.calibration_end_date
                )
                print("   DEBUG CAFEC Z-index: Successfully used groupby method "
                      "for weekly data accumulation")
            except Exception as e_groupby:
                print(f"   INFO CAFEC Z-index: groupby method failed for cube "
                      f"{c_idx_abs}, using load method: {e_groupby}")
                weekly_sums, calibration_data, calibration_years = self._accumulate_monthly_data_load(
                    P_monthly, PET_monthly, ET_actual, R_total, L_total, 
                    RO_monthly, S_s_series, S_u_series, awc_s_static, 
                    awc_u_static,
                    calibration_end_date=self.calibration_end_date
                )
                print("   DEBUG CAFEC Z-index: Successfully used load method "
                      "for weekly data accumulation")
            
            # Step 2: Calculate CAFEC coefficients
            print("   DEBUG CAFEC Z-index: Calculating CAFEC coefficients...")
            cafec_coeffs = self._calculate_cafec_coefficients(weekly_sums, 
                                                            calibration_years)
            
            # Step 3: Calculate K factors
            print("   DEBUG CAFEC Z-index: Calculating K factors...")
            k_factors = self._calculate_k_factors(weekly_sums, cafec_coeffs, 
                                                calibration_data, 
                                                calibration_years)
            
            # Step 4: Calculate Z-index for entire time series
            print(f"   DEBUG CAFEC Z-index: Calculating Z-index for entire "
                  f"time series ({P_monthly.sizes['time']} time steps)...")
            AWC_total = awc_s_static + awc_u_static
            SP_full = S_s_series + S_u_series
            PR_full = AWC_total - SP_full
            
            # Calculate potential loss for full time series
            PL_full = xr.where(S_s_series >= PET_monthly,
                              PET_monthly,
                              np.minimum(S_s_series + S_u_series,
                                        ((PET_monthly - S_s_series) * 
                                         S_u_series) / (awc_u_static + 1.0) + 
                                        S_s_series))
            
            # Calculate P_hat and Z-index for each time step
            Z_list = []
            z_stats_by_week = {i: [] for i in range(53)}  # Track Z values by week for debugging
            
            for t_idx in range(P_monthly.sizes['time']):
                # Get week index (0-52) - robust time handling
                time_val = P_monthly.time.data[t_idx]
                
                # Enhanced robustness check for time_val type
                try:
                    if isinstance(time_val, tuple):
                        # If time_val is a tuple, try to extract first element
                        if len(time_val) > 0:
                            time_val = time_val[0]
                        else:
                            # Fallback to using the time coordinate directly
                            time_val = P_monthly.time.values[t_idx]
                    
                    # Convert to pandas Timestamp for reliable week extraction
                    if hasattr(time_val, 'isocalendar'):
                        week_idx = time_val.isocalendar().week - 1
                    else:
                        # Handle numpy datetime64 or other time formats
                        try:
                            week_idx = (pd.Timestamp(time_val)
                                       .isocalendar().week - 1)
                        except (TypeError, ValueError):
                            # Final fallback - use pandas time index
                            time_pd = (pd.Timestamp(P_monthly.time
                                                   .values[t_idx]))
                            week_idx = time_pd.isocalendar().week - 1
                            
                except Exception as e_time:
                    #print(f"   Warning: Issue processing time value at index "
                    #      f"{t_idx}: {e_time}")
                    #print(f"   time_val type: {type(time_val)}, "
                    #      f"value: {time_val}")
                    # Use a simple modulo approach as absolute fallback
                    week_idx = t_idx % 53
                
                # Ensure week index is within valid range (0-52)
                week_idx = max(0, min(52, week_idx))
                
                # Add debug output for sampling
                if t_idx % 26 == 0:  # Approximately every half year
                    print(f"  [DEBUG] For time {time_val}, using weekly "
                          f"coefficient index: {week_idx}")
                
                # Calculate P_hat for this time step
                alpha_term = cafec_coeffs['alpha'][week_idx] * PET_monthly.isel(time=t_idx)
                beta_term = cafec_coeffs['beta'][week_idx] * PR_full.isel(time=t_idx)
                gamma_term = cafec_coeffs['gamma'][week_idx] * SP_full.isel(time=t_idx)
                delta_term = cafec_coeffs['delta'][week_idx] * PL_full.isel(time=t_idx)
                
                p_hat = alpha_term + beta_term + gamma_term - delta_term
                
                # Calculate moisture departure (d = P - P_hat)
                d = P_monthly.isel(time=t_idx) - p_hat
                
                # Calculate Z-index (Z = K * d)
                z_val = k_factors[week_idx] * d
                
                # Store Z value for weekly statistics
                z_stats_by_week[week_idx].append(z_val.mean().item())
                
                # Add time dimension back
                time_coord_val = P_monthly.time.data[t_idx]
                Z_list.append(z_val.expand_dims(time=[time_coord_val]))
            
            # Print weekly Z-index statistics for debugging (sample)
            for week in range(0, 53, 10):  # Sample every 10th week
                if z_stats_by_week[week]:
                    week_mean = np.mean(z_stats_by_week[week])
                    week_std = np.std(z_stats_by_week[week])
                    week_count = len(z_stats_by_week[week])
                    print(f"   DEBUG CAFEC Z-index: Week {week+1:2d} - "
                          f"{week_count:3d} values, mean Z: {week_mean:6.2f}, "
                          f"std: {week_std:6.2f}")
            
            # Concatenate all Z values
            if Z_list:
                Z_index = xr.concat(Z_list, dim='time')
                print(f"   DEBUG CAFEC Z-index: Z-index calculation completed. "
                      f"Final Z shape: {Z_index.shape}")
                print(f"   DEBUG CAFEC Z-index: Z-index statistics - "
                      f"mean: {Z_index.mean().item():.3f}, "
                      f"std: {Z_index.std().item():.3f}, "
                      f"min: {Z_index.min().item():.3f}, "
                      f"max: {Z_index.max().item():.3f}")
            else:
                print("   ERROR CAFEC Z-index: No Z values calculated, "
                      "returning None")
                return None
            
            # Set attributes
            Z_index.name = 'z_index_cafec'
            Z_index.attrs = {
                'units': 'unitless',
                'long_name': 'Palmer Z-Index (CAFEC Method)',
                'description': 'Z-Index calculated using classic CAFEC calibration method'
            }
            
            return Z_index
            
        except Exception as e_cafec:
            print(f"   Error during CAFEC Z-index calculation for cube "
                  f"{c_idx_abs}, seg {s_name_for_log}: {e_cafec}")
            traceback.print_exc()
            return None


