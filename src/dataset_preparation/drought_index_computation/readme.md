# Drought Index Calculation from Daymet
This is the work of the master's thesis "Exploring Temporal Dependencies and Forecasting Meteorological, Agricultural, and Hydrological Drought Indices Using Multi-Source Data" by Zhiyuan Zhang from the Technical University of Munich. The thesis presents additional drought indices that complement the original publication.

## Primary notice

Please ensure the path/name/structure of input/output location before using.

## Group 1: Data Preprocessing & Physics-Based PET Calculation

This module handles the sanitization of raw Daymet inputs and the calculation of Potential Evapotranspiration (PET) and Deficit (Prcp-PET).

### 1. Data Cleaning (`cleandaymet.py`)
**Objective:** To elliminate intermittent NaN in original daymet data.

* **Methodology:**
    * Applies **Temporal Linear Interpolation** to fix intermittent NaNs.
* **Output:** Generates the `daymet_cleaned` dataset, which serves as the immutable foundation for all downstream indices.

### 2. Deficit & PET Calculation (`main_Deficitandpet.py`)
**Objective:** To compute the Reference Evapotranspiration ($ET_0$) and Climatic Water Deficit ($P - PET$).

* **Physical Derivation (FAO-56 Standard):**
    This script manually adopts the **Modified Hargreaves Equation** and some equations from **FAO56** to ransfer physical variables and compute PET.
    Please refer to detail of the computation.

## Group 2: Temporal Aggregation & Multi-Scale Windowing

This module transforms the daily cleaned Daymet data into weekly data, creating datasets required for different drought indices.

### 1. Core Aggregator (`UnifiedAggregator_selected_trial.py`)
**Objective:** Aggregating daily data (pet/prcp/deficit) into weekly resolution according to rolling and standard mode.

* **Temporal Alignment Logic:**
    * **Frequency:** `1W` (Weekly frequency anchoring on Sundays).
    * **Interval Definition:** **Left-Open, Right-Closed** `(t-1, t]`.
    * **Timestamping:** Labeled with the closing date (Sunday), ensuring the timestamp represents the end of the preceding week.

### 2. Execution Modes (`main_unifiedaggregator_selected.py`)
The codes call the UnifiedAggregator_selected_trial to work in two distinct modes to serve different downstream physics:

* **Mode A: Standard Aggregation (Target: PDSI/PHDI)**
    * Computes discrete weekly variables (Precipitation Sum, PET Sum).

* **Mode B: Rolling Aggregation (Target: SPI/SPEI)**
    * Applies moving window convolutions over the baseline weekly data.
    * **Windows:** Generates cumulative sums for multi time scales (e.g. 30/90/180/360/720 days).

## Group 3: Soil Data Harmonization & AWC Derivation

This module aligns high-resolution soil databases with the Daymet grid and derives the Available Water Capacity (AWC) required for the hydrological balance model.

### 1. Grid Harmonization (`main_downsamplesoil.py`)
**Objective:** To unify the spatial resolution of soil inputs (originally 48x48 sub-grids) with the master Daymet geometry (12x12 grids).

* **Methodology:**
    * **Downsampling:** Applies mean-based coarsening to reduce spatial resolution while preserving regional soil properties.
    * **Alignment:** Ensures pixel-perfect correspondence with the Daymet `prcp` and `pet` layers used in Group 1 & 2.

### 2. AWC Calculation & Imputation (`main_processoildata2.py`)
**Objective:** To generate a gap-free AWC map for the top **1500mm** soil profile.

* **Rationale:** The Palmer indices do not require precise absolute AWC values (Palmer, 1965). The prcp and pet ar e more important. Therefore, for AWC, physical continuity is prioritized over local granular precision.
* **Processing Logic:**
    * **Depth Integration:** Calculates total AWC by assuming soil depth = 1500mm. The reference is the equation from Saxton (Saxton, 2006). 
    * **Two-Stage Gap Filling:**
        1.  **Nearest Neighbor:** Fills localized gaps using immediate spatial context.
        2.  **Global Mean:** Fills any remaining voids to ensure the dataset is strictly NaN-free.
* **Further check:** The cubes which required for filling will be listed in the report. The plausibility of values of those cubes can be checked by observing filled values. 

## Group 4: Multi-Scalar Drought Indices (SPI/SPEI)

This module implements the computation of standardized indices SPI and SPEI across multiple temporal scales at a weekly resolution.

### 1. Statistical Core (`Calculator_SPIandSPEI.py`)
**Objective:** To fit probability density functions (PDFs) to historical data and transform them into standardized normal variants (Z-scores).

* **SPI (Standardized Precipitation Index):**
    * **Distribution:** **Gamma Distribution** (2-parameter).
    * **Fitting Method:** Gamma Distribution (via `scipy.stats`).
* **SPEI (Standardized Precipitation Evapotranspiration Index):**
    * **Distribution:** **Generalised Logistic Distribution (GLO)** (3-parameter).
    * **Fitting Method:** **L-Moments** (via `lmoments3`).
    * *Note:* L-Moments are strictly selected over MLE for SPEI to provide robust parameter estimation for distributions with higher skewness.
* **Seasonal Calibration:** Parameters are fitted independently for each seasonal window (Weeks 1-52) to preserve the local climatological probability structure.

### 2. Execution Logic (`main_SPIandSPEI.py`)
**Objective:** Utilize the calculator to generate indices across meteorological and hydrological scales using the rolling variables from Group 2.

* **Current Time Scales:**
    * Processed scales: **n * 30days, n = 1/3/6/12/24** .

## Group 5: Palmer Drought Indices (PDSI/PHDI) & Water Balance

This module acts as the core hydrological engine, integrating the weekly climatic inputs (Group 2) and soil properties (Group 3) to compute the classic Palmer Drought Severity Index (PDSI) and Palmer Hydrological Drought Index (PHDI) at a weekly resolution.

### 1. Core Physics Engine (`Calculator_Week_Palmer_Final.py`)
**Objective:** To implement the "pseudo_weekly" based calculation logic of weekly Palmer indices.

* **The "Pseudo-Weekly" Adaptation:**
    * *Context:* The original Palmer algorithm was designed for monthly data. No standardized consensus exists for weekly implementations.
    * *Implementation:* This workflow adopts a **Pseudo-Weekly methodology**, this makes adapted some modifications for reasonal operational need and weekly computation of PDSI/PHDI based on literature.
    * *Plausibility check:* The plausibility of obtained PDSI was checked by comparing the plots of PDSI with 5-day PDSI from gridMET dataset. This comparison was conducted on some randomly selected and distributed cubes.

### 2. Execution Pipeline (`main_Week_Palmer_Final.py`)
**Objective:** To drive the calculation across spatially distributed cubes.

* **Inputs:**
    * `prcp`, `pet` (Weekly resolution).
    * `awc` (Static, it can be scaled as long as the value is not very small (Palmer, 1965)).
    * Calibration period is set to be 43 years (1980-2022).
* **Outputs:**
    * **PDSI:** Drought severity with memory.
    * **PHDI:** Hydrological drought severity (longer-term memory).
    * **RSM (Relative soil moisture) and RWD (Relative water deficit)** Side products from water balance model (short-term).

## Introduction of raw materials:

**Standard weekly variables**: 'prcp' and 'pet'. Time stamp is on every Sunday.
Meaning: Summation of prcp/pet of the past week. E.g. value on 2023.01.01 (Sunday) means the sum of prcp/pet from 2022.12.26 to 2023.01.01 (7days)
    
**Rolling weekly variables**: 'prcp-n' and 'Deficit-n', n is the time scale. Time stamp is on every Sunday.
Meaning: Summation of prcp/Deficit of n * 30days. E.g. value on 2023.01.01 (Sunday) means the sum of prcp/Deficit of past n * 30days (including 2023.01.01)
