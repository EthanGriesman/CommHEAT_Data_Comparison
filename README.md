<h1 align="center">CommHEAT Data Analysis Pipeline</h1>

## Overview
The **CommHEAT Data Analysis Pipeline** processes HOBO temperature sensor data, compares it with EnergyPlus building simulation outputs, and generates statistical analyses and visualizations. The system is designed to validate residential building energy models by comparing simulated indoor temperatures against measured sensor data during summer 2025 heat events.

---

## Installation
To start, ensure you have the latest version of Python installed on your machine. 

```bash
python --version
```

Once complete, install the following dependency:

```bash
pip install pandas numpy matplotlib openpyxl tqdm
```

Before running the script as described below in section 1.0, download and then open `config.py` and update the directory paths so they match your local file structure.

```python
# Directory paths for input data
hobo_dir: Path = Path(
    r"C:\Users\Ethan\Downloads\2025SCC_OnsetHobo_InHome_Dataloggers\2025SCC_OnsetHobo_InHome_Dataloggers"
)

mapping_file: Path = Path(
    r"C:\Users\Ethan\Downloads\2025SCC_OnsetHobo_InHome_Dataloggers\2025SCC_OnsetHobo_InHome_Dataloggers\Sensor Contact_101325_PickUP.xlsx"
)

latest_ep_dir: Path = Path(
    r"C:\Users\Ethan\Downloads\Latest_EP_Output_Files\Latest_EP_Output_Files"
)

output_dir: Path = Path(
    r"C:\Users\Ethan\OneDrive - Iowa State University\Desktop\CommHEAT Output"
)
```


## System Architecture
The pipeline is organized as a modular data processing system designed for high performance through optimized file I/O, intelligent caching, and parallel execution of tasks in the process of generating analyses and plots of temperature comparisons.

---
### `starting1.py`
Main orchestrator responsible for coordinating all pipeline operations and managing parallel processing workflows.

---

### `config.py`
Central configuration hub that defines:
- Directory paths
- Logging setup
- Patterns for sensor ID extraction
- Plot styling
- Heat event definitions

Automatically creates the required output directory structure during initialization.

---

### `data_loader.py`
Handles all file input/output operations, including:
- Automatic header detection
- Column name normalization
- Intelligent column discovery using multiple heuristics
- Data validation

---

### `plotting_manager.py`
Manages all visualizations with consistent styling. Supports five plot types:
- Full pilot-period comparisons
- Individual archetype heat event plots
- Averaged archetype heat event plots
- Period intersection plots
- AC versus no-AC comparisons

---

## Processing Workflow

## 1.0 Useage
The following command begins the entire pipeline

```bash
python starting1.py
```

### 1.1 Initialization
- Loads configuration settings
- Creates output directories
- Initializes the `PlottingManager` and `DataLoader`
- Sets up UTF-8 logging for Windows compatibility

### 1.2 Sensor Mapping
- Loads the Excel mapping file linking:
  - Sensor IDs
  - Addresses
  - House types
  - Archetypes
  - CommHEAT usage periods
- Derives house type (Apartment or Individual)
- Cleans addresses for filesystem-safe filenames
- Parses sensor and application date ranges

---

### 1.3 HOBO Data Processing (Parallel)
- Scans for sensor Excel files
- Extracts sensor IDs using regex patterns
- Loads raw data with automatic header detection
- Converts Fahrenheit to Celsius (detects pre-converted datasets)
- Resamples data to hourly mean and maximum temperatures
- Saves processed datasets to `hobo_data_processed/`
- Generates `HoboHouseIndex.xlsx` summary file

---

### 1.4 Archetype Preloading
- Extracts unique archetype names from the mapping file
- Loads all EnergyPlus simulation files into memory
- Parses EnergyPlus datetime format (including 24:00:00 edge-case handling)
- Resamples simulation data to hourly resolution
- Stores results in a global cache using composite keys

This approach eliminates redundant file I/O during downstream analysis.

## 1.5 Clearing Output
To avoid manually having to delete output after each time the script is run, use the following command

```bash
python clear_output.py
```


---

### 1.6 MSE Analysis
- Computes Mean Squared Error (MSE) for AC and no-AC scenarios
- Determines period intersection between sensor measurement periods and CommHEAT usage dates
- Aligns simulated and measured data to a common time index
- Averages multiple archetypes when applicable
- Uses vectorized NumPy operations for efficient computation
- Outputs Excel comparison tables and PNG plots with embedded MSE statistics

---

### 1.6 Comprehensive MSE Comparison
- Generates a simplified summary comparing AC versus no-AC model accuracy
- Output includes:
  - Address
  - Archetypes
  - AC MSE
  - No-AC MSE
  - Observed temperature statistics

---

### 1.7 Heat Event Analysis (Batch)
- Processes predefined heat events (H1, H2, H3) and baseline periods (B1, B2)
- Loads archetype data once per address
- Generates:
  - Individual archetype heat event plots
  - Averaged archetype plots combining multiple models
- Batch processing significantly reduces file I/O overhead

---

### 1.8 AC vs No-AC Comparison
- Directly compares AC and no-AC simulation variants
- Calculates:
  - Mean temperature differences
  - Maximum temperature differences
- Produces plots with statistical overlays highlighting cooling effects

---

### 1.9 Period Intersection Visualization
- Creates full pilot-period plots showing:
  - Averaged EnergyPlus predictions
  - HOBO sensor measurements
- Individual archetype lines are hidden to improve clarity over long time spans

---

## Outputs
- Processed HOBO sensor datasets
- Excel summary tables
- High-resolution PNG plots
- Performance-optimized cached simulation data

---

## License
Internal research use for the CommHEAT project.
