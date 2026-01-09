<h3>Large Title</h3>

The CommHEAT Data Analysis Pipeline processes HOBO temperature sensor data, compares it with EnergyPlus building simulation outputs, and generates statistical analyses and visualizations. This system validates building energy models by comparing simulated indoor temperatures against measured temperatures in residential buildings during summer 2025 heat events.

<h3>Large Title</h3>

pip install pandas numpy matplotlib openpyxl tqdm

<h3>Large Title</h3>

cd "C:\Users\Ethan\OneDrive - Iowa State University\Desktop\CommHEAT Comparison"
python starting1.py


System Architecture

Module Structure

The pipeline uses four interconnected modules:

starting1.py - Main orchestrator that coordinates all operations and manages parallel processing workflows.
config.py - Configuration hub that defines directory paths, logging setup, regex patterns for data extraction, plot styling, and heat event definitions. Automatically creates output directory structure on initialization.
data_loader.py - Handles all file I/O with automatic header detection, column name normalization, intelligent column finding using multiple strategies, data validation, and LRU caching for performance optimization.
plotting_manager.py - Manages all visualizations with consistent styling, handling five plot types: full pilot period comparisons, individual archetype heat events, averaged archetype heat events, period intersections, and AC versus no-AC comparisons.

Processing Workflow
1. Initialization
The script loads configuration, creates output directories, initializes the PlottingManager and DataLoader, and sets up UTF-8 logging for Windows compatibility.
2. Sensor Mapping
Loads the Excel mapping file that connects sensor IDs to addresses, house types, archetypes, and CommHEAT usage periods. Derives house type (Apartment/Individual), cleans addresses for filenames, and parses date ranges.
3. HOBO Processing (Parallel)
Scans for sensor Excel files, extracts sensor IDs via regex, loads raw data with automatic header detection, converts Fahrenheit to Celsius (detects pre-converted data), resamples to hourly mean/max temperatures, and saves processed files to hobo_data_processed/. Generates HoboHouseIndex.xlsx summary.
4. Archetype Preloading
Extracts unique archetype names from mapping, loads all EnergyPlus simulation files into memory cache, parses EnergyPlus datetime format (handles 24:00:00 edge case), resamples to hourly, and stores in global cache with composite keys. This eliminates redundant file I/O during analysis.
5. MSE Analysis
Computes Mean Squared Error for AC and no-AC scenarios by determining period intersection between sensor and app usage dates, aligning simulation and measured data to common time index, averaging multiple archetypes when applicable, and calculating MSE using vectorized NumPy operations. Outputs Excel comparison files and PNG plots with MSE statistics.
6. Comprehensive MSE Comparison
Generates simplified summary comparing AC versus no-AC model accuracy with columns for address, archetypes, AC/no-AC MSE values, and actual temperatures.
7. Heat Event Analysis (Batch)
Processes predefined heat events (H1, H2, H3) and baseline periods (B1, B2) by loading archetype data once per address, generating individual archetype plots with separate MSE calculations, and creating averaged archetype plots combining multiple models. Batch processing dramatically reduces file I/O overhead.
8. AC vs No-AC Comparison
Directly compares AC and no-AC simulation variants, calculates mean and max temperature differences, and generates plots with statistical overlays showing cooling effects.
9. Period Intersection Plotting
Creates full pilot period visualizations showing averaged predictions and HOBO measurements. Individual archetype lines are hidden for clarity over long time spans.
---
