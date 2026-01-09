<h3>Overview</h3>

The CommHEAT Data Analysis Pipeline processes HOBO temperature sensor data, compares it with EnergyPlus building simulation outputs, and generates statistical analyses and visualizations. This system validates building energy models by comparing simulated indoor temperatures against measured temperatures in residential buildings during summer 2025 heat events.

<h3>Installation</h3>

pip install pandas numpy matplotlib openpyxl tqdm

<h3>Basic Usage</h3>

cd "C:\Users\Ethan\OneDrive - Iowa State University\Desktop\CommHEAT Comparison"
python starting1.py


<h3>System Architecture</h3>

<h3>Module Structure</h3>

The pipeline uses four interconnected modules:

starting1.py - Main orchestrator that coordinates all operations and manages parallel processing workflows.
config.py - Configuration hub that defines directory paths, logging setup, regex patterns for data extraction, plot styling, and heat event definitions. Automatically creates output directory structure on initialization.
data_loader.py - Handles all file I/O with automatic header detection, column name normalization, intelligent column finding using multiple strategies, data validation, and LRU caching for performance optimization.
plotting_manager.py - Manages all visualizations with consistent styling, handling five plot types: full pilot period comparisons, individual archetype heat events, averaged archetype heat events, period intersections, and AC versus no-AC comparisons.

<h3>Processing Workflow</h3>
<h3>1. Initialization</h3>
The script loads configuration, creates output directories, initializes the PlottingManager and DataLoader, and sets up UTF-8 logging for Windows compatibility.
<h3>2. Sensor Mapping</h3>
Loads the Excel mapping file that connects sensor IDs to addresses, house types, archetypes, and CommHEAT usage periods. Derives house type (Apartment/Individual), cleans addresses for filenames, and parses date ranges.
<h3>3. HOBO Processing (Parallel)</h3>
Scans for sensor Excel files, extracts sensor IDs via regex, loads raw data with automatic header detection, converts Fahrenheit to Celsius (detects pre-converted data), resamples to hourly mean/max temperatures, and saves processed files to hobo_data_processed/. Generates HoboHouseIndex.xlsx summary.
<h3>4. Archetype Preloading</h3>
Extracts unique archetype names from mapping, loads all EnergyPlus simulation files into memory cache, parses EnergyPlus datetime format (handles 24:00:00 edge case), resamples to hourly, and stores in global cache with composite keys. This eliminates redundant file I/O during analysis.
<h3>5. MSE Analysis</h3>
Computes Mean Squared Error for AC and no-AC scenarios by determining period intersection between sensor and app usage dates, aligning simulation and measured data to common time index, averaging multiple archetypes when applicable, and calculating MSE using vectorized NumPy operations. Outputs Excel comparison files and PNG plots with MSE statistics.
<h3>6. Comprehensive MSE Comparison</h3>
Generates simplified summary comparing AC versus no-AC model accuracy with columns for address, archetypes, AC/no-AC MSE values, and actual temperatures.
<h3>7. Heat Event Analysis (Batch)</h3>
Processes predefined heat events (H1, H2, H3) and baseline periods (B1, B2) by loading archetype data once per address, generating individual archetype plots with separate MSE calculations, and creating averaged archetype plots combining multiple models. Batch processing dramatically reduces file I/O overhead.
<h3>8. AC vs No-AC Comparison</h3>
Directly compares AC and no-AC simulation variants, calculates mean and max temperature differences, and generates plots with statistical overlays showing cooling effects.
<h3>9. Period Intersection Plotting</h3>
Creates full pilot period visualizations showing averaged predictions and HOBO measurements. Individual archetype lines are hidden for clarity over long time spans.
---
