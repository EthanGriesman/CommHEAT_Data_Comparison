"""
CommHEAT Data Analysis Pipeline
================================

This script processes HOBO temperature sensor data, compares it with EnergyPlus
simulation outputs, and generates comparison plots and statistics.

Usage:
    python starting1.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
import logging
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache
from tqdm import tqdm
import warnings
import sys
import io

# import the plotting manager class
from plotting_manager import PlottingManager

# suppress pandas and numpy warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOGGING SETUP
# ============================================================

# handle windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('commheat_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# PATTERNS AND STYLING (MOVED BEFORE CONFIG)
# ============================================================

# regex patterns for data extraction and cleaning
PATTERNS = {
    'sensor_id': re.compile(r"(\d+)"),
    'archetype_temp': re.compile(r'^(FIRSTFLOOR_\d+|HOUSE_\d+):Zone Mean Air Temperature \[C\]\(Hourly\)$', re.IGNORECASE),
    'ordinal': re.compile(r"(\d+)(st|nd|rd|th)"),
    'whitespace': re.compile(r"\s+"),
    'address_clean': re.compile(r"[^A-Za-z0-9]")
}

# matplotlib style configuration for plots
PLOT_STYLE = {
    'figure.figsize': (12, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
}

# column name mappings for different data types
COLUMN_TYPES = {
    'time': ['date', 'time', 'timestamp'],
    'temp': ['temperature', 'zone mean air temperature'],
}

# heat event and baseline period definitions
HEAT_EVENTS = {
    'H1': {
        'start': '2025-08-07',
        'end': '2025-08-10',
        'name': 'Heat Event 1'
    },
    'H2': {
        'start': '2025-08-15',
        'end': '2025-08-19',
        'name': 'Heat Event 2'
    },
    'H3': {
        'start': '2025-09-12',
        'end': '2025-09-16',
        'name': 'Heat Event 3'
    },
    'B1': {
        'start': '2025-09-26',
        'end': '2025-09-30',
        'name': 'Baseline Period 1'
    },
    'B2': {
        'start': '2025-10-09',
        'end': '2025-10-12',
        'name': 'Baseline Period 2'
    },
}


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    """Centralized configuration management"""

    # directory paths for input data
    hobo_dir: Path = Path(r"C:\Users\Ethan\Downloads\2025SCC_OnsetHobo_InHome_Dataloggers\2025SCC_OnsetHobo_InHome_Dataloggers")
    mapping_file: Path = Path(r"C:\Users\Ethan\Downloads\2025SCC_OnsetHobo_InHome_Dataloggers\2025SCC_OnsetHobo_InHome_Dataloggers\Sensor Contact_101325_PickUP.xlsx")
    latest_ep_dir: Path = Path(r"C:\Users\Ethan\Downloads\Latest_EP_Output_Files\Latest_EP_Output_Files")
    output_dir: Path = Path(r"C:\Users\Ethan\OneDrive - Iowa State University\Desktop\CommHEAT Output")

    # hobo sensor data column names
    hobo_time_col: str = "Date-Time (CDT)"
    hobo_temp_col: str = "Temperature , °F"
    hobo_rh_col: str = "RH , %"

    # energyplus target columns for temperature data
    target_columns: List[str] = None
    # plotting manager instance
    plotter: PlottingManager = None

    def __post_init__(self):
        # create output directory structure
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.comparison_dir = self.output_dir / "comparisons"
        self.comparison_dir.mkdir(exist_ok=True, parents=True)
        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True, parents=True)
        self.hobo_output_dir = self.output_dir / "hobo_data_processed"
        self.hobo_output_dir.mkdir(exist_ok=True, parents=True)
        # create heat events subdirectory
        self.heat_events_dir = self.plot_dir / "heat_events_each_archetype"
        self.heat_events_dir.mkdir(exist_ok=True, parents=True)
        # create averaged heat events subdirectory
        self.heat_events_averaged_dir = self.plot_dir / "heat_events_averaged_archetypes"
        self.heat_events_averaged_dir.mkdir(exist_ok=True, parents=True)
        # create period intersection subdirectory
        self.period_intersection_dir = self.plot_dir / "period_intersection"
        self.period_intersection_dir.mkdir(exist_ok=True, parents=True)
        # create ac/noac subdirectory
        self.ac_noac_dir = self.plot_dir / "ac_noac"
        self.ac_noac_dir.mkdir(exist_ok=True, parents=True)
        # create averaged comparisons subdirectory in plots
        self.averaged_comparisons_dir = self.plot_dir / "averaged_comparisons"
        self.averaged_comparisons_dir.mkdir(exist_ok=True, parents=True)

        # set default target columns if not provided
        if self.target_columns is None:
            self.target_columns = [
                "FIRSTFLOOR_0:Zone Mean Air Temperature [C](Hourly)",
                "FIRSTFLOOR_1:Zone Mean Air Temperature [C](Hourly)",
                "FIRSTFLOOR_2:Zone Mean Air Temperature [C](Hourly)",
                "HOUSE_0:Zone Mean Air Temperature [C](Hourly)"
            ]
        
        # initialize plotting manager with output directory and style
        self.plotter = PlottingManager(self.output_dir, PLOT_STYLE)

# instantiate global config
config = Config()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def color_text(txt: str, code: str) -> str:
    """Apply ANSI color code to text"""
    return f"\033[{code}m{txt}\033[0m"


def clean_text(s: Union[str, any]) -> str:
    """Remove ordinal suffixes and normalize whitespace"""
    # convert non-strings to string
    if not isinstance(s, str):
        return str(s)
    # remove newlines and carriage returns
    s = s.strip().replace("\n", " ").replace("\r", " ")
    # remove ordinal suffixes (1st -> 1, 2nd -> 2)
    s = PATTERNS['ordinal'].sub(r"\1", s)
    # normalize multiple spaces to single space
    return PATTERNS['whitespace'].sub(" ", s)


def parse_date(date_str: any, target_year: Optional[int] = None, ep_format: bool = False) -> pd.Timestamp:
    """
    Universal date parser for both HOBO and EnergyPlus formats.

    Args:
        date_str: Date string to parse
        target_year: Year to assign (for EnergyPlus dates)
        ep_format: If True, parse as EnergyPlus format (MM/DD HH:MM:SS)
    """
    # handle missing values
    if pd.isna(date_str):
        return pd.NaT

    # clean the input string
    s = clean_text(date_str)

    # energyplus format: "08/27 13:00:00"
    if ep_format:
        try:
            # normalize whitespace
            s = " ".join(s.split())
            dt = pd.to_datetime(s, format="%m/%d %H:%M:%S", errors="coerce")
            # assign target year if provided
            if pd.notna(dt) and target_year:
                return dt.replace(year=target_year)
            return dt
        except:
            return pd.NaT

    # try multiple standard date formats
    formats = [
        "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%m-%d-%Y", "%m-%d-%y",
        "%b %d %Y", "%b %d, %Y", "%B %d %Y", "%B %d, %Y",
        "%b %d %I:%M%p", "%B %d %I:%M%p"
    ]

    # attempt each format until one succeeds
    for fmt in formats:
        try:
            return pd.to_datetime(s, format=fmt)
        except:
            continue

    # fallback to pandas automatic parsing
    return pd.to_datetime(s, errors='coerce')


def parse_energyplus_datetime(value: any, target_year: int) -> pd.Timestamp:
    """
    More robust EP datetime parser.
    Handles:
      - 'MM/DD HH:MM:SS' without year
      - '24:00:00' edge case by rolling to next day 00:00:00
    """
    # handle missing values
    if pd.isna(value):
        return pd.NaT

    # clean the input string
    s = clean_text(value)

    # handle energyplus "24:00:00" which pandas doesn't parse with %h
    # example: "08/27 24:00:00" -> "08/28 00:00:00"
    try:
        # split into date and time components
        parts = s.split()
        if len(parts) >= 2:
            md = parts[0]
            t = parts[1]
            # check for 24:xx:xx time
            if t.startswith("24:"):
                # parse date part, then add 1 day and set time to 00:...
                base = pd.to_datetime(f"{md} 00:00:00", format="%m/%d %H:%M:%S", errors="coerce")
                if pd.notna(base):
                    # add one day and assign target year
                    base = base.replace(year=target_year) + pd.Timedelta(days=1)
                    return base
    except Exception:
        pass

    # standard parsing for normal times
    dt = pd.to_datetime(s, format="%m/%d %H:%M:%S", errors="coerce")
    if pd.notna(dt):
        # assign target year
        dt = dt.replace(year=target_year)

    return dt


def validate_period_intersection(start: pd.Timestamp, end: pd.Timestamp, min_hours: int = 1) -> Tuple[bool, str]:
    """Validate period_intersection period meets minimum requirements"""
    # check for invalid timestamps
    if pd.isna(start) or pd.isna(end):
        return False, "Invalid timestamps (NaT)"
    # ensure start is before end
    if start >= end:
        return False, f"Start must be before end"

    # calculate duration in hours
    hours = (end - start).total_seconds() / 3600
    # check minimum duration requirement
    if hours < min_hours:
        return False, f"Period ({hours:.1f}h) < minimum ({min_hours}h)"

    return True, f"Valid: {hours:.1f} hours"


def convert_to_celsius(series: pd.Series) -> pd.Series:
    """Convert Fahrenheit to Celsius if needed"""
    # convert to numeric, handling errors
    series = pd.to_numeric(series, errors='coerce')
    # assume values > 60 are fahrenheit and convert
    return np.where(series > 60, (series - 32) * 5 / 9, series)


# ============================================================
# FILE LOADING
# ============================================================

def load_dataframe(filepath: Path, max_skiprows: int = 3) -> Optional[pd.DataFrame]:
    """Load CSV or Excel with automatic skiprows detection"""
    # try different skiprows values to handle header rows
    for skiprows in range(max_skiprows + 1):
        try:
            # load excel or csv based on extension
            df = pd.read_excel(filepath, skiprows=skiprows) if filepath.suffix.lower() == ".xlsx" \
                else pd.read_csv(filepath, skiprows=skiprows)

            # clean column names
            df.columns = df.columns.str.strip()

            # validate: must have time column and >100 rows
            time_cols = [c for c in df.columns if any(t in c.lower() for t in COLUMN_TYPES['time'])]
            if time_cols and len(df) > 100:
                # log successful load with skiprows
                if skiprows > 0:
                    logger.debug(f"Loaded {filepath.name} with {skiprows} rows skipped")
                return df
        except Exception as e:
            logger.debug(f"Skip {skiprows} failed for {filepath.name}: {e}")

    # all attempts failed
    logger.warning(f"Could not load {filepath.name}")
    return None


def find_column(df: pd.DataFrame, col_type: str) -> Optional[str]:
    """Find column by type using COLUMN_TYPES mapping"""
    # special handling for temperature columns
    if col_type == 'temp':
        # strategy 1: pattern match for archetype format
        cols = [c for c in df.columns if PATTERNS['archetype_temp'].match(c)]
        if cols:
            return cols[0]

        # strategy 2: contains specific keywords
        cols = [c for c in df.columns
                if all(k in c.lower() for k in ['zone mean air temperature', '[c]'])]
        if cols:
            return cols[0]

        # strategy 3: match target columns list
        cols = [c for c in config.target_columns if c in df.columns]
        if cols:
            return cols[0]

        # strategy 4: any temp column with [c] unit
        cols = [c for c in df.columns if 'temperature' in c.lower() and '[c]' in c.lower()]
        return cols[0] if cols else None

    # generic search for other column types
    keywords = COLUMN_TYPES.get(col_type, [])
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None


# ============================================================
# DATA LOADING
# ============================================================

def load_mapping() -> pd.DataFrame:
    """Load and process sensor mapping file"""
    try:
        # load excel file with header on row 3
        df = pd.read_excel(config.mapping_file, header=2)
        # normalize column names to lowercase
        df.columns = df.columns.str.strip().str.lower()

        # rename key columns to standard names
        df = df.rename(columns={"home address": "address", "sensor #": "sensor_id"})
        # drop rows missing critical data
        df = df.dropna(subset=["sensor_id", "address"])
        # ensure sensor_id is string type
        df["sensor_id"] = df["sensor_id"].astype(str)

        # classify house type based on address
        df["housetype"] = df["address"].str.contains("apt|apartment", case=False, regex=True).map({True: "Apt", False: "Ind"})
        # create clean address for filenames
        df["clean_address"] = df["address"].apply(lambda x: PATTERNS['address_clean'].sub("", x))
        # generate output filename
        df["outfile"] = df["housetype"] + "_" + df["clean_address"] + ".xlsx"

        # parse commheat start and end dates
        for col in ["commheat start", "commheat end"]:
            if col in df.columns:
                df[col] = df[col].apply(parse_date)

        logger.info(f"Loaded {len(df)} sensor mappings")
        return df

    except Exception as e:
        logger.error(f"Error loading mapping: {e}", exc_info=True)
        raise


def process_hobo_file(file_path: Path, mapping_df: pd.DataFrame) -> Optional[Dict]:
    """Process HOBO sensor file"""
    # extract sensor id from filename
    sensor_id = PATTERNS['sensor_id'].match(file_path.name)
    if not sensor_id:
        return None

    # get sensor id as string
    sensor_id = sensor_id.group(1)
    # find matching row in mapping
    row_df = mapping_df[mapping_df["sensor_id"] == sensor_id]
    if row_df.empty:
        return None

    # get first matching row
    row = row_df.iloc[0]
    # determine output file path
    outfile = config.hobo_output_dir / row["outfile"]

    try:
        # load hobo data file
        df = pd.read_excel(file_path)
        # parse datetime column
        df[config.hobo_time_col] = pd.to_datetime(df[config.hobo_time_col])
        # set datetime as index and sort
        df = df.set_index(config.hobo_time_col).sort_index()
        # convert temperature to celsius
        df["temp_C"] = convert_to_celsius(df[config.hobo_temp_col])

        # resample to hourly data
        hourly = df.resample("h").agg({
            "temp_C": ["mean", "max"],
            config.hobo_rh_col: "mean"
        })

        # rename columns to standard names
        hourly.columns = ["actual_average_temperature", "actual_max_temperature", "average_relative_humidity"]
        # save processed data to excel
        hourly.to_excel(outfile, index_label="timestamp")

        logger.info(f"Processed {file_path.name}: {hourly.index.min()} to {hourly.index.max()}")

        # return summary metadata
        return {
            "sensor_id": sensor_id,
            "address": row["address"],
            "outfile": outfile.name,
            "housetype": row["housetype"],
            "period_hobologger_start": hourly.index.min(),
            "period_hobologger_end": hourly.index.max(),
            "period_app_usage_start": row.get("commheat start", pd.NaT),
            "period_app_usage_end": row.get("commheat end", pd.NaT),
            "archetypes_used": row.get("archtypes used") if isinstance(row.get("archtypes used"), str) else None
        }

    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        return None


@lru_cache(maxsize=64)
def load_simulation_cached(prefix: str, target_year: int, sim_dir_str: str) -> Optional[pd.DataFrame]:
    """Cached simulation loader"""
    # convert string path back to path object
    return load_simulation(prefix, target_year, Path(sim_dir_str))


def load_simulation(prefix: str, target_year: int = 2025, sim_dir: Path = None) -> Optional[pd.DataFrame]:
    """Load simulation data for archetype"""
    # use default simulation directory if not provided
    if sim_dir is None:
        sim_dir = config.latest_ep_dir

    # search for matching simulation files (csv or xlsx)
    matches = list(sim_dir.glob(f"{prefix}*.csv")) or list(sim_dir.glob(f"{prefix}*.xlsx"))
    if not matches:
        logger.warning(f"No simulation file for: {prefix}")
        return None

    try:
        # load first matching file
        df = load_dataframe(matches[0])
        if df is None:
            return None

        # find time column
        time_col = find_column(df, 'time')
        if not time_col:
            return None

        # parse datetime with energyplus format
        df[time_col] = df[time_col].apply(lambda x: parse_energyplus_datetime(x, target_year))
        # set datetime as index and sort
        df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

        # find temperature column
        temp_col = find_column(df, 'temp')
        if not temp_col:
            return None

        # extract temperature data
        out = df[[temp_col]].copy()
        # rename to standard column name
        out.columns = ["archetype_internal_temperature"]

        # align timestamps to hobo hourly index
        # floor to hour
        out.index = out.index.floor("h")
        # group by hour and average
        out = out.groupby(out.index).mean()
        # resample to ensure hourly frequency
        out = out.resample("h").mean()

        return out

    except Exception as e:
        logger.error(f"Error loading simulation: {e}", exc_info=True)
        return None


def load_archetype_series(
    prefix: str,
    target_year: int,
    patterns: List[str],
    period_intersection_start: Optional[pd.Timestamp] = None,
    period_intersection_end: Optional[pd.Timestamp] = None
) -> Optional[pd.Series]:
    """
    Unified loader for EP output files (AC/NoAC).

    Args:
        prefix: Archetype prefix
        target_year: Year to assign
        patterns: File patterns to search
        period_intersection_start/end: Optional time window
    """
    # collect all matching files
    files = []
    for pat in patterns:
        files.extend(list(config.latest_ep_dir.glob(pat)))

    if not files:
        return None

    # collect temperature series from each file
    series_list = []

    for f in files:
        try:
            # load dataframe with header detection
            df = load_dataframe(f, max_skiprows=3)
            if df is None:
                continue

            # find time column
            time_col = find_column(df, 'time')
            if not time_col:
                continue

            # parse datetime and set as index
            df[time_col] = df[time_col].apply(lambda x: parse_energyplus_datetime(x, target_year))
            df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
            # floor timestamps to hour
            df.index = df.index.floor("h")
            # group by hour and average
            df = df.groupby(df.index).mean()

            # find temperature column
            temp_col = find_column(df, 'temp')
            if not temp_col:
                continue

            # extract temperature series
            temp_series = pd.to_numeric(df[temp_col], errors='coerce').dropna()
            # resample to hourly frequency
            temp_series = temp_series.resample("h").mean()

            # filter to period_intersection if specified
            if period_intersection_start and period_intersection_end:
                # exact date range filtering
                temp_series = temp_series.loc[period_intersection_start:period_intersection_end]

                # relaxed matching fallback if exact match is empty
                if temp_series.empty:
                    full_series = pd.to_numeric(df[temp_col], errors='coerce')
                    # create mask for approximate date range
                    mask = (
                        (full_series.index.month >= period_intersection_start.month) &
                        (full_series.index.month <= period_intersection_end.month) &
                        (full_series.index.day >= period_intersection_start.day - 1) &
                        (full_series.index.day <= period_intersection_end.day + 1)
                    )
                    temp_series = full_series[mask].dropna()

            # add non-empty series to list
            if not temp_series.empty:
                series_list.append(temp_series)

        except Exception as e:
            logger.debug(f"Error loading {f.name}: {e}")

    if not series_list:
        return None

    # combine all series and average
    df_all = pd.concat(series_list, axis=1)
    return df_all.mean(axis=1)


def load_ac_noac_series(
    prefix: str,
    target_year: int = 2025,
    period_intersection_start: Optional[pd.Timestamp] = None,
    period_intersection_end: Optional[pd.Timestamp] = None
) -> Dict[str, Optional[pd.Series]]:
    """Load AC and No-AC series separately"""
    # define file patterns for ac and noac outputs
    ac_patterns = [f"{prefix}*ac_out*.xlsx", f"{prefix}*ac_out*.csv"]
    noac_patterns = [f"{prefix}*noac_out*.xlsx", f"{prefix}*noac_out*.csv"]

    # load both series types
    return {
        "AC": load_archetype_series(prefix, target_year, ac_patterns, period_intersection_start, period_intersection_end),
        "NoAC": load_archetype_series(prefix, target_year, noac_patterns, period_intersection_start, period_intersection_end)
    }


# ============================================================
# HEAT EVENT PLOTTING
# ============================================================

def plot_heat_events_for_row(row: pd.Series, hourly: pd.DataFrame, arch: str) -> None:
    """Generate plots for each heat event and baseline period"""
    # iterate through all heat events and baseline periods
    for event_id, event_info in HEAT_EVENTS.items():
        # parse event start and end dates
        event_start = pd.to_datetime(event_info['start'])
        event_end = pd.to_datetime(event_info['end'])
        
        # extract hobo data for this event period
        hobo_slice = hourly.loc[event_start:event_end].copy()
        
        # skip if no data for this period
        if hobo_slice.empty:
            logger.debug(f"No HOBO data for {row['address']} during {event_id}")
            continue
        
        # process each archetype separately (existing logic)
        for archetype in [a.strip() for a in arch.split(",") if a.strip()]:
            # load simulation data for archetype
            sim_df = load_simulation(archetype, target_year=2025, sim_dir=config.latest_ep_dir)
            
            if sim_df is None or sim_df.empty:
                logger.debug(f"No simulation data for {archetype} during {event_id}")
                continue
            
            # extract simulation data for this event period
            sim_slice = sim_df.loc[event_start:event_end].copy()
            
            if sim_slice.empty:
                logger.debug(f"No simulation data for {archetype} during {event_id}")
                continue
            
            # create common hourly index
            common_idx = pd.date_range(
                start=event_start,
                end=event_end,
                freq="h"
            )
            
            # align hobo and simulation data
            hobo_aligned = hobo_slice.reindex(common_idx)
            sim_aligned = sim_slice.reindex(common_idx)
            
            # combine datasets
            combined = hobo_aligned.join(sim_aligned, how="outer")
            
            if combined.empty:
                continue
            
            # create output filename
            label = f"{row['clean_address']}_{archetype}_{event_id}"
            out_path = config.heat_events_dir / f"{label}.png"
            
            # use plottingmanager to create heat event plot
            config.plotter.plot_heat_event(
                combined_data=combined,
                address=row['address'],
                archetype=archetype,
                event_name=event_info['name'],
                event_id=event_id,
                output_path=out_path
            )
            
            logger.info(f"Created heat event plot: {out_path.name}")


def plot_averaged_heat_events_for_row(row: pd.Series, hourly: pd.DataFrame, arch: str) -> None:
    """Generate averaged archetype plots for each heat event and baseline period"""
    archetypes = [a.strip() for a in arch.split(",") if a.strip()]
    
    # iterate through all heat events and baseline periods
    for event_id, event_info in HEAT_EVENTS.items():
        # parse event start and end dates
        event_start = pd.to_datetime(event_info['start'])
        event_end = pd.to_datetime(event_info['end'])
        
        # extract hobo data for this event period
        hobo_slice = hourly.loc[event_start:event_end].copy()
        
        # skip if no data for this period
        if hobo_slice.empty:
            logger.debug(f"No HOBO data for {row['address']} during {event_id}")
            continue
        
        # collect archetype series for averaging
        archetype_series_list = []
        archetype_names = []
        
        for archetype in archetypes:
            # load simulation data for archetype
            sim_df = load_simulation(archetype, target_year=2025, sim_dir=config.latest_ep_dir)
            
            if sim_df is None or sim_df.empty:
                logger.debug(f"No simulation data for {archetype} during {event_id}")
                continue
            
            # extract simulation data for this event period
            sim_slice = sim_df.loc[event_start:event_end].copy()
            
            if sim_slice.empty:
                logger.debug(f"No simulation data for {archetype} during {event_id}")
                continue
            
            archetype_series_list.append(sim_slice)
            archetype_names.append(archetype)
        
        # skip if no archetype data available
        if not archetype_series_list:
            logger.debug(f"No archetype data for {row['address']} during {event_id}")
            continue
        
        # create common hourly index
        common_idx = pd.date_range(
            start=event_start,
            end=event_end,
            freq="h"
        )
        
        # align hobo data to common index
        hobo_aligned = hobo_slice.reindex(common_idx)
        
        # align all archetype series and collect for averaging
        aligned_archetype_temps = []
        for sim_slice in archetype_series_list:
            sim_aligned = sim_slice.reindex(common_idx)
            aligned_archetype_temps.append(sim_aligned["archetype_internal_temperature"])
        
        # calculate average predicted temperature across all archetypes at each timestep
        if aligned_archetype_temps:
            t_predicted_df = pd.concat(aligned_archetype_temps, axis=1)
            t_predicted_df.columns = archetype_names
            t_predicted = t_predicted_df.mean(axis=1)
        else:
            logger.debug(f"No aligned archetype data for {row['address']} during {event_id}")
            continue
        
        # combine hobo and predicted data
        combined = hobo_aligned.copy()
        combined['T_predicted'] = t_predicted
        
        # also add individual archetype temps for comparison
        for i, arch_name in enumerate(archetype_names):
            combined[f'{arch_name}_temp'] = aligned_archetype_temps[i]
        
        if combined.empty:
            continue
        
        # create output filename
        label = f"{row['clean_address']}_averaged_{event_id}"
        out_path = config.heat_events_averaged_dir / f"{label}.png"
        
        # use plottingmanager to create averaged heat event plot
        config.plotter.plot_heat_event_averaged(
            combined_data=combined,
            address=row['address'],
            archetypes=", ".join(archetype_names),
            event_name=event_info['name'],
            event_id=event_id,
            output_path=out_path
        )
        
        logger.info(f"Created averaged heat event plot: {out_path.name}")


def plot_period_intersection_for_row(row: pd.Series, hourly: pd.DataFrame, 
                                      period_intersection_start: pd.Timestamp,
                                      period_intersection_end: pd.Timestamp,
                                      arch: str) -> None:
    """Generate averaged archetype plot for entire period intersection"""
    archetypes = [a.strip() for a in arch.split(",") if a.strip()]
    
    # extract hobo data for period intersection
    hobo_slice = hourly.loc[period_intersection_start:period_intersection_end].copy()
    
    # skip if no data for this period
    if hobo_slice.empty:
        logger.debug(f"No HOBO data for {row['address']} during period intersection")
        return
    
    # collect archetype series for averaging
    archetype_series_list = []
    archetype_names = []
    
    for archetype in archetypes:
        # load simulation data for archetype
        sim_df = load_simulation(archetype, target_year=2025, sim_dir=config.latest_ep_dir)
        
        if sim_df is None or sim_df.empty:
            logger.debug(f"No simulation data for {archetype} during period intersection")
            continue
        
        # extract simulation data for period intersection
        sim_slice = sim_df.loc[period_intersection_start:period_intersection_end].copy()
        
        if sim_slice.empty:
            logger.debug(f"No simulation data for {archetype} during period intersection")
            continue
        
        archetype_series_list.append(sim_slice)
        archetype_names.append(archetype)
    
    # skip if no archetype data available
    if not archetype_series_list:
        logger.debug(f"No archetype data for {row['address']} during period intersection")
        return
    
    # create common hourly index
    common_idx = pd.date_range(
        start=period_intersection_start,
        end=period_intersection_end,
        freq="h"
    )
    
    # align hobo data to common index
    hobo_aligned = hobo_slice.reindex(common_idx)
    
    # align all archetype series and collect for averaging
    aligned_archetype_temps = []
    for sim_slice in archetype_series_list:
        sim_aligned = sim_slice.reindex(common_idx)
        aligned_archetype_temps.append(sim_aligned["archetype_internal_temperature"])
    
    # calculate average predicted temperature across all archetypes at each timestep
    if aligned_archetype_temps:
        t_predicted_df = pd.concat(aligned_archetype_temps, axis=1)
        t_predicted_df.columns = archetype_names
        t_predicted = t_predicted_df.mean(axis=1)
    else:
        logger.debug(f"No aligned archetype data for {row['address']} during period intersection")
        return
    
    # combine hobo and predicted data
    combined = hobo_aligned.copy()
    combined['T_predicted'] = t_predicted
    
    # also add individual archetype temps for comparison
    for i, arch_name in enumerate(archetype_names):
        combined[f'{arch_name}_temp'] = aligned_archetype_temps[i]
    
    if combined.empty:
        return
    
    # create output filename
    label = f"{row['clean_address']}_period_intersection"
    out_path = config.period_intersection_dir / f"{label}.png"
    
    # use plottingmanager to create period intersection plot
    config.plotter.plot_period_intersection(
        combined_data=combined,
        address=row['address'],
        archetypes=", ".join(archetype_names),
        period_start=period_intersection_start,
        period_end=period_intersection_end,
        output_path=out_path
    )
    
    logger.info(f"Created period intersection plot: {out_path.name}")


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def process_mapping_row(row: pd.Series, analysis_type: str) -> Optional[Dict]:
    """
    Unified processor for mapping rows.

    Args:
        row: Mapping DataFrame row
        analysis_type: 'mse', 'period_intersection_means', 'ac_noac_plots', 'heat_events', 
                      'heat_events_averaged', or 'period_intersection_plots'
    """
    # get archetype information from row
    arch = row.get("archtypes used")
    if not isinstance(arch, str):
        return None

    # check if processed hobo file exists
    hobo_file = config.hobo_output_dir / row["outfile"]
    if not hobo_file.exists():
        return None

    try:
        # load processed hobo data
        hourly = pd.read_excel(hobo_file, index_col="timestamp")
        # ensure datetime index
        hourly.index = pd.to_datetime(hourly.index)

        # handle heat events separately (no period intersection needed)
        if analysis_type == 'heat_events':
            plot_heat_events_for_row(row, hourly, arch)
            return None
        
        # handle averaged heat events
        if analysis_type == 'heat_events_averaged':
            plot_averaged_heat_events_for_row(row, hourly, arch)
            return None

        # get app usage period from mapping
        app_start = row.get("commheat start")
        app_end = row.get("commheat end")

        # skip if missing usage period
        if pd.isna(app_start) or pd.isna(app_end):
            return None

        # calculate intersection period between hobo data and app usage
        period_intersection_start = max(hourly.index.min(), app_start)
        period_intersection_end = min(hourly.index.max(), app_end)

        # validate intersection period
        is_valid, msg = validate_period_intersection(period_intersection_start, period_intersection_end)
        if not is_valid:
            logger.warning(f"Invalid period_intersection for {row['address']}: {msg}")
            return None

        # route to appropriate analysis function
        if analysis_type == 'mse':
            return compute_mse_for_row(row, hourly, period_intersection_start, period_intersection_end, arch)
        elif analysis_type == 'period_intersection_means':
            return compute_period_intersection_means_for_row(row, hourly, period_intersection_start, period_intersection_end, arch)
        elif analysis_type == 'ac_noac_plots':
            plot_ac_noac_for_row(row, period_intersection_start, period_intersection_end, arch)
            return None
        elif analysis_type == 'period_intersection_plots':
            plot_period_intersection_for_row(row, hourly, period_intersection_start, period_intersection_end, arch)
            return None

    except Exception as e:
        logger.error(f"Error processing {row['address']}: {e}", exc_info=True)
        return None


def compute_mse_for_row(row: pd.Series, hourly: pd.DataFrame, obs_start: pd.Timestamp, obs_end: pd.Timestamp, arch: str) -> Optional[Dict]:
    """Compute MSE between HOBO and simulation"""
    # extract observation window from hobo data
    hobo_window = hourly.loc[obs_start:obs_end].copy()
    if hobo_window.empty:
        return None

    # align timestamps to hourly intervals
    hobo_window.index = pd.to_datetime(hobo_window.index).floor("h")
    # group by hour and average
    hobo_window = hobo_window.groupby(hobo_window.index).mean()

    # collect results for each archetype
    results = []

    # print header for mse calculations
    print(f"\n{'='*80}")
    print(f"MSE CALCULATION FOR: {row['address']}")
    print(f"{'='*80}")

    # collect all archetype temperature series for averaging
    archetype_series_list = []
    archetypes = [a.strip() for a in arch.split(",")]

    # load simulation data for all archetypes first
    for archetype in archetypes:
        print(f"\nArchetype: {archetype}")

        # load simulation data for archetype
        sim_df = load_simulation(archetype, target_year=hobo_window.index.min().year, sim_dir=config.latest_ep_dir)

        if sim_df is None or sim_df.empty:
            print(f"  ✗ No simulation data found for {archetype}")
            continue

        # prepare simulation data
        sim_df = sim_df.copy()
        # align timestamps to hourly intervals
        sim_df.index = pd.to_datetime(sim_df.index).floor("h")
        # group by hour and average
        sim_df = sim_df.groupby(sim_df.index).mean()
        # resample to ensure hourly frequency
        sim_df = sim_df.resample("h").mean()

        archetype_series_list.append((archetype, sim_df))

    if not archetype_series_list:
        print(f"  ✗ No simulation data found for any archetype")
        return None

    # determine common overlap window across all archetypes
    actual_start = max(obs_start, max(s[1].index.min() for s in archetype_series_list))
    actual_end = min(obs_end, min(s[1].index.max() for s in archetype_series_list))

    # validate overlap window
    if actual_start >= actual_end:
        print(f"  ✗ Invalid time range across archetypes")
        return None

    # create common hourly index
    common_idx = pd.date_range(
        start=pd.to_datetime(actual_start).floor("h"),
        end=pd.to_datetime(actual_end).floor("h"),
        freq="h"
    )

    # align hobo data to common index
    hobo_aligned = hobo_window.reindex(common_idx)

    # align all archetype series and collect for averaging
    aligned_archetype_temps = []
    for archetype, sim_df in archetype_series_list:
        sim_aligned = sim_df.reindex(common_idx)
        aligned_archetype_temps.append(sim_aligned["archetype_internal_temperature"])

    # calculate average predicted temperature across all archetypes at each timestep
    if aligned_archetype_temps:
        t_predicted_df = pd.concat(aligned_archetype_temps, axis=1)
        t_predicted = t_predicted_df.mean(axis=1)
    else:
        print(f"  ✗ No aligned archetype data available")
        return None

    # combine hobo and predicted data
    combined = hobo_aligned.join(pd.DataFrame({"T_predicted": t_predicted}), how="inner").dropna()

    if combined.empty:
        print(f"  ✗ No overlapping data after alignment")
        return None

    # calculate mean values for display
    hobo_avg_mean = combined["actual_average_temperature"].mean()
    hobo_max_mean = combined["actual_max_temperature"].mean()
    t_predicted_mean = combined["T_predicted"].mean()

    # number of valid hourly samples
    n = len(combined)

    # --- MSE: T_predicted vs actual_average_temperature ---
    diff_avg = combined["T_predicted"] - combined["actual_average_temperature"]
    mse_predicted_avg = (diff_avg.pow(2).sum()) / n

    # --- MSE: T_predicted vs actual_max_temperature ---
    diff_max = combined["T_predicted"] - combined["actual_max_temperature"]
    mse_predicted_max = (diff_max.pow(2).sum()) / n

    # print comparison results
    print(f"\n  Comparison 1: T_predicted (averaged across archetypes) vs actual_average_temperature")
    print(f"  Actual Average Temperature (Hobologger): {hobo_avg_mean:.3f} °C")
    print(f"  T_predicted (Averaged Archetypes): {t_predicted_mean:.3f} °C")
    print(f"  Final Calculated MSE: {mse_predicted_avg:.6f} C²")

    print(f"\n  Comparison 2: T_predicted (averaged across archetypes) vs actual_max_temperature")
    print(f"  Actual Max Temperature (Hobologger): {hobo_max_mean:.3f} °C")
    print(f"  T_predicted (Averaged Archetypes): {t_predicted_mean:.3f} °C")
    print(f"  Final Calculated MSE: {mse_predicted_max:.6f} C²")
    
    print(f"\n{'='*80}\n")

    # create output label for files
    label = f"{row['housetype']}_{row['clean_address']}"

    # save comparison data with clearer column names
    comparison_data = combined.copy()
    comparison_data = comparison_data.rename(columns={
        "actual_average_temperature": "Hobologger_avg",
        "actual_max_temperature": "Hobologger_max"
    })

    # save comparison data to excel in comparisons folder (no CSV)
    xlsx_path = config.comparison_dir / f"{label}_averaged_comparison.xlsx"
    comparison_data.to_excel(xlsx_path)

    # NOW generate the plot and save to plots/averaged_comparisons folder
    print(f"Generating plot for {row['address']}...")
    png_path = config.averaged_comparisons_dir / f"{label}_averaged_comparison.png"
    config.plotter.plot_intersection_comparison_averaged(
        combined_data=combined,
        row_info={"address": row["address"]},
        archetypes=", ".join(archetypes),
        mse_predicted_avg=mse_predicted_avg,
        mse_predicted_max=mse_predicted_max,
        actual_start=actual_start,
        actual_end=actual_end,
        output_path=png_path
    )

    # append results
    results.append({
        "address": row["address"],
        "archetypes": ", ".join(archetypes),
        "period_intersection_start": actual_start,
        "period_intersection_end": actual_end,
        "period_intersection_hours": len(combined),
        "period_app_usage_start": row.get("commheat start"),
        "period_app_usage_end": row.get("commheat end"),
        "mse_predicted_vs_avg": mse_predicted_avg,
        "mse_predicted_vs_max": mse_predicted_max,
        "xlsx_output": str(xlsx_path),
        "png_output": str(png_path),
    })

    # return first result or none
    return results[0] if results else None


def compute_period_intersection_means_for_row(row: pd.Series, hourly: pd.DataFrame, period_intersection_start: pd.Timestamp,
                                   period_intersection_end: pd.Timestamp, arch: str) -> Optional[Dict]:
    """Compute period_intersection means with latest EP files"""
    # extract intersection period from hobo data
    hobo_slice = hourly.loc[period_intersection_start:period_intersection_end]
    if hobo_slice.empty:
        return None

    # calculate mean of average and max temperatures
    hobo_mean_value = ((hobo_slice["actual_average_temperature"] + hobo_slice["actual_max_temperature"]) / 2.0).mean()

    # collect simulation means for each archetype
    sim_means = {}
    for archetype in [a.strip() for a in arch.split(",") if a.strip()]:
        # load archetype series for both ac and noac
        sim_series = load_archetype_series(
            archetype,
            hourly.index.min().year,
            [f"{archetype}*ac_out*.xlsx", f"{archetype}*ac_out*.csv",
             f"{archetype}*noac_out*.xlsx", f"{archetype}*noac_out*.csv"],
            period_intersection_start,
            period_intersection_end
        )

        # calculate mean if series exists
        if sim_series is not None:
            sim_means[archetype] = sim_series.mean()

    if not sim_means:
        return None

    # calculate overall simulation mean across all archetypes
    sim_overall_mean = float(np.mean(list(sim_means.values())))
    # calculate final combined mean
    final_mean = (hobo_mean_value + sim_overall_mean) / 2.0

    # return summary statistics
    return {
        "address": row["address"],
        "housetype": row["housetype"],
        "period_intersection_start": period_intersection_start,
        "period_intersection_end": period_intersection_end,
        "n_hours": len(hobo_slice),
        "hobologger_mean_C": hobo_mean_value,
        "sim_overall_mean_C": sim_overall_mean,
        "final_combined_mean_C": final_mean,
        "archetypes": arch,
    }


def plot_ac_noac_for_row(row: pd.Series, period_intersection_start: pd.Timestamp, period_intersection_end: pd.Timestamp, arch: str) -> None:
    """Generate AC vs No-AC plots"""
    # process each archetype
    for archetype in [a.strip() for a in arch.split(",") if a.strip()]:
        # load ac and noac series
        data = load_ac_noac_series(archetype, period_intersection_start.year, period_intersection_start, period_intersection_end)

        # extract ac and noac data
        ac = data.get("AC")
        noac = data.get("NoAC")

        # skip if either series is missing
        if ac is None or noac is None:
            logger.warning(f"Missing AC/No-AC data for {archetype}")
            continue

        # create output filename in ac_noac subdirectory
        out_name = f"{row['clean_address']}_{archetype}_AC_NoAC_PERIOD_INTERSECTION.png"
        out_path = config.ac_noac_dir / out_name
        
        # use plottingmanager to create ac vs noac plot
        config.plotter.plot_ac_vs_noac(ac, noac, f"{row['address']} - {archetype}", out_path)


def run_analysis(mapping_df: pd.DataFrame, analysis_type: str, desc: str) -> List[Dict]:
    """Run analysis across all mapping rows"""
    # collect results from all rows
    results = []

    # iterate through mapping dataframe with progress bar
    for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc=desc):
        # process single row
        result = process_mapping_row(row, analysis_type)
        if result:
            # handle list or single result
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    """Main execution"""
    logger.info(color_text("=== Starting CommHEAT Data Comparison ===\n", "96"))

    # load sensor mapping file
    mapping = load_mapping()

    # find all hobo sensor files
    hobo_files = [f for f in config.hobo_dir.glob("*.xlsx")
                  if "sensor contact" not in f.name.lower() and not f.name.startswith("~$")]

    # process all hobo files
    logger.info(color_text("\n=== Processing HOBO Sensor Data ===\n", "96"))
    logger.info("Converting raw HOBO temperature data to hourly averages and saving processed files\n")
    summary = [res for f in tqdm(hobo_files, desc="Processing HOBO files")
               if (res := process_hobo_file(f, mapping))]

    # save hobo processing summary
    if summary:
        df = pd.DataFrame(summary)
        save_path = config.output_dir / "HoboHouseIndex.xlsx"

        try:
            # save to excel
            df.to_excel(save_path, index=False)
            logger.info(f"Saved HOBO index to: {save_path}")
        except PermissionError:
            # save backup if file is locked
            backup = config.output_dir / "HoboHouseIndex_backup.xlsx"
            df.to_excel(backup, index=False)
            logger.info(f"Saved backup: {backup}")

    # run mse analysis
    logger.info(color_text("\n=== Computing MSE ===\n", "96"))
    logger.info("Calculating Mean Squared Error between HOBO measurements and averaged archetype predictions for each address covering the period of intersection\n")
    mse_results = run_analysis(mapping, 'mse', "Computing MSE")

    # save mse results
    if mse_results:
        pd.DataFrame(mse_results).to_excel(config.comparison_dir / "Intersection_MSE_Summary.xlsx", index=False)

    # run period intersection means analysis
    logger.info(color_text("\n=== Computing Period Intersection Means ===\n", "96"))
    logger.info("Calculating average temperatures during overlapping HOBO logger and app usage periods\n")
    period_intersection_results = run_analysis(mapping, 'period_intersection_means', "Computing Period Intersection Means")

    # save period intersection results
    if period_intersection_results:
        pd.DataFrame(period_intersection_results).to_excel(config.output_dir / "Period_Intersection_Mean_Summary.xlsx", index=False)

    # generate ac vs no-ac plots
    logger.info(color_text("\n=== Plotting AC vs No-AC ===\n", "96"))
    logger.info("Generating comparison plots between air-conditioned and non-air-conditioned scenarios\n")
    run_analysis(mapping, 'ac_noac_plots', "AC/No-AC Plots")

    # generate heat event plots (individual archetypes)
    logger.info(color_text("\n=== Plotting Heat Events (Individual Archetypes) ===\n", "96"))
    logger.info("Creating separate plots for each archetype during heat events and baseline periods\n")
    run_analysis(mapping, 'heat_events', "Heat Event Plots")
    
    # generate heat event plots (averaged archetypes)
    logger.info(color_text("\n=== Plotting Heat Events (Averaged Archetypes) ===\n", "96"))
    logger.info("Generating averaged archetype plots showing combined predictions for heat events\n")
    run_analysis(mapping, 'heat_events_averaged', "Averaged Heat Event Plots")
    
    # generate period intersection plots
    logger.info(color_text("\n=== Plotting Period Intersection ===\n", "96"))
    logger.info("Creating full-period comparison plots for HOBO data vs averaged archetype predictions\n")
    run_analysis(mapping, 'period_intersection_plots', "Period Intersection Plots")

    logger.info(color_text("\n=== DONE ===", "92"))



if __name__ == "__main__":
    main()
