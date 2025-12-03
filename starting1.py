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
import matplotlib.pyplot as plt
import logging
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache
from tqdm import tqdm
import warnings
import sys
import io

warnings.filterwarnings('ignore')

# ============================================================
# LOGGING SETUP
# ============================================================

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    """Centralized configuration management"""
    
    hobo_dir: Path = Path(r"C:\Users\Ethan\Downloads\2025SCC_OnsetHobo_InHome_Dataloggers\2025SCC_OnsetHobo_InHome_Dataloggers")
    mapping_file: Path = Path(r"C:\Users\Ethan\Downloads\2025SCC_OnsetHobo_InHome_Dataloggers\2025SCC_OnsetHobo_InHome_Dataloggers\Sensor Contact_101325_PickUP.xlsx")
    simulation_dir: Path = Path(r"C:\Users\Ethan\Downloads\2025_Summer_EPSim_HLData_Analysis\2025_Summer_EPSim_HLData_Analysis")
    latest_ep_dir: Path = Path(r"C:\Users\Ethan\Downloads\Latest_EP_Output_Files\Latest_EP_Output_Files")
    output_dir: Path = Path(r"C:\Users\Ethan\OneDrive - Iowa State University\Desktop\CommHEAT Output")
    
    hobo_time_col: str = "Date-Time (CDT)"
    hobo_temp_col: str = "Temperature , °F"
    hobo_rh_col: str = "RH , %"
    
    target_columns: List[str] = None
    
    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.comparison_dir = self.output_dir / "comparisons"
        self.comparison_dir.mkdir(exist_ok=True, parents=True)
        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True, parents=True)
        
        if self.target_columns is None:
            self.target_columns = [
                "FIRSTFLOOR_0:Zone Mean Air Temperature [C](Hourly)",
                "FIRSTFLOOR_1:Zone Mean Air Temperature [C](Hourly)",
                "FIRSTFLOOR_2:Zone Mean Air Temperature [C](Hourly)",
                "HOUSE_0:Zone Mean Air Temperature [C](Hourly)"
            ]

config = Config()


# ============================================================
# PATTERNS AND STYLING
# ============================================================

PATTERNS = {
    'sensor_id': re.compile(r"(\d+)"),
    'archetype_temp': re.compile(r'^(FIRSTFLOOR_\d+|HOUSE_\d+):Zone Mean Air Temperature \[C\]\(Hourly\)$', re.IGNORECASE),
    'ordinal': re.compile(r"(\d+)(st|nd|rd|th)"),
    'whitespace': re.compile(r"\s+"),
    'address_clean': re.compile(r"[^A-Za-z0-9]")
}

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

COLUMN_TYPES = {
    'time': ['date', 'time', 'timestamp'],
    'temp': ['temperature', 'zone mean air temperature'],
    'rh': ['rh', 'relative humidity']
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def setup_plot_style():
    """Apply consistent plot styling"""
    plt.rcParams.update(PLOT_STYLE)


def color_text(txt: str, code: str) -> str:
    """Apply ANSI color code to text"""
    return f"\033[{code}m{txt}\033[0m"


def clean_text(s: Union[str, any]) -> str:
    """Remove ordinal suffixes and normalize whitespace"""
    if not isinstance(s, str):
        return str(s)
    s = s.strip().replace("\n", " ").replace("\r", " ")
    s = PATTERNS['ordinal'].sub(r"\1", s)
    return PATTERNS['whitespace'].sub(" ", s)


def parse_date(date_str: any, target_year: Optional[int] = None, ep_format: bool = False) -> pd.Timestamp:
    """
    Universal date parser for both HOBO and EnergyPlus formats.
    
    Args:
        date_str: Date string to parse
        target_year: Year to assign (for EnergyPlus dates)
        ep_format: If True, parse as EnergyPlus format (MM/DD HH:MM:SS)
    """
    if pd.isna(date_str):
        return pd.NaT
    
    s = clean_text(date_str)
    
    # EnergyPlus format: "08/27 13:00:00"
    if ep_format:
        try:
            s = " ".join(s.split())
            dt = pd.to_datetime(s, format="%m/%d %H:%M:%S", errors="coerce")
            if pd.notna(dt) and target_year:
                return dt.replace(year=target_year)
            return dt
        except:
            return pd.NaT
    
    # Standard formats
    formats = [
        "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%m-%d-%Y", "%m-%d-%y",
        "%b %d %Y", "%b %d, %Y", "%B %d %Y", "%B %d, %Y",
        "%b %d %I:%M%p", "%B %d %I:%M%p"
    ]
    
    for fmt in formats:
        try:
            return pd.to_datetime(s, format=fmt)
        except:
            continue
    
    return pd.to_datetime(s, errors='coerce')


def validate_overlap(start: pd.Timestamp, end: pd.Timestamp, min_hours: int = 1) -> Tuple[bool, str]:
    """Validate overlap period meets minimum requirements"""
    if pd.isna(start) or pd.isna(end):
        return False, "Invalid timestamps (NaT)"
    if start >= end:
        return False, f"Start must be before end"
    
    hours = (end - start).total_seconds() / 3600
    if hours < min_hours:
        return False, f"Period ({hours:.1f}h) < minimum ({min_hours}h)"
    
    return True, f"Valid: {hours:.1f} hours"


def convert_to_celsius(series: pd.Series) -> pd.Series:
    """Convert Fahrenheit to Celsius if needed"""
    series = pd.to_numeric(series, errors='coerce')
    return np.where(series > 60, (series - 32) * 5 / 9, series)


# ============================================================
# FILE LOADING
# ============================================================

def load_dataframe(filepath: Path, max_skiprows: int = 3) -> Optional[pd.DataFrame]:
    """Load CSV or Excel with automatic skiprows detection"""
    for skiprows in range(max_skiprows + 1):
        try:
            df = pd.read_excel(filepath, skiprows=skiprows) if filepath.suffix.lower() == ".xlsx" \
                else pd.read_csv(filepath, skiprows=skiprows)
            
            df.columns = df.columns.str.strip()
            
            # Validate: must have time column and >100 rows
            time_cols = [c for c in df.columns if any(t in c.lower() for t in COLUMN_TYPES['time'])]
            if time_cols and len(df) > 100:
                if skiprows > 0:
                    logger.debug(f"Loaded {filepath.name} with {skiprows} rows skipped")
                return df
        except Exception as e:
            logger.debug(f"Skip {skiprows} failed for {filepath.name}: {e}")
    
    logger.warning(f"Could not load {filepath.name}")
    return None


def find_column(df: pd.DataFrame, col_type: str) -> Optional[str]:
    """Find column by type using COLUMN_TYPES mapping"""
    if col_type == 'temp':
        # Strategy 1: Pattern match
        cols = [c for c in df.columns if PATTERNS['archetype_temp'].match(c)]
        if cols:
            return cols[0]
        
        # Strategy 2: Contains keywords
        cols = [c for c in df.columns 
                if all(k in c.lower() for k in ['zone mean air temperature', '[c]'])]
        if cols:
            return cols[0]
        
        # Strategy 3: Target columns
        cols = [c for c in config.target_columns if c in df.columns]
        if cols:
            return cols[0]
        
        # Strategy 4: Any temp with [C]
        cols = [c for c in df.columns if 'temperature' in c.lower() and '[c]' in c.lower()]
        return cols[0] if cols else None
    
    # Generic search
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
        df = pd.read_excel(config.mapping_file, header=2)
        df.columns = df.columns.str.strip().str.lower()

        df = df.rename(columns={"home address": "address", "sensor #": "sensor_id"})
        df = df.dropna(subset=["sensor_id", "address"])
        df["sensor_id"] = df["sensor_id"].astype(str)

        # Vectorized operations
        df["housetype"] = df["address"].str.contains("apt|apartment", case=False, regex=True).map({True: "Apt", False: "Ind"})
        df["clean_address"] = df["address"].apply(lambda x: PATTERNS['address_clean'].sub("", x))
        df["outfile"] = df["housetype"] + "_" + df["clean_address"] + ".xlsx"

        # Parse dates
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
    sensor_id = PATTERNS['sensor_id'].match(file_path.name)
    if not sensor_id:
        return None
    
    sensor_id = sensor_id.group(1)
    row_df = mapping_df[mapping_df["sensor_id"] == sensor_id]
    if row_df.empty:
        return None

    row = row_df.iloc[0]
    outfile = config.output_dir / row["outfile"]

    try:
        df = pd.read_excel(file_path)
        df[config.hobo_time_col] = pd.to_datetime(df[config.hobo_time_col])
        df = df.set_index(config.hobo_time_col).sort_index()
        df["temp_C"] = convert_to_celsius(df[config.hobo_temp_col])

        hourly = df.resample("h").agg({
            "temp_C": ["mean", "max"],
            config.hobo_rh_col: "mean"
        })

        hourly.columns = ["actual_average_temperature", "actual_max_temperature", "average_relative_humidity"]
        hourly.to_excel(outfile, index_label="timestamp")

        logger.info(f"Processed {file_path.name}: {hourly.index.min()} to {hourly.index.max()}")

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
    return load_simulation(prefix, target_year, Path(sim_dir_str))


def load_simulation(prefix: str, target_year: int = 2025, sim_dir: Path = None) -> Optional[pd.DataFrame]:
    """Load simulation data for archetype"""
    if sim_dir is None:
        sim_dir = config.simulation_dir
        
    matches = list(sim_dir.glob(f"{prefix}*.csv")) or list(sim_dir.glob(f"{prefix}*.xlsx"))
    if not matches:
        logger.warning(f"No simulation file for: {prefix}")
        return None

    try:
        df = load_dataframe(matches[0])
        if df is None:
            return None

        time_col = find_column(df, 'time')
        if not time_col:
            return None

        df[time_col] = pd.to_datetime(df[time_col], format="mixed", errors="coerce")
        df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
        df.index = df.index.map(lambda ts: ts.replace(year=target_year) if pd.notna(ts) else ts)

        temp_col = find_column(df, 'temp')
        if not temp_col:
            return None

        df = df[[temp_col]]
        df.columns = ["archetype_internal_temperature"]
        return df

    except Exception as e:
        logger.error(f"Error loading simulation: {e}", exc_info=True)
        return None


def load_archetype_series(
    prefix: str,
    target_year: int,
    patterns: List[str],
    overlap_start: Optional[pd.Timestamp] = None,
    overlap_end: Optional[pd.Timestamp] = None
) -> Optional[pd.Series]:
    """
    Unified loader for EP output files (AC/NoAC).
    
    Args:
        prefix: Archetype prefix
        target_year: Year to assign
        patterns: File patterns to search
        overlap_start/end: Optional time window
    """
    files = []
    for pat in patterns:
        files.extend(list(config.latest_ep_dir.glob(pat)))
    
    if not files:
        return None

    series_list = []

    for f in files:
        try:
            df = load_dataframe(f, max_skiprows=3)
            if df is None:
                continue

            time_col = find_column(df, 'time')
            if not time_col:
                continue

            df[time_col] = df[time_col].apply(lambda x: parse_date(x, target_year, ep_format=True))
            df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

            temp_col = find_column(df, 'temp')
            if not temp_col:
                continue

            temp_series = pd.to_numeric(df[temp_col], errors='coerce').dropna()

            # Filter to overlap if specified
            if overlap_start and overlap_end:
                temp_series = temp_series.loc[overlap_start:overlap_end]
                
                # Relaxed matching fallback
                if temp_series.empty:
                    full_series = pd.to_numeric(df[temp_col], errors='coerce')
                    mask = (
                        (full_series.index.month >= overlap_start.month) &
                        (full_series.index.month <= overlap_end.month) &
                        (full_series.index.day >= overlap_start.day - 1) &
                        (full_series.index.day <= overlap_end.day + 1)
                    )
                    temp_series = full_series[mask].dropna()

            if not temp_series.empty:
                series_list.append(temp_series)

        except Exception as e:
            logger.debug(f"Error loading {f.name}: {e}")

    if not series_list:
        return None

    # Average all series
    df_all = pd.concat(series_list, axis=1)
    return df_all.mean(axis=1)


def load_ac_noac_series(
    prefix: str,
    target_year: int = 2025,
    overlap_start: Optional[pd.Timestamp] = None,
    overlap_end: Optional[pd.Timestamp] = None
) -> Dict[str, Optional[pd.Series]]:
    """Load AC and No-AC series separately"""
    ac_patterns = [f"{prefix}*ac_out*.xlsx", f"{prefix}*ac_out*.csv"]
    noac_patterns = [f"{prefix}*noac_out*.xlsx", f"{prefix}*noac_out*.csv"]

    return {
        "AC": load_archetype_series(prefix, target_year, ac_patterns, overlap_start, overlap_end),
        "NoAC": load_archetype_series(prefix, target_year, noac_patterns, overlap_start, overlap_end)
    }


# ============================================================
# PLOTTING
# ============================================================

def create_plot(
    data: pd.DataFrame,
    series_config: Dict[str, Dict],
    title: str,
    ylabel: str,
    output_path: Path,
    vspan: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ylim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Unified plotting function.
    
    Args:
        data: DataFrame with time series
        series_config: Dict of {col_name: {label, color, linestyle, etc}}
        title: Plot title
        ylabel: Y-axis label
        output_path: Save path
        vspan: Optional highlight period
        ylim: Optional y-axis limits
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for col, style in series_config.items():
        if col in data.columns:
            ax.plot(data.index, data[col], **style)
    
    if vspan:
        ax.axvspan(*vspan, color='yellow', alpha=0.2, label="Intersection")
    
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if ylim:
        ax.set_ylim(ylim)
    
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ac_vs_noac(ac_series: pd.Series, noac_series: pd.Series, title_prefix: str, output_path: Path) -> None:
    """Plot AC vs No-AC comparison"""
    df = pd.concat([ac_series.rename("AC"), noac_series.rename("No AC")], axis=1).dropna()
    
    if df.empty:
        logger.warning(f"Empty AC/No-AC data for {title_prefix}")
        return

    mse = ((df["AC"] - df["No AC"]) ** 2).mean()

    series_config = {
        "AC": {"label": "AC", "linewidth": 2, "color": "blue"},
        "No AC": {"label": "No AC", "linewidth": 2, "linestyle": "--", "color": "red"}
    }

    create_plot(
        df,
        series_config,
        f"{title_prefix} - Zone Air Temperature\nMSE (AC vs No AC): {mse:.3f} C^2",
        "Temp [C]",
        output_path
    )

    logger.info(f"Saved AC vs No-AC plot: {output_path} (MSE: {mse:.3f})")


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def process_mapping_row(row: pd.Series, analysis_type: str) -> Optional[Dict]:
    """
    Unified processor for mapping rows.
    
    Args:
        row: Mapping DataFrame row
        analysis_type: 'mse', 'overlap_means', or 'ac_noac_plots'
    """
    arch = row.get("archtypes used")
    if not isinstance(arch, str):
        return None

    hobo_file = config.output_dir / row["outfile"]
    if not hobo_file.exists():
        return None

    try:
        hourly = pd.read_excel(hobo_file, index_col="timestamp")
        hourly.index = pd.to_datetime(hourly.index)

        app_start = row.get("commheat start")
        app_end = row.get("commheat end")

        if pd.isna(app_start) or pd.isna(app_end):
            return None

        overlap_start = max(hourly.index.min(), app_start)
        overlap_end = min(hourly.index.max(), app_end)

        is_valid, msg = validate_overlap(overlap_start, overlap_end)
        if not is_valid:
            logger.warning(f"Invalid overlap for {row['address']}: {msg}")
            return None

        # Route to appropriate analysis
        if analysis_type == 'mse':
            return compute_mse_for_row(row, hourly, overlap_start, overlap_end, arch)
        elif analysis_type == 'overlap_means':
            return compute_overlap_means_for_row(row, hourly, overlap_start, overlap_end, arch)
        elif analysis_type == 'ac_noac_plots':
            plot_ac_noac_for_row(row, overlap_start, overlap_end, arch)
            return None

    except Exception as e:
        logger.error(f"Error processing {row['address']}: {e}", exc_info=True)
        return None


def compute_mse_for_row(row: pd.Series, hourly: pd.DataFrame, obs_start: pd.Timestamp, obs_end: pd.Timestamp, arch: str) -> Optional[Dict]:
    """Compute MSE between HOBO and simulation"""
    hobo_window = hourly.loc[obs_start:obs_end]
    results = []

    for archetype in [a.strip() for a in arch.split(",")]:
        sim_df = load_simulation(archetype, target_year=hourly.index.min().year, sim_dir=config.simulation_dir)
        
        if sim_df is None:
            continue

        actual_start = max(obs_start, sim_df.index.min())
        actual_end = min(obs_end, sim_df.index.max())

        if actual_start >= actual_end:
            continue

        combined = hobo_window.loc[actual_start:actual_end].join(sim_df.loc[actual_start:actual_end], how="inner")
        
        if combined.empty:
            continue

        mse_avg = ((combined["archetype_internal_temperature"] - combined["actual_average_temperature"]) ** 2).mean()
        mse_max = ((combined["archetype_internal_temperature"] - combined["actual_max_temperature"]) ** 2).mean()

        label = f"{row['housetype']}_{row['clean_address']}"
        
        # Save comparison data
        csv_path = config.comparison_dir / f"{label}_{archetype}_comparison.csv"
        combined.to_csv(csv_path)
        combined.to_excel(csv_path.with_suffix('.xlsx'))

        # Temperature plot
        plot_prefix = config.plot_dir / f"{label}_{archetype}"
        
        series_config = {
            "archetype_internal_temperature": {"label": "Sim", "linewidth": 2, "color": "blue"},
            "actual_average_temperature": {"label": "HOBO avg", "alpha": 0.7, "color": "green"},
            "actual_max_temperature": {"label": "HOBO max", "alpha": 0.7, "color": "orange"}
        }
        
        title = f"{row['address']} - {archetype}\nMSE (Sim vs HOBO avg): {mse_avg:.3f} C^2  |  MSE (Sim vs HOBO max): {mse_max:.3f} C^2"
        
        create_plot(combined, series_config, title, "Temperature [C]", 
                   Path(str(plot_prefix) + "_intersection.png"), vspan=(actual_start, actual_end))

        # Humidity plot
        if "average_relative_humidity" in combined.columns:
            humidity_config = {
                "average_relative_humidity": {"label": "Relative Humidity", "linewidth": 2, "color": "steelblue"}
            }
            create_plot(combined, humidity_config, f"{row['address']} - {archetype}\nRelative Humidity",
                       "Relative Humidity [%]", Path(str(plot_prefix) + "_humidity.png"), 
                       vspan=(actual_start, actual_end), ylim=(0, 100))

        results.append({
            "address": row["address"],
            "archetype": archetype,
            "period_intersection_start": actual_start,
            "period_intersection_end": actual_end,
            "period_intersection_hours": len(combined),
            "period_app_usage_start": row.get("commheat start"),
            "period_app_usage_end": row.get("commheat end"),
            "mse_avg": mse_avg,
            "mse_max": mse_max,
            "csv_output": str(csv_path),
            "xlsx_output": str(csv_path.with_suffix('.xlsx')),
        })

    return results[0] if results else None


def compute_overlap_means_for_row(row: pd.Series, hourly: pd.DataFrame, overlap_start: pd.Timestamp, 
                                   overlap_end: pd.Timestamp, arch: str) -> Optional[Dict]:
    """Compute overlap means with latest EP files"""
    hobo_slice = hourly.loc[overlap_start:overlap_end]
    if hobo_slice.empty:
        return None

    hobo_mean_value = ((hobo_slice["actual_average_temperature"] + hobo_slice["actual_max_temperature"]) / 2.0).mean()

    sim_means = {}
    for archetype in [a.strip() for a in arch.split(",") if a.strip()]:
        sim_series = load_archetype_series(
            archetype,
            hourly.index.min().year,
            [f"{archetype}*ac_out*.xlsx", f"{archetype}*ac_out*.csv", 
             f"{archetype}*noac_out*.xlsx", f"{archetype}*noac_out*.csv"],
            overlap_start,
            overlap_end
        )

        if sim_series is not None:
            sim_means[archetype] = sim_series.mean()

    if not sim_means:
        return None

    sim_overall_mean = float(np.mean(list(sim_means.values())))
    final_mean = (hobo_mean_value + sim_overall_mean) / 2.0

    return {
        "address": row["address"],
        "housetype": row["housetype"],
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "n_hours": len(hobo_slice),
        "hobologger_mean_C": hobo_mean_value,
        "sim_overall_mean_C": sim_overall_mean,
        "final_combined_mean_C": final_mean,
        "archetypes": arch,
    }


def plot_ac_noac_for_row(row: pd.Series, overlap_start: pd.Timestamp, overlap_end: pd.Timestamp, arch: str) -> None:
    """Generate AC vs No-AC plots"""
    for archetype in [a.strip() for a in arch.split(",") if a.strip()]:
        data = load_ac_noac_series(archetype, overlap_start.year, overlap_start, overlap_end)

        ac = data.get("AC")
        noac = data.get("NoAC")

        if ac is None or noac is None:
            logger.warning(f"Missing AC/No-AC data for {archetype}")
            continue

        out_name = f"{row['clean_address']}_{archetype}_AC_NoAC_OVERLAP.png"
        out_path = config.plot_dir / out_name
        plot_ac_vs_noac(ac, noac, f"{row['address']} - {archetype}", out_path)


def run_analysis(mapping_df: pd.DataFrame, analysis_type: str, desc: str) -> List[Dict]:
    """Run analysis across all mapping rows"""
    results = []
    
    for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc=desc):
        result = process_mapping_row(row, analysis_type)
        if result:
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
    logger.info(color_text("=== Starting CommHEAT Analysis ===\n", "96"))
    
    setup_plot_style()
    mapping = load_mapping()

    # Process HOBO files
    hobo_files = [f for f in config.hobo_dir.glob("*.xlsx") 
                  if "sensor contact" not in f.name.lower() and not f.name.startswith("~$")]
    
    summary = [res for f in tqdm(hobo_files, desc="Processing HOBO files") 
               if (res := process_hobo_file(f, mapping))]

    if summary:
        df = pd.DataFrame(summary)
        save_path = config.output_dir / "HoboHouseIndex.xlsx"
        
        try:
            df.to_excel(save_path, index=False)
            logger.info(f"Saved HOBO index to: {save_path}")
        except PermissionError:
            backup = config.output_dir / "HoboHouseIndex_backup.xlsx"
            df.to_excel(backup, index=False)
            logger.info(f"Saved backup: {backup}")

    # MSE Analysis
    logger.info(color_text("\n=== Computing MSE ===\n", "96"))
    mse_results = run_analysis(mapping, 'mse', "Computing MSE")
    
    if mse_results:
        pd.DataFrame(mse_results).to_excel(config.output_dir / "Intersection_MSE_Summary.xlsx", index=False)

    # Overlap Means
    logger.info(color_text("\n=== Computing Overlap Means ===\n", "96"))
    overlap_results = run_analysis(mapping, 'overlap_means', "Computing Overlap Means")
    
    if overlap_results:
        pd.DataFrame(overlap_results).to_excel(config.output_dir / "Overlap_Mean_Summary.xlsx", index=False)

    # AC vs No-AC Plots
    logger.info(color_text("\n=== Plotting AC vs No-AC ===\n", "96"))
    run_analysis(mapping, 'ac_noac_plots', "AC/No-AC Plots")

    logger.info(color_text("\n=== DONE ===", "92"))


if __name__ == "__main__":
    main()