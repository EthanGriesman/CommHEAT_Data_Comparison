"""
CommHEAT Data Analysis Pipeline - Optimized Version
====================================================

This script processes HOBO temperature sensor data, compares it with EnergyPlus
simulation outputs, and generates comparison plots and statistics.

Usage:
    python starting1.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple, Union
from functools import lru_cache
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Import configuration and utilities
from config import (
    Config, 
    Patterns, 
    PlotStyle, 
    ColumnTypes, 
    HeatEvents,
    setup_logging,
    PATTERNS,  # For backward compatibility
    PLOT_STYLE,
    COLUMN_TYPES,
    HEAT_EVENTS
)

# Import the plotting manager class
from plotting_manager import PlottingManager

# Import the data loader class
from data_loader import DataLoader

# Suppress pandas and numpy warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = setup_logging()

# Instantiate global config
config = Config()

# ============================================================
# GLOBAL CACHE
# ============================================================

class ArchetypeCache:
    """Global cache for archetype simulation data"""
    def __init__(self):
        self._cache = {}
        self._loaded_files = set()
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        return self._cache.get(key)
    
    def set(self, key: str, value: pd.DataFrame):
        self._cache[key] = value
    
    def has(self, key: str) -> bool:
        return key in self._cache
    
    def clear(self):
        self._cache.clear()
        self._loaded_files.clear()
    
    def size(self) -> int:
        return len(self._cache)

# Global archetype cache instance
ARCHETYPE_CACHE = ArchetypeCache()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def color_text(txt: str, code: str) -> str:
    """Apply ANSI color code to text"""
    return f"\033[{code}m{txt}\033[0m"


def clean_text_vectorized(series: pd.Series) -> pd.Series:
    """Vectorized text cleaning for pandas Series"""
    s = series.astype(str).str.strip()
    s = s.str.replace("\n", " ", regex=False).str.replace("\r", " ", regex=False)
    s = s.str.replace(r"(\d+)(st|nd|rd|th)", r"\1", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def clean_text(s: Union[str, any]) -> str:
    """Remove ordinal suffixes and normalize whitespace"""
    if not isinstance(s, str):
        return str(s)
    s = s.strip().replace("\n", " ").replace("\r", " ")
    s = Patterns.ORDINAL.sub(r"\1", s)
    return Patterns.WHITESPACE.sub(" ", s)


def parse_energyplus_datetime_vectorized(series: pd.Series, target_year: int) -> pd.Series:
    """
    Vectorized datetime parser for EnergyPlus format.
    Handles 'MM/DD HH:MM:SS' format with 24:00:00 edge case.
    """
    s = series.astype(str).str.strip()
    
    mask_24 = s.str.contains("24:", na=False)
    s_clean = s.str.replace("24:", "00:", regex=False)
    
    dt = pd.to_datetime(
        str(target_year) + "-" + s_clean, 
        format="%Y-%m/%d %H:%M:%S", 
        errors="coerce"
    )
    
    dt = dt.where(~mask_24, dt + pd.Timedelta(days=1))
    
    return dt


def validate_period_intersection(start: pd.Timestamp, end: pd.Timestamp, min_hours: int = 1) -> Tuple[bool, str]:
    """Validate period intersection period meets minimum requirements"""
    if pd.isna(start) or pd.isna(end):
        return False, "Invalid timestamps (NaT)"
    if start >= end:
        return False, "Start must be before end"

    hours = (end - start).total_seconds() / 3600
    if hours < min_hours:
        return False, f"Period ({hours:.1f}h) < minimum ({min_hours}h)"

    return True, f"Valid: {hours:.1f} hours"


def convert_to_celsius_vectorized(series: pd.Series) -> pd.Series:
    """Vectorized Fahrenheit to Celsius conversion"""
    values = pd.to_numeric(series, errors='coerce').values
    return pd.Series(
        np.where(values > 60, (values - 32) * 5 / 9, values),
        index=series.index
    )


def calculate_mse_numpy(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Calculate MSE using NumPy for better performance"""
    return np.mean(np.square(predicted - actual))


# ============================================================
# DATA LOADING - Now using DataLoader class
# ============================================================

def load_mapping() -> pd.DataFrame:
    """Load and process sensor mapping file - delegates to DataLoader"""
    return config.data_loader.load_mapping()


def process_hobo_file(file_path: Path, mapping_df: pd.DataFrame) -> Optional[Dict]:
    """Process HOBO sensor file with vectorized operations"""
    sensor_id = PATTERNS['sensor_id'].match(file_path.name)
    if not sensor_id:
        return None

    sensor_id = sensor_id.group(1)
    row_df = mapping_df[mapping_df["sensor_id"] == sensor_id]
    if row_df.empty:
        return None

    row = row_df.iloc[0]
    outfile = config.hobo_output_dir / row["outfile"]

    try:
        # Use DataLoader to load file
        df = config.data_loader.load_dataframe(file_path)
        if df is None:
            return None
            
        df[config.hobo_time_col] = pd.to_datetime(df[config.hobo_time_col])
        df = df.set_index(config.hobo_time_col).sort_index()
        
        df["temp_C"] = convert_to_celsius_vectorized(df[config.hobo_temp_col])

        hourly = df.resample("h").agg({
            "temp_C": ["mean", "max"],
            config.hobo_rh_col: "mean"
        })

        hourly.columns = ["actual_average_temperature", "actual_max_temperature", "average_relative_humidity"]
        hourly.to_excel(outfile, index_label="timestamp", engine='openpyxl')

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


def load_simulation_optimized(prefix: str, target_year: int = 2025, sim_dir: Path = None) -> Optional[pd.DataFrame]:
    """Load simulation data with caching and vectorized datetime parsing"""
    cache_key = f"{prefix}_{target_year}"
    if ARCHETYPE_CACHE.has(cache_key):
        return ARCHETYPE_CACHE.get(cache_key)
    
    if sim_dir is None:
        sim_dir = config.latest_ep_dir

    matches = list(sim_dir.glob(f"{prefix}*.csv")) or list(sim_dir.glob(f"{prefix}*.xlsx"))
    if not matches:
        logger.debug(f"No simulation file for: {prefix}")
        return None

    try:
        # Use DataLoader to load file
        df = config.data_loader.load_dataframe(matches[0])
        if df is None:
            return None

        # Use DataLoader to find columns
        time_col = config.data_loader.find_column(df, 'time')
        if not time_col:
            return None

        df[time_col] = parse_energyplus_datetime_vectorized(df[time_col], target_year)
        df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

        temp_col = config.data_loader.find_column(df, 'temp')
        if not temp_col:
            return None

        out = df[[temp_col]].copy()
        out.columns = ["archetype_internal_temperature"]
        out = out.resample("h").mean()

        ARCHETYPE_CACHE.set(cache_key, out)

        return out

    except Exception as e:
        logger.error(f"Error loading simulation: {e}", exc_info=True)
        return None


def preload_archetypes(mapping_df: pd.DataFrame, target_year: int = 2025) -> None:
    """Preload all unique archetypes into cache"""
    unique_archetypes = set()
    for arch_str in mapping_df["archtypes used"].dropna():
        if isinstance(arch_str, str):
            unique_archetypes.update(a.strip() for a in arch_str.split(","))
    
    logger.info(f"Preloading {len(unique_archetypes)} unique archetypes...")
    
    with tqdm(total=len(unique_archetypes), desc="Preloading archetypes") as pbar:
        for archetype in unique_archetypes:
            load_simulation_optimized(archetype, target_year)
            pbar.update(1)
    
    logger.info(f"Preloaded {ARCHETYPE_CACHE.size()} archetypes into cache")


def load_archetype_series_optimized(
    prefix: str,
    target_year: int,
    patterns: List[str],
    period_intersection_start: Optional[pd.Timestamp] = None,
    period_intersection_end: Optional[pd.Timestamp] = None
) -> Optional[pd.Series]:
    """Optimized loader for EP output files with caching"""
    files = []
    for pat in patterns:
        files.extend(list(config.latest_ep_dir.glob(pat)))

    if not files:
        return None

    series_list = []

    for f in files:
        try:
            # Use DataLoader
            df = config.data_loader.load_dataframe(f, max_skiprows=3)
            if df is None:
                continue

            time_col = config.data_loader.find_column(df, 'time')
            if not time_col:
                continue

            df[time_col] = parse_energyplus_datetime_vectorized(df[time_col], target_year)
            df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

            temp_col = config.data_loader.find_column(df, 'temp')
            if not temp_col:
                continue

            temp_series = pd.to_numeric(df[temp_col], errors='coerce').dropna()
            temp_series = temp_series.resample("h").mean()

            if period_intersection_start and period_intersection_end:
                temp_series = temp_series.loc[period_intersection_start:period_intersection_end]

                if temp_series.empty:
                    full_series = pd.to_numeric(df[temp_col], errors='coerce')
                    mask = (
                        (full_series.index.month >= period_intersection_start.month) &
                        (full_series.index.month <= period_intersection_end.month) &
                        (full_series.index.day >= period_intersection_start.day - 1) &
                        (full_series.index.day <= period_intersection_end.day + 1)
                    )
                    temp_series = full_series[mask].dropna()

            if not temp_series.empty:
                series_list.append(temp_series)

        except Exception as e:
            logger.debug(f"Error loading {f.name}: {e}")

    if not series_list:
        return None

    df_all = pd.concat(series_list, axis=1)
    return df_all.mean(axis=1)


def load_ac_noac_series(
    prefix: str,
    target_year: int = 2025,
    period_intersection_start: Optional[pd.Timestamp] = None,
    period_intersection_end: Optional[pd.Timestamp] = None
) -> Dict[str, Optional[pd.Series]]:
    """Load AC and No-AC series separately"""
    ac_patterns = [f"{prefix}*ac_out*.xlsx", f"{prefix}*ac_out*.csv"]
    noac_patterns = [f"{prefix}*noac_out*.xlsx", f"{prefix}*noac_out*.csv"]

    return {
        "AC": load_archetype_series_optimized(prefix, target_year, ac_patterns, period_intersection_start, period_intersection_end),
        "NoAC": load_archetype_series_optimized(prefix, target_year, noac_patterns, period_intersection_start, period_intersection_end)
    }


# ============================================================
# BATCH HEAT EVENT PROCESSING
# ============================================================

def process_all_heat_events_batch(row: pd.Series, hourly: pd.DataFrame, arch: str) -> None:
    """
    Batch process all heat events for a single address.
    Load archetype data once and reuse for all events.
    """
    archetypes = [a.strip() for a in arch.split(",") if a.strip()]
    
    # Load all archetype data once from cache
    archetype_data = {}
    for archetype in archetypes:
        sim_df = load_simulation_optimized(archetype, 2025)
        if sim_df is not None:
            archetype_data[archetype] = sim_df
    
    if not archetype_data:
        logger.debug(f"No archetype data for {row['address']}")
        return
    
    # Process all events with pre-loaded data
    for event_id, event_info in HEAT_EVENTS.items():
        event_start = pd.to_datetime(event_info['start'])
        event_end = pd.to_datetime(event_info['end'])
        
        hobo_slice = hourly.loc[event_start:event_end].copy()
        
        if hobo_slice.empty:
            continue
        
        # Process individual archetype plots
        for archetype, sim_df in archetype_data.items():
            sim_slice = sim_df.loc[event_start:event_end].copy()
            
            if sim_slice.empty:
                continue
            
            common_idx = pd.date_range(start=event_start, end=event_end, freq="h")
            hobo_aligned = hobo_slice.reindex(common_idx)
            sim_aligned = sim_slice.reindex(common_idx)
            
            combined = hobo_aligned.join(sim_aligned, how="outer").dropna()
            
            if combined.empty:
                continue
            
            n = len(combined)
            
            # NumPy-based MSE calculation
            mse_avg = calculate_mse_numpy(
                combined["archetype_internal_temperature"].values,
                combined["actual_average_temperature"].values
            )
            mse_max = calculate_mse_numpy(
                combined["archetype_internal_temperature"].values,
                combined["actual_max_temperature"].values
            )
            
            label = f"{row['clean_address']}_{archetype}_{event_id}"
            out_path = config.heat_events_dir / f"{label}.png"
            
            config.plotter.plot_heat_event(
                combined_data=combined,
                address=row['address'],
                archetype=archetype,
                event_name=event_info['name'],
                event_id=event_id,
                output_path=out_path,
                mse_avg=mse_avg,
                mse_max=mse_max
            )
        
        # Process averaged archetype plots
        process_averaged_heat_event(row, hobo_slice, archetype_data, event_id, event_info, event_start, event_end)


def process_averaged_heat_event(row, hobo_slice, archetype_data, event_id, event_info, event_start, event_end):
    """Process averaged heat event plot"""
    archetype_series_list = []
    archetype_names = []
    
    for archetype, sim_df in archetype_data.items():
        sim_slice = sim_df.loc[event_start:event_end].copy()
        
        if not sim_slice.empty:
            archetype_series_list.append(sim_slice)
            archetype_names.append(archetype)
    
    if not archetype_series_list:
        return
    
    common_idx = pd.date_range(start=event_start, end=event_end, freq="h")
    hobo_aligned = hobo_slice.reindex(common_idx)
    
    aligned_archetype_temps = []
    for sim_slice in archetype_series_list:
        sim_aligned = sim_slice.reindex(common_idx)
        aligned_archetype_temps.append(sim_aligned["archetype_internal_temperature"])
    
    if aligned_archetype_temps:
        t_predicted_df = pd.concat(aligned_archetype_temps, axis=1)
        t_predicted_df.columns = archetype_names
        t_predicted = t_predicted_df.mean(axis=1)
    else:
        return
    
    combined = hobo_aligned.copy()
    combined['T_predicted'] = t_predicted
    
    for i, arch_name in enumerate(archetype_names):
        combined[f'{arch_name}_temp'] = aligned_archetype_temps[i]
    
    combined = combined.dropna()
    
    if combined.empty:
        return
    
    # NumPy-based MSE calculation
    mse_avg = calculate_mse_numpy(
        combined["T_predicted"].values,
        combined["actual_average_temperature"].values
    )
    mse_max = calculate_mse_numpy(
        combined["T_predicted"].values,
        combined["actual_max_temperature"].values
    )
    
    label = f"{row['clean_address']}_averaged_{event_id}"
    out_path = config.heat_events_averaged_dir / f"{label}.png"
    
    config.plotter.plot_heat_event_averaged(
        combined_data=combined,
        address=row['address'],
        archetypes=", ".join(archetype_names),
        event_name=event_info['name'],
        event_id=event_id,
        output_path=out_path,
        mse_avg=mse_avg,
        mse_max=mse_max
    )


def plot_period_intersection_for_row(row: pd.Series, hourly: pd.DataFrame, 
                                      period_intersection_start: pd.Timestamp,
                                      period_intersection_end: pd.Timestamp,
                                      arch: str) -> None:
    """Generate averaged archetype plot for entire period intersection"""
    archetypes = [a.strip() for a in arch.split(",") if a.strip()]
    
    hobo_slice = hourly.loc[period_intersection_start:period_intersection_end].copy()
    
    if hobo_slice.empty:
        return
    
    archetype_series_list = []
    archetype_names = []
    
    for archetype in archetypes:
        sim_df = load_simulation_optimized(archetype, 2025)
        
        if sim_df is None or sim_df.empty:
            continue
        
        sim_slice = sim_df.loc[period_intersection_start:period_intersection_end].copy()
        
        if sim_slice.empty:
            continue
        
        archetype_series_list.append(sim_slice)
        archetype_names.append(archetype)
    
    if not archetype_series_list:
        return
    
    common_idx = pd.date_range(
        start=period_intersection_start,
        end=period_intersection_end,
        freq="h"
    )
    
    hobo_aligned = hobo_slice.reindex(common_idx)
    
    aligned_archetype_temps = []
    for sim_slice in archetype_series_list:
        sim_aligned = sim_slice.reindex(common_idx)
        aligned_archetype_temps.append(sim_aligned["archetype_internal_temperature"])
    
    if aligned_archetype_temps:
        t_predicted_df = pd.concat(aligned_archetype_temps, axis=1)
        t_predicted_df.columns = archetype_names
        t_predicted = t_predicted_df.mean(axis=1)
    else:
        return
    
    combined = hobo_aligned.copy()
    combined['T_predicted'] = t_predicted
    
    for i, arch_name in enumerate(archetype_names):
        combined[f'{arch_name}_temp'] = aligned_archetype_temps[i]
    
    if combined.empty:
        return
    
    label = f"{row['clean_address']}_period_intersection"
    out_path = config.period_intersection_dir / f"{label}.png"
    
    config.plotter.plot_period_intersection(
        combined_data=combined,
        address=row['address'],
        archetypes=", ".join(archetype_names),
        period_start=period_intersection_start,
        period_end=period_intersection_end,
        output_path=out_path
    )


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def process_mapping_row(row: pd.Series, analysis_type: str) -> Optional[Union[Dict, List[Dict]]]:
    """
    Unified processor for mapping rows with optimized operations.
    """
    arch = row.get("archtypes used")
    if not isinstance(arch, str):
        return None

    hobo_file = config.hobo_output_dir / row["outfile"]
    if not hobo_file.exists():
        return None

    try:
        hourly = pd.read_excel(hobo_file, index_col="timestamp", engine='openpyxl')
        hourly.index = pd.to_datetime(hourly.index)

        if analysis_type == 'heat_events_batch':
            process_all_heat_events_batch(row, hourly, arch)
            return None

        app_start = row.get("commheat start")
        app_end = row.get("commheat end")

        if pd.isna(app_start) or pd.isna(app_end):
            return None

        period_intersection_start = max(hourly.index.min(), app_start)
        period_intersection_end = min(hourly.index.max(), app_end)

        is_valid, msg = validate_period_intersection(period_intersection_start, period_intersection_end)
        if not is_valid:
            logger.warning(f"Invalid period for {row['address']}: {msg}")
            return None

        if analysis_type == 'mse':
            return compute_mse_for_row_optimized(row, hourly, period_intersection_start, period_intersection_end, arch)
        elif analysis_type == 'period_intersection_means':
            return compute_period_intersection_means_for_row(row, hourly, period_intersection_start, period_intersection_end, arch)
        elif analysis_type == 'ac_noac_plots':
            plot_ac_noac_for_row(row, period_intersection_start, period_intersection_end, arch)
            return None
        elif analysis_type == 'period_intersection_plots':
            plot_period_intersection_for_row(row, hourly, period_intersection_start, period_intersection_end, arch)
            return None
        elif analysis_type == 'comprehensive_mse':
            return compute_comprehensive_mse_for_row(row, hourly, period_intersection_start, period_intersection_end, arch)

    except Exception as e:
        logger.error(f"Error processing {row['address']}: {e}", exc_info=True)
        return None


def compute_mse_for_row_optimized(row: pd.Series, hourly: pd.DataFrame, obs_start: pd.Timestamp, obs_end: pd.Timestamp, arch: str) -> Optional[List[Dict]]:
    """
    Optimized MSE computation with NumPy and reduced operations.
    """
    hobo_window = hourly.loc[obs_start:obs_end].copy()
    if hobo_window.empty:
        return None

    # Single resample operation
    hobo_window = hobo_window.resample("h").mean()

    results = []
    archetypes = [a.strip() for a in arch.split(",")]
    
    for scenario in ['AC', 'NoAC']:
        archetype_series_list = []

        if scenario == 'AC':
            patterns = lambda archetype: [f"{archetype}*ac_out*.xlsx", f"{archetype}*ac_out*.csv"]
        else:
            patterns = lambda archetype: [f"{archetype}*noac_out*.xlsx", f"{archetype}*noac_out*.csv"]

        for archetype in archetypes:
            sim_series = load_archetype_series_optimized(
                archetype,
                hobo_window.index.min().year,
                patterns(archetype),
                obs_start,
                obs_end
            )

            if sim_series is None or sim_series.empty:
                continue

            sim_df = pd.DataFrame({
                "archetype_internal_temperature": sim_series
            })
            
            sim_df = sim_df.resample("h").mean()
            archetype_series_list.append((archetype, sim_df))

        if not archetype_series_list:
            continue

        actual_start = max(obs_start, max(s[1].index.min() for s in archetype_series_list))
        actual_end = min(obs_end, min(s[1].index.max() for s in archetype_series_list))

        if actual_start >= actual_end:
            continue

        common_idx = pd.date_range(
            start=pd.to_datetime(actual_start).floor("h"),
            end=pd.to_datetime(actual_end).floor("h"),
            freq="h"
        )

        hobo_aligned = hobo_window.reindex(common_idx)

        aligned_archetype_temps = []
        for archetype, sim_df in archetype_series_list:
            sim_aligned = sim_df.reindex(common_idx)
            aligned_archetype_temps.append(sim_aligned["archetype_internal_temperature"])

        if aligned_archetype_temps:
            t_predicted_df = pd.concat(aligned_archetype_temps, axis=1)
            t_predicted_df.columns = [arch for arch, _ in archetype_series_list]
            t_predicted = t_predicted_df.mean(axis=1)
        else:
            continue

        combined = hobo_aligned.join(pd.DataFrame({"T_predicted": t_predicted}), how="inner").dropna()
        
        for i, (archetype, _) in enumerate(archetype_series_list):
            combined[f'{archetype}_temp'] = aligned_archetype_temps[i]

        if combined.empty:
            continue

        n = len(combined)

        # NumPy-based MSE calculations (much faster)
        mse_predicted_avg = calculate_mse_numpy(
            combined["T_predicted"].values,
            combined["actual_average_temperature"].values
        )
        mse_predicted_max = calculate_mse_numpy(
            combined["T_predicted"].values,
            combined["actual_max_temperature"].values
        )

        # Create output files
        label = f"{row['housetype']}_{row['clean_address']}_{scenario}"

        comparison_data = combined.copy()
        comparison_data = comparison_data.rename(columns={
            "actual_average_temperature": "Hobologger_avg",
            "actual_max_temperature": "Hobologger_max"
        })

        xlsx_path = config.comparison_dir / f"{label}_entire_pilot_period.xlsx"
        comparison_data.to_excel(xlsx_path, engine='openpyxl')

        png_path = config.entire_pilot_period / f"{label}_entire_pilot_period.png"
        config.plotter.plot_intersection_comparison_averaged(
            combined_data=combined,
            row_info={"address": f"{row['address']} ({scenario})"},
            archetypes=", ".join([arch for arch, _ in archetype_series_list]),
            mse_predicted_avg=mse_predicted_avg,
            mse_predicted_max=mse_predicted_max,
            actual_start=actual_start,
            actual_end=actual_end,
            output_path=png_path
        )

        results.append({
            "address": row["address"],
            "scenario": scenario,
            "archetypes": ", ".join([arch for arch, _ in archetype_series_list]),
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

    return results if results else None


def compute_period_intersection_means_for_row(row: pd.Series, hourly: pd.DataFrame, period_intersection_start: pd.Timestamp,
                                   period_intersection_end: pd.Timestamp, arch: str) -> Optional[Dict]:
    """Compute period intersection means with optimized operations"""
    hobo_slice = hourly.loc[period_intersection_start:period_intersection_end]
    if hobo_slice.empty:
        return None

    # Vectorized mean calculation
    hobo_mean_value = ((hobo_slice["actual_average_temperature"] + hobo_slice["actual_max_temperature"]) / 2.0).mean()

    sim_means = {}
    for archetype in [a.strip() for a in arch.split(",") if a.strip()]:
        sim_series = load_archetype_series_optimized(
            archetype,
            hourly.index.min().year,
            [f"{archetype}*ac_out*.xlsx", f"{archetype}*ac_out*.csv",
             f"{archetype}*noac_out*.xlsx", f"{archetype}*noac_out*.csv"],
            period_intersection_start,
            period_intersection_end
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
        "period_intersection_start": period_intersection_start,
        "period_intersection_end": period_intersection_end,
        "n_hours": len(hobo_slice),
        "hobologger_mean_C": hobo_mean_value,
        "sim_overall_mean_C": sim_overall_mean,
        "final_combined_mean_C": final_mean,
        "archetypes": arch,
    }


def compute_comprehensive_mse_for_row(row: pd.Series, hourly: pd.DataFrame, obs_start: pd.Timestamp, obs_end: pd.Timestamp, arch: str) -> Optional[Dict]:
    """Compute comprehensive MSE with NumPy optimizations"""
    hobo_window = hourly.loc[obs_start:obs_end].copy()
    if hobo_window.empty:
        return None

    hobo_window = hobo_window.resample("h").mean()

    mse_comparison_data = {
        "address": row["address"],
        "housetype": row["housetype"],
        "archetypes": arch,
        "period_start": obs_start,
        "period_end": obs_end,
    }

    archetypes = [a.strip() for a in arch.split(",")]
    
    for scenario in ['AC', 'NoAC']:
        archetype_series_list = []

        if scenario == 'AC':
            patterns = lambda archetype: [f"{archetype}*ac_out*.xlsx", f"{archetype}*ac_out*.csv"]
        else:
            patterns = lambda archetype: [f"{archetype}*noac_out*.xlsx", f"{archetype}*noac_out*.csv"]

        for archetype in archetypes:
            sim_series = load_archetype_series_optimized(
                archetype,
                hobo_window.index.min().year,
                patterns(archetype),
                obs_start,
                obs_end
            )

            if sim_series is None or sim_series.empty:
                continue

            sim_df = pd.DataFrame({"archetype_internal_temperature": sim_series})
            sim_df = sim_df.resample("h").mean()
            archetype_series_list.append((archetype, sim_df))

        if not archetype_series_list:
            continue

        actual_start = max(obs_start, max(s[1].index.min() for s in archetype_series_list))
        actual_end = min(obs_end, min(s[1].index.max() for s in archetype_series_list))

        if actual_start >= actual_end:
            continue

        common_idx = pd.date_range(
            start=pd.to_datetime(actual_start).floor("h"),
            end=pd.to_datetime(actual_end).floor("h"),
            freq="h"
        )

        hobo_aligned = hobo_window.reindex(common_idx)

        aligned_archetype_temps = []
        for archetype, sim_df in archetype_series_list:
            sim_aligned = sim_df.reindex(common_idx)
            aligned_archetype_temps.append(sim_aligned["archetype_internal_temperature"])

        if aligned_archetype_temps:
            t_predicted_df = pd.concat(aligned_archetype_temps, axis=1)
            t_predicted_df.columns = [arch for arch, _ in archetype_series_list]
            t_predicted = t_predicted_df.mean(axis=1)
        else:
            continue

        combined = hobo_aligned.join(pd.DataFrame({"T_predicted": t_predicted}), how="inner").dropna()

        if combined.empty:
            continue

        n = len(combined)

        # NumPy-based MSE calculations
        mse_vs_avg = calculate_mse_numpy(
            combined["T_predicted"].values,
            combined["actual_average_temperature"].values
        )
        mse_vs_max = calculate_mse_numpy(
            combined["T_predicted"].values,
            combined["actual_max_temperature"].values
        )

        prefix = scenario.lower()
        mse_comparison_data[f"{prefix}_mse_vs_actual_avg"] = mse_vs_avg
        mse_comparison_data[f"{prefix}_mse_vs_actual_max"] = mse_vs_max
        mse_comparison_data[f"{prefix}_mean_temp"] = combined["T_predicted"].mean()
        mse_comparison_data[f"{prefix}_n_hours"] = n

    if "ac_mse_vs_actual_avg" in mse_comparison_data and "noac_mse_vs_actual_avg" in mse_comparison_data:
        mse_comparison_data["mse_diff_ac_vs_noac_avg"] = (
            mse_comparison_data["ac_mse_vs_actual_avg"] - mse_comparison_data["noac_mse_vs_actual_avg"]
        )
        mse_comparison_data["mse_diff_ac_vs_noac_max"] = (
            mse_comparison_data["ac_mse_vs_actual_max"] - mse_comparison_data["noac_mse_vs_actual_max"]
        )
        
        mse_comparison_data["actual_avg_mean"] = hobo_aligned["actual_average_temperature"].mean()
        mse_comparison_data["actual_max_mean"] = hobo_aligned["actual_max_temperature"].mean()

    return mse_comparison_data if len(mse_comparison_data) > 5 else None


def plot_ac_noac_for_row(row: pd.Series, period_intersection_start: pd.Timestamp, period_intersection_end: pd.Timestamp, arch: str) -> None:
    """Generate AC vs No-AC plots"""
    for archetype in [a.strip() for a in arch.split(",") if a.strip()]:
        data = load_ac_noac_series(archetype, period_intersection_start.year, period_intersection_start, period_intersection_end)

        ac = data.get("AC")
        noac = data.get("NoAC")

        if ac is None or noac is None:
            logger.warning(f"Missing AC/No-AC data for {archetype}")
            continue

        out_name = f"{row['clean_address']}_{archetype}_AC_NoAC_PERIOD_INTERSECTION.png"
        out_path = config.ac_noac_dir / out_name
        
        config.plotter.plot_ac_vs_noac(ac, noac, f"{row['address']} - {archetype}", out_path)


def get_optimal_workers(operation_type: str) -> int:
    """Get optimal worker count based on operation type"""
    cpu_count = multiprocessing.cpu_count()
    
    if operation_type == 'io_bound':
        return min(32, cpu_count * 4)
    elif operation_type == 'cpu_bound':
        return max(1, cpu_count - 1)
    elif operation_type == 'mixed':
        return cpu_count
    
    return cpu_count


def run_analysis_optimized(mapping_df: pd.DataFrame, analysis_type: str, desc: str) -> List[Dict]:
    """
    Optimized analysis runner with appropriate parallelization.
    """
    results = []
    
    # Determine operation type and workers
    if analysis_type in ['mse', 'comprehensive_mse', 'period_intersection_means']:
        max_workers = get_optimal_workers('cpu_bound')
    else:
        max_workers = get_optimal_workers('io_bound')
    
    # Use ThreadPoolExecutor for all operations (simpler, avoids pickling issues)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_mapping_row, row, analysis_type): idx 
            for idx, row in mapping_df.iterrows()
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                result = future.result()
                if result:
                    if isinstance(result, list):
                        results.extend(result)
                    else:
                        results.append(result)
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
    
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    """Main execution with optimizations"""
    logger.info(color_text("=== Starting CommHEAT Data Comparison ===\n", "96"))

    # Initialize plotting manager and data loader
    config.initialize_dependencies(
        plotter=PlottingManager(config.output_dir, PlotStyle.get_style()),
        data_loader=DataLoader(config, ColumnTypes.to_dict(), Patterns.to_dict())
    )

    # Load sensor mapping file using DataLoader
    mapping = load_mapping()

    # Find all HOBO sensor files
    hobo_files = [f for f in config.hobo_dir.glob("*.xlsx")
                  if "sensor contact" not in f.name.lower() and not f.name.startswith("~$")]

    # Process HOBO files with optimal parallelization
    logger.info(color_text("\n=== Processing HOBO Sensor Data ===\n", "92"))
    
    max_workers = get_optimal_workers('io_bound')
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_hobo_file, f, mapping) for f in hobo_files]
        summary = [f.result() for f in tqdm(as_completed(futures), total=len(futures), desc="Processing HOBO files")
                   if f.result() is not None]

    # Save HOBO processing summary
    if summary:
        df = pd.DataFrame(summary)
        save_path = config.output_dir / "HoboHouseIndex.xlsx"

        try:
            df.to_excel(save_path, index=False, engine='openpyxl')
            logger.info(f"Saved HOBO index to: {save_path}")
        except PermissionError:
            backup = config.output_dir / "HoboHouseIndex_backup.xlsx"
            df.to_excel(backup, index=False, engine='openpyxl')
            logger.info(f"Saved backup: {backup}")

    # Preload all archetypes into cache
    preload_archetypes(mapping, target_year=2025)

    # Run MSE analysis
    logger.info(color_text("\n=== Computing MSE ===\n", "92"))
    mse_results = run_analysis_optimized(mapping, 'mse', "Computing MSE")

    if mse_results:
        pd.DataFrame(mse_results).to_excel(
            config.summary_dir / "Intersection_MSE_Summary.xlsx", 
            index=False, 
            engine='openpyxl'
        )

    # Run comprehensive MSE comparison analysis
    logger.info(color_text("\n=== Computing Comprehensive MSE Comparisons ===\n", "92"))
    comprehensive_mse_results = run_analysis_optimized(mapping, 'comprehensive_mse', "Computing Comprehensive MSE")

    if comprehensive_mse_results:
        comprehensive_df = pd.DataFrame(comprehensive_mse_results)
        save_path = config.summary_dir / "Comprehensive_MSE_Comparison.xlsx"
        comprehensive_df.to_excel(save_path, index=False, engine='openpyxl')
        logger.info(f"Saved MSE comparison to: {save_path}")

    # Run period intersection means analysis
    logger.info(color_text("\n=== Computing Period Intersection Means ===\n", "92"))
    period_intersection_results = run_analysis_optimized(mapping, 'period_intersection_means', "Computing Period Intersection Means")

    if period_intersection_results:
        pd.DataFrame(period_intersection_results).to_excel(
            config.summary_dir / "Period_Intersection_Mean_Summary.xlsx", 
            index=False, 
            engine='openpyxl'
        )

    # Generate AC vs No-AC plots
    logger.info(color_text("\n=== Plotting AC vs No-AC ===\n", "92"))
    run_analysis_optimized(mapping, 'ac_noac_plots', "AC/No-AC Plots")

    # Generate heat event plots (batch processing - both individual and averaged)
    logger.info(color_text("\n=== Plotting Heat Events (Batch Processing) ===\n", "92"))
    run_analysis_optimized(mapping, 'heat_events_batch', "Heat Event Plots")
    
    # Generate period intersection plots
    logger.info(color_text("\n=== Plotting Period Intersection ===\n", "96"))
    run_analysis_optimized(mapping, 'period_intersection_plots', "Period Intersection Plots")

    logger.info(color_text("\n=== DONE ===\n", "92"))

if __name__ == "__main__":
    main()
