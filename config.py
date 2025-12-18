"""
CommHEAT Configuration Module
==============================

Centralized configuration, logging setup, patterns, and styling for the CommHEAT data analysis pipeline.
"""

from pathlib import Path
import re
import logging
import sys
import io
from typing import List, Dict
from dataclasses import dataclass


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(log_file: str = 'commheat_analysis.log') -> logging.Logger:
    """
    Configure logging to file and console with Windows console encoding support.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Configured logger instance
    """
    # Handle Windows console encoding issues
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    # Configure logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


# ============================================================
# PATTERNS AND STYLING
# ============================================================

class Patterns:
    """Regex patterns for data extraction and cleaning"""
    
    SENSOR_ID = re.compile(r"(\d+)")
    ARCHETYPE_TEMP = re.compile(
        r'^(FIRSTFLOOR_\d+|HOUSE_\d+):Zone Mean Air Temperature \[C\]\(Hourly\)$', 
        re.IGNORECASE
    )
    ORDINAL = re.compile(r"(\d+)(st|nd|rd|th)")
    WHITESPACE = re.compile(r"\s+")
    ADDRESS_CLEAN = re.compile(r"[^A-Za-z0-9]")
    
    @classmethod
    def to_dict(cls) -> Dict[str, re.Pattern]:
        """Convert patterns to dictionary format for backward compatibility"""
        return {
            'sensor_id': cls.SENSOR_ID,
            'archetype_temp': cls.ARCHETYPE_TEMP,
            'ordinal': cls.ORDINAL,
            'whitespace': cls.WHITESPACE,
            'address_clean': cls.ADDRESS_CLEAN
        }


class PlotStyle:
    """Matplotlib style configuration for plots"""
    
    DEFAULT = {
        'figure.figsize': (12, 5),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    }
    
    @classmethod
    def get_style(cls) -> Dict:
        """Get the default plot style configuration"""
        return cls.DEFAULT.copy()


class ColumnTypes:
    """Column name mappings for different data types"""
    
    TIME = ['date', 'time', 'timestamp']
    TEMP = ['temperature', 'zone mean air temperature']
    
    @classmethod
    def to_dict(cls) -> Dict[str, List[str]]:
        """Convert column types to dictionary format for backward compatibility"""
        return {
            'time': cls.TIME,
            'temp': cls.TEMP,
        }


class HeatEvents:
    """Heat event and baseline period definitions"""
    
    EVENTS = {
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
    
    @classmethod
    def get_events(cls) -> Dict:
        """Get all heat events and baseline periods"""
        return cls.EVENTS.copy()


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    """Centralized configuration management"""

    # Directory paths for input data
    hobo_dir: Path = Path(r"C:\Users\Ethan\Downloads\2025SCC_OnsetHobo_InHome_Dataloggers\2025SCC_OnsetHobo_InHome_Dataloggers")
    mapping_file: Path = Path(r"C:\Users\Ethan\Downloads\2025SCC_OnsetHobo_InHome_Dataloggers\2025SCC_OnsetHobo_InHome_Dataloggers\Sensor Contact_101325_PickUP.xlsx")
    latest_ep_dir: Path = Path(r"C:\Users\Ethan\Downloads\Latest_EP_Output_Files\Latest_EP_Output_Files")
    output_dir: Path = Path(r"C:\Users\Ethan\OneDrive - Iowa State University\Desktop\CommHEAT Output")

    # HOBO sensor data column names
    hobo_time_col: str = "Date-Time (CDT)"
    hobo_temp_col: str = "Temperature , °F"
    hobo_rh_col: str = "RH , %"

    # EnergyPlus target columns for temperature data
    target_columns: List[str] = None
    
    # External dependencies (set after initialization)
    plotter: 'PlottingManager' = None
    data_loader: 'DataLoader' = None

    def __post_init__(self):
        """Initialize directory structure and default values"""
        self._create_directory_structure()
        self._set_default_target_columns()

    def _create_directory_structure(self):
        """Create all required output directories"""
        # Create main output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        self.comparison_dir = self.output_dir / "comparisons"
        self.comparison_dir.mkdir(exist_ok=True, parents=True)
        
        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True, parents=True)
        
        self.hobo_output_dir = self.output_dir / "hobo_data_processed"
        self.hobo_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create plot subdirectories
        self.heat_events_dir = self.plot_dir / "heat_events_each_archetype"
        self.heat_events_dir.mkdir(exist_ok=True, parents=True)
        
        self.heat_events_averaged_dir = self.plot_dir / "heat_events_averaged_archetypes"
        self.heat_events_averaged_dir.mkdir(exist_ok=True, parents=True)
        
        self.period_intersection_dir = self.plot_dir / "period_intersection_no_archetypes"
        self.period_intersection_dir.mkdir(exist_ok=True, parents=True)
        
        self.ac_noac_dir = self.plot_dir / "ac_noac_period_intersection"
        self.ac_noac_dir.mkdir(exist_ok=True, parents=True)
        
        self.entire_pilot_period = self.plot_dir / "entire_pilot_period"
        self.entire_pilot_period.mkdir(exist_ok=True, parents=True)
        
        # Create summary directory
        self.summary_dir = self.output_dir / "mse_statistics"
        self.summary_dir.mkdir(exist_ok=True, parents=True)

    def _set_default_target_columns(self):
        """Set default target columns if not provided"""
        if self.target_columns is None:
            self.target_columns = [
                "FIRSTFLOOR_0:Zone Mean Air Temperature [C](Hourly)",
                "FIRSTFLOOR_1:Zone Mean Air Temperature [C](Hourly)",
                "FIRSTFLOOR_2:Zone Mean Air Temperature [C](Hourly)",
                "HOUSE_0:Zone Mean Air Temperature [C](Hourly)"
            ]
    
    def initialize_dependencies(self, plotter: 'PlottingManager', data_loader: 'DataLoader'):
        """
        Initialize plotting manager and data loader instances.
        
        Args:
            plotter: PlottingManager instance
            data_loader: DataLoader instance
        """
        self.plotter = plotter
        self.data_loader = data_loader


def create_config() -> Config:
    """
    Factory function to create and initialize the configuration.
    
    Returns:
        Configured Config instance
    """
    return Config()


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

# For backward compatibility with existing code
PATTERNS = Patterns.to_dict()
PLOT_STYLE = PlotStyle.get_style()
COLUMN_TYPES = ColumnTypes.to_dict()
HEAT_EVENTS = HeatEvents.get_events()
