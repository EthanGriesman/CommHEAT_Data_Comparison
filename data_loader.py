"""
Data Loading Module for CommHEAT Analysis
==========================================

Handles all file I/O operations including:
- CSV and Excel file loading
- Sensor mapping data
- Column detection and validation
- Caching for performance
"""

from pathlib import Path
import pandas as pd
import logging
from typing import Optional, List, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Centralized data loading class for all file I/O operations.
    
    Features:
    - Automatic skiprows detection for headers
    - Column name normalization
    - Type-based column finding
    - LRU caching for frequently accessed files
    - Support for CSV and Excel formats
    """
    
    def __init__(self, config, column_types: Dict[str, List[str]], patterns: Dict):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: Configuration object with paths and settings
            column_types: Mapping of column types to keyword lists
            patterns: Regex patterns for data validation
        """
        self.config = config
        self.column_types = column_types
        self.patterns = patterns
        self._file_cache = {}
    
    @lru_cache(maxsize=256)
    def load_dataframe_cached(self, filepath_str: str, max_skiprows: int = 3) -> Optional[pd.DataFrame]:
        """
        Cached wrapper for load_dataframe.
        
        Args:
            filepath_str: String path to file
            max_skiprows: Maximum header rows to skip
            
        Returns:
            Loaded DataFrame or None if loading fails
        """
        return self.load_dataframe(Path(filepath_str), max_skiprows)
    
    def load_dataframe(self, filepath: Path, max_skiprows: int = 3) -> Optional[pd.DataFrame]:
        """
        Load CSV or Excel file with automatic skiprows detection.
        
        Tries multiple skiprows values to find valid data with:
        - At least one time column
        - More than 100 rows of data
        
        Args:
            filepath: Path to file
            max_skiprows: Maximum rows to skip for header detection
            
        Returns:
            Loaded and validated DataFrame or None
        """
        for skiprows in range(max_skiprows + 1):
            try:
                # Load based on file extension
                if filepath.suffix.lower() == ".xlsx":
                    df = pd.read_excel(filepath, skiprows=skiprows, engine='openpyxl')
                elif filepath.suffix.lower() == ".csv":
                    df = pd.read_csv(filepath, skiprows=skiprows)
                else:
                    logger.warning(f"Unsupported file format: {filepath.suffix}")
                    return None

                # Normalize column names
                df.columns = df.columns.str.strip()

                # Validate: must have time column and sufficient data
                time_cols = [c for c in df.columns 
                           if any(t in c.lower() for t in self.column_types.get('time', []))]
                
                if time_cols and len(df) > 100:
                    if skiprows > 0:
                        logger.debug(f"Loaded {filepath.name} with {skiprows} rows skipped")
                    return df
                    
            except Exception as e:
                logger.debug(f"Skip {skiprows} failed for {filepath.name}: {e}")
                continue

        logger.warning(f"Could not load {filepath.name}")
        return None
    
    def find_column(self, df: pd.DataFrame, col_type: str) -> Optional[str]:
        """
        Find column by type using multiple strategies.
        
        Strategies for 'temp' type:
        1. Pattern match for archetype format
        2. Contains specific keywords
        3. Match target columns list
        4. Any temperature column with [C] unit
        
        Args:
            df: DataFrame to search
            col_type: Type of column ('time', 'temp', etc.)
            
        Returns:
            Column name or None if not found
        """
        if col_type == 'temp':
            return self._find_temperature_column(df)
        
        # Generic search for other column types
        return self._find_generic_column(df, col_type)
    
    def _find_temperature_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find temperature column using multiple strategies.
        
        Args:
            df: DataFrame to search
            
        Returns:
            Temperature column name or None
        """
        # Strategy 1: Pattern match for archetype format
        if 'archetype_temp' in self.patterns:
            cols = [c for c in df.columns if self.patterns['archetype_temp'].match(c)]
            if cols:
                return cols[0]

        # Strategy 2: Contains specific keywords
        cols = [c for c in df.columns
                if all(k in c.lower() for k in ['zone mean air temperature', '[c]'])]
        if cols:
            return cols[0]

        # Strategy 3: Match target columns list
        if hasattr(self.config, 'target_columns'):
            cols = [c for c in self.config.target_columns if c in df.columns]
            if cols:
                return cols[0]

        # Strategy 4: Any temp column with [c] unit
        cols = [c for c in df.columns if 'temperature' in c.lower() and '[c]' in c.lower()]
        return cols[0] if cols else None
    
    def _find_generic_column(self, df: pd.DataFrame, col_type: str) -> Optional[str]:
        """
        Find column using generic keyword matching.
        
        Args:
            df: DataFrame to search
            col_type: Type of column to find
            
        Returns:
            Column name or None
        """
        keywords = self.column_types.get(col_type, [])
        for col in df.columns:
            if any(k in col.lower() for k in keywords):
                return col
        return None
    
    def load_mapping(self) -> pd.DataFrame:
        """
        Load and process sensor mapping file with vectorized operations.
        
        Processing steps:
        1. Load Excel file with header on row 3
        2. Normalize column names
        3. Rename standard columns
        4. Drop rows with missing critical data
        5. Derive house type, clean address, and output filename
        6. Parse date columns
        
        Returns:
            Processed mapping DataFrame
            
        Raises:
            Exception: If file cannot be loaded or processed
        """
        try:
            df = pd.read_excel(self.config.mapping_file, header=2, engine='openpyxl')
            df.columns = df.columns.str.strip().str.lower()

            # Rename standard columns
            df = df.rename(columns={
                "home address": "address", 
                "sensor #": "sensor_id"
            })
            
            # Drop rows missing critical data
            df = df.dropna(subset=["sensor_id", "address"])
            df["sensor_id"] = df["sensor_id"].astype(str)

            # Vectorized derivations
            df["housetype"] = df["address"].str.contains(
                "apt|apartment", 
                case=False, 
                regex=True
            ).map({True: "Apt", False: "Ind"})
            
            df["clean_address"] = df["address"].str.replace(
                r"[^A-Za-z0-9]", 
                "", 
                regex=True
            )
            
            df["outfile"] = df["housetype"] + "_" + df["clean_address"] + ".xlsx"

            # Vectorized date parsing
            for col in ["commheat start", "commheat end"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

            logger.info(f"Loaded {len(df)} sensor mappings")
            return df

        except Exception as e:
            logger.error(f"Error loading mapping: {e}", exc_info=True)
            raise
    
    def load_multiple_files(
        self, 
        file_patterns: List[str], 
        directory: Path = None
    ) -> List[pd.DataFrame]:
        """
        Load multiple files matching patterns.
        
        Args:
            file_patterns: List of glob patterns
            directory: Directory to search (defaults to config.latest_ep_dir)
            
        Returns:
            List of loaded DataFrames
        """
        if directory is None:
            directory = self.config.latest_ep_dir
        
        dataframes = []
        for pattern in file_patterns:
            files = list(directory.glob(pattern))
            for file in files:
                df = self.load_dataframe(file)
                if df is not None:
                    dataframes.append(df)
        
        return dataframes
    
    def validate_dataframe(
        self, 
        df: pd.DataFrame, 
        required_columns: Optional[List[str]] = None,
        min_rows: int = 100
    ) -> bool:
        """
        Validate DataFrame meets requirements.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows
            
        Returns:
            True if valid, False otherwise
        """
        if df is None or df.empty:
            return False
        
        if len(df) < min_rows:
            logger.debug(f"DataFrame has {len(df)} rows, minimum is {min_rows}")
            return False
        
        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                logger.debug(f"Missing required columns: {missing}")
                return False
        
        return True
    
    def get_file_info(self, filepath: Path) -> Dict:
        """
        Get metadata about a file.
        
        Args:
            filepath: Path to file
            
        Returns:
            Dictionary with file metadata
        """
        if not filepath.exists():
            return {"exists": False}
        
        stat = filepath.stat()
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": stat.st_mtime,
            "extension": filepath.suffix,
            "name": filepath.name
        }
    
    def clear_cache(self):
        """Clear the LRU cache for cached file loading."""
        self.load_dataframe_cached.cache_clear()
        self._file_cache.clear()
        logger.info("Data loader cache cleared")
    
    def get_cache_info(self) -> Dict:
        """
        Get information about cache usage.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_info = self.load_dataframe_cached.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
            "hit_rate": cache_info.hits / (cache_info.hits + cache_info.misses) 
                       if (cache_info.hits + cache_info.misses) > 0 else 0
        }
    
    def __repr__(self) -> str:
        """String representation of DataLoader."""
        cache_info = self.get_cache_info()
        return (f"DataLoader(cache_size={cache_info['currsize']}/{cache_info['maxsize']}, "
                f"hit_rate={cache_info['hit_rate']:.2%})")
