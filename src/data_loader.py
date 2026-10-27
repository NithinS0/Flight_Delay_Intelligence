import os
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import streamlit as st

# Search locations for flight_data.csv
ROOT_DIR = Path(__file__).resolve().parent.parent
SEARCH_PATHS = [
    ROOT_DIR / "flight_data.csv",
    ROOT_DIR / "data" / "flight_data.csv",
    Path("flight_data.csv"),
    Path("data/flight_data.csv"),
]

# Required columns expected in the dataset
REQUIRED_COLUMNS = [
    "FL_DATE",
    "AIRLINE",
    "ORIGIN",
    "DEST",
    "DEP_DELAY",
    "CRS_DEP_TIME",
    "DISTANCE",
    "CRS_ELAPSED_TIME",
]

def find_dataset_path() -> Optional[Path]:
    """Searches for flight_data.csv across project root and data directory."""
    for p in SEARCH_PATHS:
        if p.exists() and p.is_file():
            return p
    # Check fallback for development environment if full flight_data is in root
    fallback = ROOT_DIR / "data" / "raw" / "benchmark_sample.csv"
    if fallback.exists():
        return fallback
    return None

def validate_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validates that all essential required columns exist in the loaded dataframe."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return (len(missing) == 0, missing)

@st.cache_data(show_spinner=False)
def load_raw_dataset(max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    """
    Loads flight_data.csv with optimized memory footprints.
    Cached with Streamlit st.cache_data for instant subsequent renders.
    """
    file_path = find_dataset_path()
    if file_path is None:
        raise FileNotFoundError(
            "CRITICAL DATASET MISSING: Could not locate 'flight_data.csv'.\n"
            "Please ensure 'flight_data.csv' is placed directly in the project root directory "
            f"({ROOT_DIR}) or inside the '{ROOT_DIR / 'data'}' directory."
        )

    # Determine load parameters
    # Read with downcasted dtypes to preserve system RAM while keeping precision
    dtypes = {
        "CRS_DEP_TIME": "int32",
        "DEP_DELAY": "float32",
        "DISTANCE": "float32",
        "CRS_ELAPSED_TIME": "float32",
        "CANCELLED": "float32",
        "DIVERTED": "float32",
    }
    
    # Check available columns first
    header_df = pd.read_csv(file_path, nrows=5)
    valid_dtypes = {k: v for k, v in dtypes.items() if k in header_df.columns}
    
    df = pd.read_csv(file_path, dtype=valid_dtypes, nrows=max_rows)
    
    is_valid, missing_cols = validate_columns(df)
    if not is_valid:
        raise ValueError(
            f"Dataset at {file_path} is missing required columns: {missing_cols}. "
            "Expected columns include FL_DATE, AIRLINE, ORIGIN, DEST, DEP_DELAY, CRS_DEP_TIME, DISTANCE, CRS_ELAPSED_TIME."
        )
        
    return df, file_path.name
