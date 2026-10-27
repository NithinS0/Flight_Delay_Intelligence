import pandas as pd
import numpy as np

# Leakage-safe feature lists for Classification and Baseline Regression
CORE_CATEGORICAL_FEATURES = [
    "AIRLINE",
    "ORIGIN",
    "DEST",
]

CORE_NUMERICAL_FEATURES = [
    "DISTANCE",
    "CRS_ELAPSED_TIME",
    "YEAR",
    "MONTH",
    "DAY_OF_WEEK",
    "SCHEDULED_HOUR",
    "SCHEDULED_MINUTE",
    "TIME_SIN",
    "TIME_COS",
]

CORE_ALL_FEATURES = CORE_CATEGORICAL_FEATURES + CORE_NUMERICAL_FEATURES

# Advanced interaction features for Transformed GBR Regression experiment
ADVANCED_CATEGORICAL_FEATURES = [
    "AIRLINE",
    "ORIGIN",
    "DEST",
    "ROUTE",
    "AIRLINE_ROUTE",
    "DAY_PART",
]

ADVANCED_NUMERICAL_FEATURES = CORE_NUMERICAL_FEATURES + ["IS_WEEKEND"]

ADVANCED_ALL_FEATURES = ADVANCED_CATEGORICAL_FEATURES + ADVANCED_NUMERICAL_FEATURES

def get_day_part(hour: int) -> str:
    """Categorizes scheduled departure hour into standard aviation day parts."""
    if 0 <= hour < 6:
        return "Night"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    else:
        return "Evening"

def get_day_part_vectorized(hours: pd.Series) -> pd.Series:
    """Vectorized categorization of scheduled departure hour into standard aviation day parts."""
    conditions = [
        (hours >= 0) & (hours < 6),
        (hours >= 6) & (hours < 12),
        (hours >= 12) & (hours < 18),
        (hours >= 18) & (hours <= 24),
    ]
    choices = ["Night", "Morning", "Afternoon", "Evening"]
    return pd.Series(np.select(conditions, choices, default="Day"), index=hours.index)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates standard leakage-safe features for classification & regression."""
    df_eng = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df_eng["FL_DATE"]):
        df_eng["FL_DATE"] = pd.to_datetime(df_eng["FL_DATE"], errors="coerce")
        
    if "YEAR" not in df_eng.columns:
        df_eng["YEAR"] = df_eng["FL_DATE"].dt.year
    if "MONTH" not in df_eng.columns:
        df_eng["MONTH"] = df_eng["FL_DATE"].dt.month
    if "DAY_OF_WEEK" not in df_eng.columns:
        df_eng["DAY_OF_WEEK"] = df_eng["FL_DATE"].dt.dayofweek
        
    if "SCHEDULED_HOUR" not in df_eng.columns or "SCHEDULED_MINUTE" not in df_eng.columns:
        crs = df_eng["CRS_DEP_TIME"].astype(int)
        df_eng["SCHEDULED_HOUR"] = crs // 100
        df_eng["SCHEDULED_MINUTE"] = crs % 100
        
    if "TIME_SIN" not in df_eng.columns or "TIME_COS" not in df_eng.columns:
        total_minutes = df_eng["SCHEDULED_HOUR"].astype(float) * 60.0 + df_eng["SCHEDULED_MINUTE"].astype(float)
        df_eng["TIME_SIN"] = np.sin(2.0 * np.pi * total_minutes / 1440.0).astype(np.float32)
        df_eng["TIME_COS"] = np.cos(2.0 * np.pi * total_minutes / 1440.0).astype(np.float32)
        
    if "DELAYED" not in df_eng.columns and "DEP_DELAY" in df_eng.columns:
        df_eng["DELAYED"] = (df_eng["DEP_DELAY"] >= 15.0).astype(int)
        
    return df_eng

def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates advanced interaction features: ROUTE, AIRLINE_ROUTE, DAY_PART, IS_WEEKEND."""
    df_adv = engineer_features(df)
    
    if "ROUTE" not in df_adv.columns:
        df_adv["ROUTE"] = df_adv["ORIGIN"].astype(str) + "_" + df_adv["DEST"].astype(str)
        
    if "AIRLINE_ROUTE" not in df_adv.columns:
        df_adv["AIRLINE_ROUTE"] = df_adv["AIRLINE"].astype(str) + "_" + df_adv["ROUTE"]
        
    if "DAY_PART" not in df_adv.columns:
        df_adv["DAY_PART"] = get_day_part_vectorized(df_adv["SCHEDULED_HOUR"])
        
    if "IS_WEEKEND" not in df_adv.columns:
        df_adv["IS_WEEKEND"] = df_adv["DAY_OF_WEEK"].isin([5, 6]).astype(int)
        
    return df_adv
