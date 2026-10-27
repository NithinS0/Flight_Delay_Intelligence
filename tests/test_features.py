import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import engineer_features, engineer_advanced_features, get_day_part

def test_day_part_mapping():
    assert get_day_part(3) == "Night"
    assert get_day_part(8) == "Morning"
    assert get_day_part(14) == "Afternoon"
    assert get_day_part(20) == "Evening"

def test_feature_engineering_cyclical():
    sample = pd.DataFrame([{
        "FL_DATE": "2023-04-12",
        "AIRLINE": "Delta Air Lines Inc.",
        "ORIGIN": "ATL",
        "DEST": "MCO",
        "CRS_DEP_TIME": 600,
        "DEP_DELAY": 0.0,
        "DISTANCE": 404.0,
        "CRS_ELAPSED_TIME": 90.0,
    }])
    
    eng = engineer_advanced_features(sample)
    
    assert eng.iloc[0]["SCHEDULED_HOUR"] == 6
    assert eng.iloc[0]["SCHEDULED_MINUTE"] == 0
    assert eng.iloc[0]["DAY_PART"] == "Morning"
    assert eng.iloc[0]["ROUTE"] == "ATL_MCO"
    assert eng.iloc[0]["AIRLINE_ROUTE"] == "Delta Air Lines Inc._ATL_MCO"
    assert -1.0 <= eng.iloc[0]["TIME_SIN"] <= 1.0
    assert -1.0 <= eng.iloc[0]["TIME_COS"] <= 1.0
