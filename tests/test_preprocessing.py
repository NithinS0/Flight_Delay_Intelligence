import pytest
import pandas as pd
from src.preprocessing import clean_flight_data
from src.feature_engineering import engineer_features, engineer_advanced_features

def test_cleaning_pipeline():
    sample_data = pd.DataFrame([
        {
            "FL_DATE": "2022-01-15",
            "AIRLINE": "Delta Air Lines Inc.",
            "ORIGIN": "JFK",
            "DEST": "LAX",
            "CRS_DEP_TIME": 1130,
            "DEP_DELAY": 10.0,
            "CANCELLED": 0.0,
            "DIVERTED": 0.0,
            "DISTANCE": 2475.0,
            "CRS_ELAPSED_TIME": 360.0,
            "FL_NUMBER": 100,
        },
        {
            "FL_DATE": "2022-01-15",
            "AIRLINE": "American Airlines Inc.",
            "ORIGIN": "ORD",
            "DEST": "MIA",
            "CRS_DEP_TIME": 1400,
            "DEP_DELAY": 45.0,
            "CANCELLED": 1.0,  # Cancelled flight
            "DIVERTED": 0.0,
            "DISTANCE": 1197.0,
            "CRS_ELAPSED_TIME": 185.0,
            "FL_NUMBER": 200,
        },
        {
            "FL_DATE": "2022-11-12",
            "AIRLINE": "Delta Air Lines Inc.",
            "ORIGIN": "GRR",
            "DEST": "MSP",
            "CRS_DEP_TIME": 730,
            "DEP_DELAY": -10.0,
            "CANCELLED": 0.0,
            "DIVERTED": 0.0,
            "DISTANCE": 408.0,
            "CRS_ELAPSED_TIME": 96.0,
            "FL_NUMBER": 2706,  # Incomplete record
        },
        {
            "FL_DATE": "2022-03-20",
            "AIRLINE": "SkyWest Airlines Inc.",
            "ORIGIN": "DEN",
            "DEST": "SLC",
            "CRS_DEP_TIME": 1600,
            "DEP_DELAY": 5.0,
            "CANCELLED": 0.0,
            "DIVERTED": 1.0,  # Non-cancelled diverted flight
            "DISTANCE": 391.0,
            "CRS_ELAPSED_TIME": 90.0,
            "FL_NUMBER": 300,
        }
    ])
    
    clean_df, summary = clean_flight_data(sample_data)
    
    # 1 cancelled and 1 incomplete removed; 1 diverted retained; total = 2
    assert len(clean_df) == 2
    assert summary["cancelled_flights"] == 1
    assert summary["incomplete_records_excluded"] == 1
    assert summary["diverted_retained"] == 1

def test_feature_engineering_dimensions():
    sample_data = pd.DataFrame([{
        "FL_DATE": "2023-05-10",
        "AIRLINE": "United Air Lines Inc.",
        "ORIGIN": "EWR",
        "DEST": "SFO",
        "CRS_DEP_TIME": 845,
        "DEP_DELAY": 20.0,
        "DISTANCE": 2565.0,
        "CRS_ELAPSED_TIME": 380.0,
    }])
    
    eng_df = engineer_advanced_features(sample_data)
    
    assert "YEAR" in eng_df.columns
    assert "MONTH" in eng_df.columns
    assert "DAY_OF_WEEK" in eng_df.columns
    assert "SCHEDULED_HOUR" in eng_df.columns
    assert "SCHEDULED_MINUTE" in eng_df.columns
    assert "TIME_SIN" in eng_df.columns
    assert "TIME_COS" in eng_df.columns
    assert "ROUTE" in eng_df.columns
    assert "AIRLINE_ROUTE" in eng_df.columns
    assert "DAY_PART" in eng_df.columns
    assert "IS_WEEKEND" in eng_df.columns
    assert "DELAYED" in eng_df.columns
    
    assert eng_df.iloc[0]["SCHEDULED_HOUR"] == 8
    assert eng_df.iloc[0]["SCHEDULED_MINUTE"] == 45
    assert eng_df.iloc[0]["DELAYED"] == 1
    assert eng_df.iloc[0]["DAY_PART"] == "Morning"
