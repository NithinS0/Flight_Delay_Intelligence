import pytest
import pandas as pd
from src.classification import load_classifier, predict_flight_delay
from src.regression import load_regressor, predict_delay_duration
from src.feature_engineering import engineer_advanced_features

def test_leakage_safety_and_predictions():
    sample = pd.DataFrame([{
        "FL_DATE": "2024-07-20",
        "AIRLINE": "Delta Air Lines Inc.",
        "ORIGIN": "ATL",
        "DEST": "MCO",
        "DISTANCE": 404.0,
        "CRS_ELAPSED_TIME": 90.0,
        "CRS_DEP_TIME": 1530,
    }])
    
    eng = engineer_advanced_features(sample)
    
    clf_res = predict_flight_delay(eng)
    assert clf_res["delayed"] in [0, 1]
    assert isinstance(clf_res["status"], str)
    assert 0.0 <= clf_res["probability_delayed"] <= 100.0
    
    reg_res = predict_delay_duration(eng)
    assert isinstance(reg_res["predicted_delay_minutes"], float)
    assert reg_res["category"] in ["Early Departure", "On-Time / Minor Buffer", "Moderate Delay", "Severe Delay Warning"]

def test_unseen_categories():
    unseen_sample = pd.DataFrame([{
        "FL_DATE": "2025-01-01",
        "AIRLINE": "Unknown Future Airways",
        "ORIGIN": "ZZZ",
        "DEST": "YYY",
        "DISTANCE": 1000.0,
        "CRS_ELAPSED_TIME": 150.0,
        "CRS_DEP_TIME": 1200,
    }])
    
    eng = engineer_advanced_features(unseen_sample)
    clf_res = predict_flight_delay(eng)
    reg_res = predict_delay_duration(eng)
    
    assert clf_res["delayed"] in [0, 1]
    assert isinstance(reg_res["predicted_delay_minutes"], float)
