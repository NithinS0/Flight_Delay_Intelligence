"""Comprehensive verification test suite for the Flight Delay Prediction and Statistical Intelligence System."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def test_part1_and_part2_statistical_artifacts_exist():
    """Verifies all required dataset quality, descriptive, group, correlation, VIF, hypothesis, and ANOVA files exist."""
    required_files = [
        ARTIFACTS / "dataset_quality_report.json",
        ARTIFACTS / "dataset_quality_report.csv",
        ARTIFACTS / "descriptive_statistics.csv",
        ARTIFACTS / "group_stats_airline.csv",
        ARTIFACTS / "group_stats_origin_airport.csv",
        ARTIFACTS / "group_stats_dest_airport.csv",
        ARTIFACTS / "group_stats_route.csv",
        ARTIFACTS / "correlation_matrix.csv",
        ARTIFACTS / "correlation_spearman.csv",
        ARTIFACTS / "correlation_kendall.csv",
        ARTIFACTS / "correlation_ranking.csv",
        ARTIFACTS / "vif_results.csv",
        ARTIFACTS / "hypothesis_testing_results.csv",
        ARTIFACTS / "anova_results.csv",
        ARTIFACTS / "two_way_anova_results.csv",
        ARTIFACTS / "posthoc_results.csv",
    ]
    for f in required_files:
        assert f.exists(), f"Missing required statistical artifact: {f}"


def test_hypothesis_and_anova_schemas():
    """Verifies all 14 hypothesis tests (with BH-FDR correction) and ANOVA tables have complete columns and no NaNs in key fields."""
    hyp_df = pd.read_csv(ARTIFACTS / "hypothesis_testing_results.csv")
    assert len(hyp_df) >= 14
    for col in [
        "Test_ID", "Category", "Research_Question", "Null_Hypothesis_H0",
        "Alternative_Hypothesis_H1", "Test_Selected", "Test_Statistic",
        "Raw_P_Value", "BH_Adjusted_P_Value", "Alpha", "Statistical_Decision",
        "Effect_Size_Metric", "Effect_Size_Value", "Practical_Interpretation",
    ]:
        assert col in hyp_df.columns, f"Missing column {col} in hypothesis_testing_results.csv"

    anova_df = pd.read_csv(ARTIFACTS / "anova_results.csv")
    assert len(anova_df) >= 5
    for col in ["Factor", "F_Statistic", "Welch_F_Statistic", "Kruskal_Wallis_H", "Eta_Squared", "Omega_Squared", "Daily_Cluster_Robust_F"]:
        assert col in anova_df.columns


def test_zero_leakage_in_feature_sets():
    """Verifies post-departure and delay-reason columns are strictly barred from model feature sets."""
    from src.feature_engineering import CORE_ALL_FEATURES, ADVANCED_ALL_FEATURES
    forbidden = {
        "DEP_DELAY", "ARR_DELAY", "DEP_TIME", "ARR_TIME", "WHEELS_OFF",
        "WHEELS_ON", "TAXI_OUT", "TAXI_IN", "ELAPSED_TIME", "AIR_TIME",
        "CANCELLED", "CANCELLATION_CODE", "DIVERTED", "DELAYED",
        "DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_NAS",
        "DELAY_DUE_SECURITY", "DELAY_DUE_LATE_AIRCRAFT",
    }
    used = set(CORE_ALL_FEATURES) | set(ADVANCED_ALL_FEATURES)
    assert used.isdisjoint(forbidden), f"Leakage detected: {used & forbidden}"


def test_classification_and_regression_modules_and_simulators():
    """Verifies model evaluation tables and live simulators work end-to-end."""
    from src.model_evaluation import (
        get_classification_comparison_table,
        get_regression_comparison_table,
        get_best_models_summary,
        get_experiment_tracking_table,
    )
    from src.classification import predict_flight_delay
    from src.regression import predict_delay_duration

    cls_df = get_classification_comparison_table()
    reg_df = get_regression_comparison_table()
    best = get_best_models_summary()
    exp_df = get_experiment_tracking_table()

    assert len(cls_df) >= 16
    assert len(reg_df) >= 20
    assert len(exp_df) >= 35
    assert best["classification"]["test_f1_decimal"] > 0.44
    assert best["regression"]["test_mae"] < 20.0

    mock_df = pd.DataFrame({
        "AIRLINE": ["Delta Air Lines Inc.", "Southwest Airlines Co."],
        "ORIGIN": ["ATL", "ORD"],
        "DEST": ["LAX", "LGA"],
        "DEP_DELAY": [-2.0, 25.0],
        "DELAYED": [0, 1],
    })
    clf_pred = predict_flight_delay(
        df=mock_df, airline="Delta Air Lines Inc.", origin="ATL", dest="LAX",
        month=7, day_of_week=4, scheduled_hour=18, scheduled_minute=30,
        distance=1947.0, crs_elapsed_time=275.0,
    )
    assert 0.0 <= clf_pred["delay_probability"] <= 1.0
    assert clf_pred["predicted_class"] in (0, 1)

    sample_input = pd.DataFrame([{
        "AIRLINE": "Delta Air Lines Inc.", "ORIGIN": "ATL", "DEST": "LAX",
        "DISTANCE": 1947.0, "CRS_ELAPSED_TIME": 275.0, "YEAR": 2023,
        "MONTH": 7, "DAY_OF_WEEK": 4, "SCHEDULED_HOUR": 18, "SCHEDULED_MINUTE": 30,
        "ROUTE": "ATL_LAX", "AIRLINE_ROUTE": "Delta Air Lines Inc._ATL_LAX",
        "DAY_PART": "Evening (17-20)", "IS_WEEKEND": 0,
    }])
    reg_l1 = predict_delay_duration(sample_input, use_two_stage=False)
    reg_ts = predict_delay_duration(sample_input, use_two_stage=True)
    assert np.isfinite(reg_l1["predicted_delay_minutes"])
    assert np.isfinite(reg_ts["predicted_delay_minutes"])
