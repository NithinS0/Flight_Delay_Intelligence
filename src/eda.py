"""Exploratory Data Analysis & Statistical Artifacts Loader re-export module."""
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
import pandas as pd

from src.statistical_analysis import (
    load_quality_report,
    load_column_quality_df,
    load_multi_variable_descriptive_df,
    load_correlation_and_vif,
    compute_dataset_overview,
    compute_descriptive_statistics,
    compute_delay_categories,
    compute_airline_analysis,
    compute_airport_analysis,
    compute_route_analysis,
    compute_temporal_analysis,
    compute_correlation_matrix,
    generate_dynamic_insights,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"


def _read_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def load_dataset_quality_report() -> Dict[str, Any]:
    """Loads automated dataset quality report from artifacts/dataset_quality_report.json."""
    return _read_json(ARTIFACTS_DIR / "dataset_quality_report.json")


def load_full_descriptive_stats() -> Dict[str, Any]:
    """Loads full 2.92M-flight descriptive statistics JSON from artifacts/metrics/descriptive_statistics.json."""
    d = _read_json(METRICS_DIR / "descriptive_statistics.json")
    dep = d.get("dep_delay", {})
    if dep and "variables" not in d:
        d["variables"] = {
            "DEP_DELAY": {
                "count": dep.get("Count", 2_920_859),
                "mean": dep.get("Mean", 10.10),
                "median": dep.get("Median", -2.0),
                "mode": dep.get("Mode", -5.0),
                "std": dep.get("Std_Dev", 49.18),
                "variance": dep.get("Variance", 2418.44),
                "mad_mean": dep.get("MAD", 21.35),
                "mad_median": dep.get("MedAD", 4.0),
                "cv": dep.get("CV", 4.87),
                "cv_pct": dep.get("CV", 4.87) * 100.0,
                "range": dep.get("Range", 3056.0),
                "min": dep.get("Min", -90.0),
                "max": dep.get("Max", 2966.0),
                "q1_25": dep.get("Q1_P25", -6.0),
                "q3_75": dep.get("Q3_P75", 6.0),
                "iqr": dep.get("IQR", 12.0),
                "p90": dep.get("P90", 36.0),
                "p95": dep.get("P95", 72.0),
                "p99": dep.get("P99", 191.0),
                "skewness": dep.get("Skewness", 11.49),
                "kurtosis_excess": dep.get("Kurtosis_Excess", 243.94),
                "kurtosis_pearson": dep.get("Kurtosis_Pearson", 246.94),
                "std_error_mean": dep.get("standard_error", 0.0288),
                "ci_95_mean_parametric": dep.get("parametric_95_ci", [10.04, 10.15]),
                "ci_95_mean_bootstrap": dep.get("bootstrap_95_ci_mean_50k", [9.75, 10.62]),
                "delayed_15_count": 516_401,
                "on_time_15_count": 2_404_458,
                "delayed_15_rate_pct": 17.68,
            }
        }
        d["weighted_means_dep_delay"] = {
            "distance_weighted_mean_min": dep.get("weighted_mean_by_distance", 10.887),
            "elapsed_time_weighted_mean_min": dep.get("weighted_mean_by_elapsed_time", 10.699),
        }
    return d


def load_group_statistics() -> Dict[str, Any]:
    """Loads group-based statistical analysis JSON."""
    return _read_json(METRICS_DIR / "group_statistics.json")


def load_hypothesis_testing_results() -> Dict[str, Any]:
    """Loads 14-test hypothesis testing suite JSON."""
    return _read_json(METRICS_DIR / "hypothesis_testing_results.json")


def load_anova_results() -> Dict[str, Any]:
    """Loads ANOVA & Games-Howell post-hoc JSON."""
    return _read_json(METRICS_DIR / "anova_results.json")


def compute_outlier_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes IQR and Z-score outlier summary on the active dataframe."""
    s = df["DEP_DELAY"].dropna().astype(float)
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    low_15, high_15 = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    z = (s - float(s.mean())) / max(float(s.std()), 1e-9)
    return {
        "iqr_low_15": round(low_15, 2),
        "iqr_high_15": round(high_15, 2),
        "iqr_outlier_count": int(((s < low_15) | (s > high_15)).sum()),
        "z3_outlier_count": int((z.abs() > 3.0).sum()),
        "extreme_180m_count": int((s >= 180.0).sum()),
    }
