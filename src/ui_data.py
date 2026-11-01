"""Cached data-access layer for the Streamlit UI.

All numbers shown in the dashboard are read from the real dataset or from the
persisted result artifacts. Nothing here re-trains models or changes statistics;
it only caches reads and cheap aggregations so page navigation stays fast.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT_DIR / "artifacts"
METRICS = ARTIFACTS / "metrics"

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------- #
# Generic artifact readers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def read_json(rel_path: str, _mtime: float = 0.0) -> Dict[str, Any]:
    p = ARTIFACTS / rel_path
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def read_csv(rel_path: str, _mtime: float = 0.0) -> pd.DataFrame:
    p = ARTIFACTS / rel_path
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def json_artifact(rel_path: str) -> Dict[str, Any]:
    p = ARTIFACTS / rel_path
    return read_json(rel_path, p.stat().st_mtime if p.exists() else 0.0)


def csv_artifact(rel_path: str) -> pd.DataFrame:
    p = ARTIFACTS / rel_path
    return read_csv(rel_path, p.stat().st_mtime if p.exists() else 0.0)


# --------------------------------------------------------------------------- #
# Dataset aggregations (computed once per session, cached across pages)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def overview_stats(_df: pd.DataFrame, n_rows: int) -> Dict[str, Any]:
    delay = _df["DEP_DELAY"]
    delayed = int((delay >= 15).sum())
    total = int(len(_df))
    return {
        "total": total,
        "delayed": delayed,
        "on_time": total - delayed,
        "delayed_pct": delayed / total * 100.0 if total else 0.0,
        "mean": float(delay.mean()),
        "median": float(delay.median()),
        "airlines": int(_df["AIRLINE"].nunique()),
        "airports": int(_df["ORIGIN"].nunique()),
        "routes": int(_df["ROUTE"].nunique()),
        "date_min": str(pd.to_datetime(_df["FL_DATE"]).min().date()),
        "date_max": str(pd.to_datetime(_df["FL_DATE"]).max().date()),
        "early_pct": float((delay < 0).mean() * 100.0),
        "le0_pct": float((delay <= 0).mean() * 100.0),
        "ge60_count": int((delay >= 60).sum()),
        "ge60_pct": float((delay >= 60).mean() * 100.0),
    }


@st.cache_data(show_spinner=False)
def monthly_series(_df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    g = (
        _df.assign(_D=(_df["DEP_DELAY"] >= 15).astype("int8"))
        .groupby(["YEAR", "MONTH"], observed=True)
        .agg(flights=("DEP_DELAY", "size"), mean_delay=("DEP_DELAY", "mean"), delayed=("_D", "sum"))
        .reset_index()
    )
    g["period"] = pd.to_datetime(dict(year=g["YEAR"].astype(int), month=g["MONTH"].astype(int), day=1))
    g["delayed_pct"] = g["delayed"] / g["flights"] * 100.0
    return g.sort_values("period").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def split_table(_df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    if "SPLIT" not in _df.columns:
        return pd.DataFrame()
    g = (
        _df.groupby("SPLIT", observed=True)
        .agg(
            flights=("DEP_DELAY", "size"),
            start=("FL_DATE", "min"),
            end=("FL_DATE", "max"),
            delayed_pct=("DELAYED", "mean"),
            mean_delay=("DEP_DELAY", "mean"),
        )
        .reset_index()
    )
    order = {"fit": 0, "val": 1, "test": 2}
    g["_o"] = g["SPLIT"].map(order)
    g = g.sort_values("_o").drop(columns="_o")
    g["delayed_pct"] = g["delayed_pct"] * 100.0
    return g.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def delay_bands(_df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    from src.statistical_analysis import compute_delay_categories

    return compute_delay_categories(_df)


@st.cache_data(show_spinner=False)
def temporal(_df: pd.DataFrame, n_rows: int) -> Dict[str, pd.DataFrame]:
    from src.statistical_analysis import compute_temporal_analysis

    return compute_temporal_analysis(_df)


@st.cache_data(show_spinner=False)
def airline_table(_df: pd.DataFrame, n_rows: int, min_samples: int = 100) -> pd.DataFrame:
    from src.statistical_analysis import compute_airline_analysis

    return compute_airline_analysis(_df, min_samples=min_samples)


@st.cache_data(show_spinner=False)
def airport_tables(_df: pd.DataFrame, n_rows: int, min_flights: int) -> Dict[str, pd.DataFrame]:
    from src.statistical_analysis import compute_airport_analysis

    return {
        "origin": compute_airport_analysis(_df, "ORIGIN", min_samples=min_flights),
        "dest": compute_airport_analysis(_df, "DEST", min_samples=min_flights),
    }


@st.cache_data(show_spinner=False)
def route_table(_df: pd.DataFrame, n_rows: int, min_flights: int) -> pd.DataFrame:
    from src.statistical_analysis import compute_route_analysis

    return compute_route_analysis(_df, min_samples=min_flights)


@st.cache_data(show_spinner=False)
def outlier_summary(_df: pd.DataFrame, n_rows: int) -> Dict[str, Any]:
    from src.eda import compute_outlier_analysis

    return compute_outlier_analysis(_df)


@st.cache_data(show_spinner=False)
def figure(kind: str, _df: pd.DataFrame, n_rows: int):
    """Builds (and caches) the heavier Plotly figures that depend on the full dataframe."""
    from src import visualizations as viz

    builders = {
        "histogram": viz.plot_delay_histogram,
        "boxplot": viz.plot_delay_boxplot,
        "kde": viz.plot_kde_distribution,
        "violin": viz.plot_violin_by_daypart,
        "ecdf": viz.plot_ecdf_delay,
        "qq": viz.plot_qq_normality,
        "categories": viz.plot_delay_categories_bar,
    }
    return builders[kind](_df)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_p(p) -> str:
    """Formats a p-value; exact zeros (below float precision) are shown as '< 1e-300'."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "n/a"
    if np.isnan(p):
        return "n/a"
    if p == 0.0:
        return "< 1e-300"
    return f"{p:.3e}" if p < 1e-3 else f"{p:.4f}"


def fmt_num(x, digits: int = 2, suffix: str = "") -> str:
    try:
        return f"{float(x):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


@st.cache_data(show_spinner=False)
def top_groups(_df: pd.DataFrame, n_rows: int, column: str, k: int = 25) -> list:
    return _df[column].astype(str).value_counts().head(k).index.tolist()


@st.cache_data(show_spinner=False)
def two_group_test(_df: pd.DataFrame, n_rows: int, column: str, a: str, b: str) -> Dict[str, Any]:
    from src.hypothesis_testing import run_custom_two_group_test

    return run_custom_two_group_test(_df, column, a, b)


@st.cache_data(show_spinner=False)
def scatter_matrix_sample(_df: pd.DataFrame, n_rows: int, columns: tuple, sample: int = 3000) -> pd.DataFrame:
    cols = [c for c in columns if c in _df.columns]
    return _df[cols].sample(min(sample, len(_df)), random_state=42).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Descriptive statistics (real artifact keys)
# --------------------------------------------------------------------------- #
def dep_delay_descriptives() -> Dict[str, Any]:
    return json_artifact("metrics/descriptive_statistics.json").get("dep_delay", {})


def quality_report() -> Dict[str, Any]:
    return json_artifact("metrics/dataset_quality_report.json")


# --------------------------------------------------------------------------- #
# Model artifacts
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def classifier_records() -> list:
    from src.classification import load_all_classifier_records

    return load_all_classifier_records()


@st.cache_data(show_spinner=False)
def regressor_records() -> list:
    from src.regression import load_all_regressor_records

    return load_all_regressor_records()


@st.cache_data(show_spinner=False)
def classification_predictions() -> pd.DataFrame:
    df_p = csv_artifact("classification_predictions.csv")
    if df_p.empty:
        df_p = csv_artifact("test_predictions_classification.csv")
    return df_p


@st.cache_data(show_spinner=False)
def regression_predictions() -> pd.DataFrame:
    df_p = csv_artifact("regression_predictions.csv")
    if df_p.empty:
        df_p = csv_artifact("test_predictions_regression.csv")
    return df_p


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def classifier_probability_column(model_name: str, columns) -> Optional[str]:
    """Maps a classifier name to its stored test-probability column, if any."""
    probs = [c for c in columns if c.startswith("Prob_")]
    key = _norm(model_name)
    rules = [
        ("xgboost", "Prob_XGBoost"),
        ("histgradientboosting", "Prob_HistGradientBoosting"),
        ("lightgbm", "Prob_LightGBM"),
        ("logisticregression", "Prob_Logistic_Regression"),
        ("xgboost", "Prob_XGBoost_Tuned"),
        ("lightgbm", "Prob_LightGBM_Tuned"),
    ]
    for token, col in rules:
        if token in key and col in probs:
            return col
    return probs[0] if probs else None


def regressor_prediction_column(model_name: str, columns) -> Optional[str]:
    preds = [c for c in columns if c.startswith("Pred_")]
    key = _norm(model_name)
    rules = [
        ("lightgbm", "Pred_LightGBM_L1_min"),
        ("histgradientboosting", "Pred_HistGradientBoosting_L1_min"),
        ("xgboost", "Pred_XGBoost_L1_min"),
        ("gradientboosting", "Pred_Gradient_Boosting_L1_min"),
        ("lightgbm", "Pred_LightGBM_L1_MAE_Champion_min"),
        ("histgb", "Pred_HistGB_L1_min"),
    ]
    for token, col in rules:
        if token in key and col in preds:
            return col
    return preds[0] if preds else None
