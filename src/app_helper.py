import os
import json
import streamlit as st
import pandas as pd
from pathlib import Path
from src.data_loader import load_raw_dataset
from src.preprocessing import clean_flight_data
from src.feature_engineering import engineer_advanced_features
from src.ui_theme import apply_theme, sidebar_status

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_TABLE_PATH = ROOT_DIR / "data" / "processed" / "model_table.parquet"
META_PATH = ROOT_DIR / "artifacts" / "metrics" / "data_prep_meta.json"


def inject_aviation_css():
    """Injects the shared design system (see src/ui_theme.py). Kept for backwards compatibility."""
    apply_theme()


def render_sidebar_header(df: pd.DataFrame = None, dataset_name: str = "flight_data.csv"):
    """Renders the compact dataset status card in the sidebar (branding is part of the shared theme)."""
    if df is None:
        return
    total_count = len(df)
    delayed_count = int((df["DEP_DELAY"] >= 15).sum()) if "DEP_DELAY" in df.columns else 0
    horizon = ""
    if "FL_DATE" in df.columns and total_count:
        dates = pd.to_datetime(df["FL_DATE"])
        horizon = f"{dates.min().year}-{dates.max().year}"
    sidebar_status(total_count, delayed_count, dataset_name, horizon)


@st.cache_data(show_spinner=False)
def load_and_preprocess_cached_data():
    """Loads the 2,920,859-row preprocessed model table (or raw flight_data.csv fallback) and caches in memory."""
    if MODEL_TABLE_PATH.exists():
        cols = [
            "FL_DATE", "AIRLINE", "ORIGIN", "DEST", "ROUTE", "AIRLINE_ROUTE",
            "CRS_DEP_TIME", "DEP_DELAY", "DELAYED", "DIVERTED",
            "CRS_ELAPSED_TIME", "DISTANCE", "YEAR", "MONTH", "DAY_OF_WEEK",
            "SCHEDULED_HOUR", "SCHEDULED_MINUTE", "TIME_SIN", "TIME_COS",
            "DAY_PART", "IS_WEEKEND", "HOLIDAY_DIST", "SPLIT"
        ]
        df = pd.read_parquet(MODEL_TABLE_PATH, columns=cols)
        summary = {}
        if META_PATH.exists():
            try:
                with open(META_PATH, "r", encoding="utf-8") as f:
                    summary = json.load(f).get("cleaning_summary", {})
            except Exception:
                summary = {}
        return df, "flight_data.csv", summary

    raw_df, dataset_name = load_raw_dataset()
    cleaned_df, summary = clean_flight_data(raw_df)
    engineered_df = engineer_advanced_features(cleaned_df)
    return engineered_df, dataset_name, summary


def initialize_app_state():
    """Initializes Streamlit session state and loads dataset automatically in the background."""
    inject_aviation_css()

    if "df" not in st.session_state:
        with st.spinner("Loading flight records..."):
            df, dataset_name, summary = load_and_preprocess_cached_data()
            st.session_state.df = df
            st.session_state.dataset_name = dataset_name
            st.session_state.summary = summary
            st.session_state.prediction_history = []

    render_sidebar_header(st.session_state.df, st.session_state.dataset_name)
