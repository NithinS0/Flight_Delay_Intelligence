import os
import streamlit as st
import pandas as pd
from pathlib import Path
from src.data_loader import load_raw_dataset
from src.preprocessing import clean_flight_data
from src.feature_engineering import engineer_advanced_features

def inject_aviation_css():
    """Injects custom Streamlit CSS design system for a modern, clean LIGHT THEME."""
    st.markdown(
        """
        <style>
        /* Base page styling - Light Theme Only */
        .stApp {
            background-color: #ffffff;
            color: #0f172a;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Main container padding */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            padding-left: 2.5rem;
            padding-right: 2.5rem;
            max-width: 1400px;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.04);
            transition: all 0.2s ease-in-out;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #cbd5e1;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07);
        }
        div[data-testid="stMetric"] label {
            color: #475569 !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            font-weight: 700 !important;
            font-size: 1.55rem !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
            font-size: 0.8rem !important;
            font-weight: 500 !important;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        section[data-testid="stSidebar"] div.block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* Custom badge */
        .aviation-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            background: #eff6ff;
            color: #2563eb;
            border: 1px solid #bfdbfe;
            margin-bottom: 12px;
            letter-spacing: 0.02em;
        }

        /* Light Cards and Section Boxes */
        .light-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 22px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.03);
        }
        .light-card:hover {
            border-color: #cbd5e1;
        }

        .highlight-card {
            background: #f0f7ff;
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }

        /* Clean Tab styling */
        button[data-baseweb="tab"] {
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            color: #475569 !important;
            padding: 10px 16px !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #2563eb !important;
            border-bottom-color: #2563eb !important;
        }

        /* Clean Table & Dataframe styling */
        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
        }

        /* Info boxes */
        div[data-testid="stAlert"] {
            border-radius: 10px;
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_sidebar_header(df: pd.DataFrame = None, dataset_name: str = "flight_data.csv"):
    """Renders consistent, professional sidebar branding across all pages."""
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 10px 0 16px 0;">
            <div style="width: 48px; height: 48px; margin: 0 auto 10px auto; border-radius: 14px; background: #2563eb; display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25); color: white;">
                ✈️
            </div>
            <strong style="color: #0f172a; font-size: 1.05rem; letter-spacing: -0.01em;">FLIGHT DELAY INTELLIGENCE</strong>
            <p style="color: #64748b; font-size: 0.78rem; margin-top: 3px; font-weight: 500;">Machine Learning Analytics Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")
    
    if df is not None:
        total_count = len(df)
        delayed_count = int((df["DEP_DELAY"] >= 15).sum()) if "DEP_DELAY" in df.columns else 0
        delayed_pct = (delayed_count / total_count * 100) if total_count > 0 else 0.0
        
        st.sidebar.markdown(f"**Loaded Dataset:** `{dataset_name}`")
        st.sidebar.markdown(f"**Total Records:** `{total_count:,}`")
        st.sidebar.markdown(f"**Delayed (≥15m):** `{delayed_count:,} ({delayed_pct:.2f}%)`")
        st.sidebar.markdown("---")

@st.cache_data(show_spinner=False)
def load_and_preprocess_cached_data():
    """Loads and preprocesses flight_data.csv automatically and caches in memory."""
    raw_df, dataset_name = load_raw_dataset()
    cleaned_df, summary = clean_flight_data(raw_df)
    engineered_df = engineer_advanced_features(cleaned_df)
    return engineered_df, dataset_name, summary

def initialize_app_state():
    """Initializes Streamlit session state and loads dataset automatically."""
    inject_aviation_css()
    
    if "df" not in st.session_state:
        with st.spinner("✈️ Automatically loading and preprocessing flight_data.csv..."):
            df, dataset_name, summary = load_and_preprocess_cached_data()
            st.session_state.df = df
            st.session_state.dataset_name = dataset_name
            st.session_state.summary = summary
            st.session_state.prediction_history = []
            
    render_sidebar_header(st.session_state.df, st.session_state.dataset_name)
