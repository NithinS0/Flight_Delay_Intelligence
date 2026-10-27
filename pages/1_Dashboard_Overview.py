import streamlit as st
import pandas as pd
from src.app_helper import initialize_app_state
from src.eda import (
    compute_descriptive_statistics, 
    compute_airline_analysis, 
    compute_temporal_analysis,
    generate_dynamic_insights
)
from src.visualizations import (
    plot_delay_donut, 
    plot_monthly_trend, 
    plot_airline_bar, 
    plot_hourly_trend
)
from src.model_evaluation import get_best_models_summary

st.set_page_config(
    page_title="Dashboard Overview | Flight Delay Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_app_state()
df = st.session_state.df
summary_meta = st.session_state.get("summary", {})

# Top Aviation Badge & Header
st.markdown('<div class="aviation-badge">✈️ AVIATION INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
st.title("FLIGHT DELAY INTELLIGENCE")
st.subheader("Statistical Analysis, Machine Learning Classification and Regression")

st.markdown(
    """
    An intelligent aviation analytics system that analyses historical flight data, identifies departure delay patterns, 
    and predicts flight delays using machine learning. The platform operates under strict **zero-target-leakage** methodology, 
    evaluating both disruption alerting (classification) and delay duration estimation (regression).
    """
)

st.markdown("---")

# Dynamic KPI Calculations from actual dataset
stats = compute_descriptive_statistics(df)
total_flights = stats["count"]
delayed_flights = stats["delayed_flights"]
nondelayed_flights = stats["non_delayed_flights"]
delayed_pct = stats["delayed_percentage"]
avg_delay = stats["mean"]

airlines_count = df["AIRLINE"].nunique()
origins_count = df["ORIGIN"].nunique()
dest_count = df["DEST"].nunique()

st.markdown("### Executive Performance Indicators")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Flights Analysed", value=f"{total_flights:,}", delta="Excludes Cancellations")
with col2:
    st.metric(label="Delayed Flights (≥15m)", value=f"{delayed_flights:,}", delta=f"{delayed_pct:.2f}% of Total", delta_color="inverse")
with col3:
    st.metric(label="Non-Delayed Flights", value=f"{nondelayed_flights:,}", delta=f"{100 - delayed_pct:.2f}% On-Time")
with col4:
    st.metric(label="Average Departure Delay", value=f"{avg_delay:.2f} min", delta="Preserves Extremes")

col5, col6, col7, col8 = st.columns(4)
with col5:
    st.metric(label="Operating Airlines", value=f"{airlines_count:,}", delta="US Commercial Carriers")
with col6:
    st.metric(label="Origin Airports", value=f"{origins_count:,}", delta="Commercial Gateways")
with col7:
    st.metric(label="Destination Airports", value=f"{dest_count:,}", delta="Arrival Terminals")
with col8:
    st.metric(label="Extreme Delay Max", value=f"{stats['max']:.0f} min", delta="Black-Swan Outlier")

st.markdown("---")

# Main Visualizations Grid
st.markdown("### Macro Delay Distributions & Operational Dynamics")

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    fig_donut = plot_delay_donut(delayed_flights, nondelayed_flights)
    st.plotly_chart(fig_donut, use_container_width=True)

with row1_col2:
    temporal_dict = compute_temporal_analysis(df)
    fig_monthly = plot_monthly_trend(temporal_dict["monthly"])
    st.plotly_chart(fig_monthly, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    airline_df = compute_airline_analysis(df)
    fig_airline = plot_airline_bar(airline_df)
    st.plotly_chart(fig_airline, use_container_width=True)

with row2_col2:
    fig_hourly = plot_hourly_trend(temporal_dict["hourly"])
    st.plotly_chart(fig_hourly, use_container_width=True)

st.markdown("---")

# Best-Performing Models Summary
best_models = get_best_models_summary()
best_cls = best_models["classification"]
best_reg = best_models["regression"]

st.markdown("### Champion Machine Learning Models")

m_col1, m_col2 = st.columns(2)
with m_col1:
    st.markdown(
        f"""
        <div class="light-card" style="border-left: 4px solid #2563eb;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4 style="margin:0; color:#1e3a8a;">🎯 CLASSIFICATION CHAMPION</h4>
                <span class="aviation-badge">Disruption Alerting</span>
            </div>
            <h3 style="margin: 8px 0 4px 0; color:#0f172a;">{best_cls['name']}</h3>
            <p style="font-size:0.85rem; color:#475569; margin-bottom:12px;">{best_cls['selection_criterion']}</p>
            <div style="background:#f1f5f9; padding:12px; border-radius:8px; display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; text-align:center;">
                <div><small style="color:#64748b;">Accuracy</small><br><strong style="color:#0f172a;">{best_cls['test_accuracy']:.1f}%</strong></div>
                <div><small style="color:#64748b;">Precision</small><br><strong style="color:#0f172a;">{best_cls['test_precision']:.1f}%</strong></div>
                <div><small style="color:#64748b;">Recall</small><br><strong style="color:#2563eb;">{best_cls['test_recall']:.1f}%</strong></div>
                <div><small style="color:#64748b;">F1-Score</small><br><strong style="color:#0f172a;">{best_cls['test_f1']:.2f}</strong></div>
            </div>
            <p style="font-size:0.8rem; color:#64748b; margin-top:10px; margin-bottom:0;">
                Confusion Matrix: <strong>{best_cls['confusion_matrix'][0]}</strong> on-time, <strong>{best_cls['confusion_matrix'][1]}</strong> delayed.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with m_col2:
    st.markdown(
        f"""
        <div class="light-card" style="border-left: 4px solid #0ea5e9;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4 style="margin:0; color:#0369a1;">📈 REGRESSION CHAMPION</h4>
                <span class="aviation-badge" style="background:#f0fdf4; color:#16a34a; border-color:#bbf7d0;">Duration Estimation</span>
            </div>
            <h3 style="margin: 8px 0 4px 0; color:#0f172a;">{best_reg['name']}</h3>
            <p style="font-size:0.85rem; color:#475569; margin-bottom:12px;">{best_reg['selection_criterion']}</p>
            <div style="background:#f1f5f9; padding:12px; border-radius:8px; display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; text-align:center;">
                <div><small style="color:#64748b;">Test MAE</small><br><strong style="color:#0284c7;">{best_reg['test_mae']:.2f}m</strong></div>
                <div><small style="color:#64748b;">Baseline MAE</small><br><strong style="color:#64748b;">{best_reg['baseline_mean_mae']:.2f}m</strong></div>
                <div><small style="color:#64748b;">Improvement</small><br><strong style="color:#16a34a;">+{best_reg['mae_improvement_pct']:.1f}%</strong></div>
                <div><small style="color:#64748b;">Test RMSE</small><br><strong style="color:#0f172a;">{best_reg['test_rmse']:.2f}m</strong></div>
            </div>
            <p style="font-size:0.8rem; color:#64748b; margin-top:10px; margin-bottom:0;">
                Compressed extreme skew via signed logarithmic target transform: <code>sign(y)·log1p(|y|)</code>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Key Analytical Findings & Quick Navigation
st.markdown("### Key Analytical Findings")
st.markdown(
    """
    - **Diurnal Disruption Cascade:** Morning departures (06:00–07:00) experience minimal delays (~2.7–3.2 min avg), whereas evening departures (18:00–21:00) experience peak delays (>15–21 min avg) due to compounded aircraft turnaround friction.
    - **Carrier Variance:** Low-cost carriers with tight turnaround buffers show higher average delays compared to legacy legacy hub-and-spoke carriers with greater recovery capacity.
    - **Class Imbalance Realities:** Delay events (≥15m) represent ~18% of flights. Optimizing for recall ensures 73.9% of disruptions are caught in advance, serving dispatch triage.
    - **Regression Metric Integrity:** Mean Absolute Error (MAE) is the primary operational benchmark, delivering a 15.95% error reduction over baseline. Negative R² reflects extreme black-swan outliers that cannot be predicted from scheduled data alone.
    """
)

st.markdown("---")
st.markdown("### Navigation Shortcuts")
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)
with nav_col1:
    if st.button("📊 Statistical Analysis & EDA", use_container_width=True):
        st.switch_page("pages/2_Statistical_Analysis.py")
with nav_col2:
    if st.button("🎯 Classification Analysis", use_container_width=True):
        st.switch_page("pages/3_Classification_Analysis.py")
with nav_col3:
    if st.button("📈 Regression Analysis", use_container_width=True):
        st.switch_page("pages/4_Regression_Analysis.py")
with nav_col4:
    if st.button("⚖️ Model Comparison", use_container_width=True):
        st.switch_page("pages/5_Model_Comparison.py")
with nav_col5:
    if st.button("ℹ️ About the Project", use_container_width=True):
        st.switch_page("pages/6_About_Project.py")
