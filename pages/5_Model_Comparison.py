import streamlit as st
import pandas as pd
from src.app_helper import initialize_app_state
from src.model_evaluation import (
    get_classification_comparison_table, 
    get_regression_comparison_table, 
    get_best_models_summary
)
from src.visualizations import (
    plot_model_metric_bars, 
    plot_confusion_matrix
)

st.set_page_config(
    page_title="Model Comparison | Flight Delay Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_app_state()

# Page Header
st.markdown('<div class="aviation-badge">⚖️ SYSTEM BENCHMARK MATRIX</div>', unsafe_allow_html=True)
st.title("Model Comparison: Classification & Regression Benchmarks")
st.markdown(
    """
    Dedicated comprehensive comparison of all evaluated machine learning models across both operational objectives.
    Models are validated chronologically using TimeSeries cross-validation to guarantee zero target leakage.
    Classification accuracy is strictly separated from regression MAE to maintain scientific integrity.
    """
)

st.markdown("---")

best_models = get_best_models_summary()
best_cls = best_models["classification"]
best_reg = best_models["regression"]

# Best Model Summary (Two Separate Highlighted Sections)
st.markdown("### Selected Best-Performing Models")

col_best1, col_best2 = st.columns(2)

with col_best1:
    st.markdown(
        f"""
        <div class="light-card" style="border-left: 5px solid #2563eb; background: #f0f7ff; height: 100%;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4 style="margin:0; color:#1e3a8a;">🎯 BEST CLASSIFICATION MODEL</h4>
                <span class="aviation-badge" style="background:#2563eb; color:white; border:none;">Champion</span>
            </div>
            <h3 style="margin:10px 0 6px 0; color:#0f172a;">{best_cls['name']}</h3>
            <p style="font-size:0.85rem; color:#475569; margin-bottom:14px;">
                <strong>Selection Criterion:</strong> {best_cls['selection_criterion']}
            </p>
            <div style="background:white; border:1px solid #bfdbfe; border-radius:8px; padding:12px; margin-bottom:12px;">
                <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
                    <tr><td style="padding:4px 0; color:#64748b;">Best Validation F1:</td><td style="text-align:right; font-weight:700; color:#2563eb;">{best_cls['val_f1']:.4f}</td></tr>
                    <tr><td style="padding:4px 0; color:#64748b;">Test Accuracy:</td><td style="text-align:right; font-weight:700;">{best_cls['test_accuracy']:.2f}%</td></tr>
                    <tr><td style="padding:4px 0; color:#64748b;">Delayed Precision:</td><td style="text-align:right; font-weight:700;">{best_cls['test_precision']:.2f}%</td></tr>
                    <tr><td style="padding:4px 0; color:#64748b;">Delayed Recall:</td><td style="text-align:right; font-weight:700; color:#2563eb;">{best_cls['test_recall']:.2f}%</td></tr>
                    <tr><td style="padding:4px 0; color:#64748b;">Delayed F1-Score:</td><td style="text-align:right; font-weight:700;">{best_cls['test_f1']:.4f}</td></tr>
                </table>
            </div>
            <p style="font-size:0.82rem; color:#64748b; margin:0;">
                Confusion Matrix: <code>{best_cls['confusion_matrix']}</code> (TN: 1,259 | FP: 1,269 | FN: 186 | TP: 527)
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_best2:
    st.markdown(
        f"""
        <div class="light-card" style="border-left: 5px solid #0ea5e9; background: #f0fdf4; height: 100%;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4 style="margin:0; color:#0369a1;">📈 BEST REGRESSION MODEL</h4>
                <span class="aviation-badge" style="background:#0ea5e9; color:white; border:none;">Champion</span>
            </div>
            <h3 style="margin:10px 0 6px 0; color:#0f172a;">{best_reg['name']}</h3>
            <p style="font-size:0.85rem; color:#475569; margin-bottom:14px;">
                <strong>Selection Criterion:</strong> {best_reg['selection_criterion']}
            </p>
            <div style="background:white; border:1px solid #bbf7d0; border-radius:8px; padding:12px; margin-bottom:12px;">
                <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
                    <tr><td style="padding:4px 0; color:#64748b;">Best Validation MAE:</td><td style="text-align:right; font-weight:700; color:#0284c7;">{best_reg['val_mae']:.4f} min</td></tr>
                    <tr><td style="padding:4px 0; color:#64748b;">Test MAE:</td><td style="text-align:right; font-weight:700; color:#16a34a;">{best_reg['test_mae']:.4f} min</td></tr>
                    <tr><td style="padding:4px 0; color:#64748b;">Baseline Mean MAE:</td><td style="text-align:right; font-weight:700; color:#64748b;">{best_reg['baseline_mean_mae']:.4f} min</td></tr>
                    <tr><td style="padding:4px 0; color:#64748b;">MAE Improvement:</td><td style="text-align:right; font-weight:700; color:#16a34a;">+{best_reg['mae_improvement_pct']:.2f}%</td></tr>
                    <tr><td style="padding:4px 0; color:#64748b;">Test RMSE:</td><td style="text-align:right; font-weight:700;">{best_reg['test_rmse']:.4f} min</td></tr>
                    <tr><td style="padding:4px 0; color:#64748b;">Test R² Score:</td><td style="text-align:right; font-weight:700; color:#64748b;">{best_reg['test_r2']:.4f}</td></tr>
                </table>
            </div>
            <p style="font-size:0.82rem; color:#64748b; margin:0;">
                Employs signed logarithmic compression to counter positive right-tail variance.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Part 1: Classification Comparison
st.subheader("1. Classification Models Comparison")
st.markdown("Evaluated on predicting binary delay alerts (`DELAYED = 1` if departure delay ≥ 15 mins).")

cls_df = get_classification_comparison_table()
st.dataframe(cls_df, use_container_width=True, hide_index=True)

st.markdown("#### Classification Metric Visualizations")
cls_chart_df = cls_df.copy()
cls_chart_df["Acc_num"] = cls_chart_df["Accuracy"].str.rstrip("%").astype(float)
cls_chart_df["Prec_num"] = cls_chart_df["Precision"].str.rstrip("%").astype(float)
cls_chart_df["Rec_num"] = cls_chart_df["Recall"].str.rstrip("%").astype(float)
cls_chart_df["F1_num"] = cls_chart_df["F1-score"].astype(float)

col_cv1, col_cv2 = st.columns(2)
with col_cv1:
    st.plotly_chart(plot_model_metric_bars(cls_chart_df, "Acc_num", "Classification Accuracy Comparison (%)", "Accuracy (%)"), use_container_width=True)
with col_cv2:
    st.plotly_chart(plot_model_metric_bars(cls_chart_df, "Prec_num", "Delayed Class Precision Comparison (%)", "Precision (%)"), use_container_width=True)

col_cv3, col_cv4 = st.columns(2)
with col_cv3:
    st.plotly_chart(plot_model_metric_bars(cls_chart_df, "Rec_num", "Delayed Class Recall Comparison (%)", "Recall (%)"), use_container_width=True)
with col_cv4:
    st.plotly_chart(plot_model_metric_bars(cls_chart_df, "F1_num", "Delayed Class F1-Score Comparison", "F1 Score"), use_container_width=True)

st.markdown("---")

# Part 2: Regression Comparison
st.subheader("2. Regression Models Comparison")
st.markdown("Evaluated on continuous departure delay duration estimation in minutes (`DEP_DELAY`).")

reg_df = get_regression_comparison_table()
st.dataframe(reg_df, use_container_width=True, hide_index=True)

st.markdown("#### Regression Metric Visualizations")
col_rv1, col_rv2, col_rv3 = st.columns(3)
with col_rv1:
    st.plotly_chart(plot_model_metric_bars(reg_df, "MAE (min)", "Mean Absolute Error (Lower is Better)", "MAE (minutes)"), use_container_width=True)
with col_rv2:
    st.plotly_chart(plot_model_metric_bars(reg_df, "RMSE (min)", "Root Mean Squared Error (Outlier Sensitive)", "RMSE (minutes)"), use_container_width=True)
with col_rv3:
    st.plotly_chart(plot_model_metric_bars(reg_df, "R²", "R² Score Relative to Mean Baseline", "R² Value"), use_container_width=True)

st.markdown("---")

# Part 3: Operational Synthesis Matrix
st.subheader("3. Dual-Objective Synthesis: Why Classification & Regression Serve Distinct Roles")
st.markdown(
    """
    In commercial flight operations, binary classification and continuous regression address fundamentally different decision timeframes.
    They must never be conflated or judged by the same criterion.
    """
)

synthesis_data = [
    {
        "Operational Dimension": "Decision Objective",
        "Classification (Random Forest)": "Proactive disruption triage (Flag flights requiring ground surveillance)",
        "Regression (Transformed GBR)": "Turnaround buffer planning (Estimate connecting passenger & gate buffers)"
    },
    {
        "Operational Dimension": "Target Variable",
        "Classification (Random Forest)": "DELAYED (Binary: 0 = on-time, 1 = delay ≥ 15 mins)",
        "Regression (Transformed GBR)": "DEP_DELAY (Continuous: -41 to 2,966 minutes)"
    },
    {
        "Operational Dimension": "Champion Metric",
        "Classification (Random Forest)": "Recall: 73.90% | F1-Score: 0.4200",
        "Regression (Transformed GBR)": "MAE: 17.51 min (+15.95% error reduction over baseline)"
    },
    {
        "Operational Dimension": "Tolerance for False Alarms",
        "Classification (Random Forest)": "High tolerance: A false alert prompts gate check with minimal cost",
        "Regression (Transformed GBR)": "Low tolerance: Overestimating delays misleads connecting flight rebooking"
    },
    {
        "Operational Dimension": "Physical Limiting Factor",
        "Classification (Random Forest)": "Class imbalance (~18% positive delay rate)",
        "Regression (Transformed GBR)": "Extreme black-swan variance (>120m mechanical / weather groundings)"
    },
]
st.dataframe(pd.DataFrame(synthesis_data), use_container_width=True, hide_index=True)
