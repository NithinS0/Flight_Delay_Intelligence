import streamlit as st
import pandas as pd
from pathlib import Path

from src.app_helper import initialize_app_state
from src.ui_theme import page_header, section, kpi_grid, callout, note
from src.model_evaluation import (
    get_classification_comparison_table,
    get_regression_comparison_table,
    get_best_models_summary,
    get_experiment_tracking_table,
)
from src.visualizations import (
    plot_model_metric_bars,
    plot_confusion_matrix,
)

st.set_page_config(
    page_title="Model Comparison | Flight Delay Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_app_state()
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

page_header(
    "Model Comparison & Selection",
    "Performance Comparison Across the 4 Best Classification & 4 Best Regression Models",
    "Evaluated using chronological time-series splitting (1.78M fit, 504k validation, 632k untouched out-of-time test). "
    "All metrics reflect natural operational distributions without synthetic balancing.",
    eyebrow="4 Classification Models | 4 Regression Models",
)

best_models = get_best_models_summary()
best_cls = best_models.get("classification", {})
best_reg = best_models.get("regression", {})
cls_df = get_classification_comparison_table()
reg_df = get_regression_comparison_table()
exp_df = get_experiment_tracking_table()

tabs = st.tabs([
    "1. Classification Comparison",
    "2. Regression Comparison",
    "3. Experiment Tracking Ledger",
])

# ==============================================================================
# TAB 1: CLASSIFICATION COMPARISON
# ==============================================================================
with tabs[0]:
    section("Classification Models Leaderboard", "Comparison table across the top 4 classification algorithms.")
    req_clf_cols = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    valid_clf_cols = [c for c in req_clf_cols if c in cls_df.columns]
    st.dataframe(cls_df[valid_clf_cols], use_container_width=True, hide_index=True)

    section("Performance Metric Charts", "Visual comparison of key classification metrics across the 4 models.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(plot_model_metric_bars(cls_df, "F1 Score", "F1 Score Comparison (Delayed Class)", "F1 Score"), use_container_width=True)
    with c2:
        st.plotly_chart(plot_model_metric_bars(cls_df, "Recall", "Recall Comparison (Delay Detection Rate)", "Recall (%)"), use_container_width=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.plotly_chart(plot_model_metric_bars(cls_df, "Accuracy", "Accuracy Comparison", "Accuracy (%)"), use_container_width=True)
    with c4:
        st.plotly_chart(plot_model_metric_bars(cls_df, "ROC-AUC", "ROC-AUC Comparison (Ranking Discrimination)", "ROC-AUC"), use_container_width=True)

    if best_cls:
        section("Best Classification Model", f"Champion: {best_cls['name']}")
        kpi_grid([
            {"label": "Accuracy", "value": f"{best_cls['test_accuracy']:.2f}%", "note": "Out-of-Time Test Set", "tone": "blue"},
            {"label": "Precision", "value": f"{best_cls['test_precision']:.2f}%", "note": "Positive Predictive Value", "tone": "teal"},
            {"label": "Recall", "value": f"{best_cls['test_recall']:.2f}%", "note": "Delayed Class Coverage", "tone": "orange"},
            {"label": "F1 Score", "value": f"{best_cls['test_f1_decimal']:.4f}", "note": f"Val F1: {best_cls['val_f1']:.4f}", "tone": "teal"},
        ])

        callout(
            f"**Selection Rationale:** {best_cls['selection_criterion']} "
            f"Trained with 74 pre-departure features and calibrated with decision threshold τ* = {best_cls.get('threshold', 0.237):.3f}.",
            tone="success",
        )

        col_diag1, col_diag2 = st.columns([1.1, 1.0], gap="large")
        with col_diag1:
            st.markdown("#### Champion Confusion Matrix (632,306 Test Flights)")
            st.plotly_chart(plot_confusion_matrix(best_cls["confusion_matrix"]), use_container_width=True)
        with col_diag2:
            st.markdown("#### ROC & Precision-Recall Discrimination Curves")
            p_roc = ARTIFACTS / "ppt_charts" / "clf_06_roc_curves.png"
            p_pr = ARTIFACTS / "ppt_charts" / "clf_07_pr_curves.png"
            if p_roc.exists():
                st.image(str(p_roc), caption="Out-of-Time Multi-Model ROC Curves", use_column_width=True)
            elif p_pr.exists():
                st.image(str(p_pr), caption="Out-of-Time Multi-Model Precision-Recall Curves", use_column_width=True)

# ==============================================================================
# TAB 2: REGRESSION COMPARISON
# ==============================================================================
with tabs[1]:
    section("Regression Models Leaderboard", "Comparison table across the top 4 regression algorithms.")
    req_reg_cols = ["Model", "MAE (minutes)", "RMSE (minutes)", "R² Score", "Median Absolute Error"]
    valid_reg_cols = [c for c in req_reg_cols if c in reg_df.columns]
    st.dataframe(reg_df[valid_reg_cols], use_container_width=True, hide_index=True)

    section("Performance Metric Charts", "Visual comparison of key regression error metrics across the 4 models.")
    r1, r2 = st.columns(2, gap="large")
    with r1:
        st.plotly_chart(plot_model_metric_bars(reg_df, "MAE (minutes)", "Mean Absolute Error Comparison (Lower is Better)", "MAE (minutes)"), use_container_width=True)
    with r2:
        st.plotly_chart(plot_model_metric_bars(reg_df, "RMSE (minutes)", "Root Mean Squared Error Comparison (Lower is Better)", "RMSE (minutes)"), use_container_width=True)

    r3, r4 = st.columns(2, gap="large")
    with r3:
        st.plotly_chart(plot_model_metric_bars(reg_df, "R² Score", "R² Score Comparison", "R² Score"), use_container_width=True)
    with r4:
        st.plotly_chart(plot_model_metric_bars(reg_df, "Median Absolute Error", "Median Absolute Error Comparison (Lower is Better)", "Median AE (minutes)"), use_container_width=True)

    if best_reg:
        section("Best Regression Model", f"Champion: {best_reg['name']}")
        kpi_grid([
            {"label": "MAE", "value": f"{best_reg['test_mae']:.4f} min", "note": f"Val: {best_reg['val_mae']:.4f} min", "tone": "blue"},
            {"label": "RMSE", "value": f"{best_reg['test_rmse']:.4f} min", "note": "Squared Deviation Scale", "tone": "orange"},
            {"label": "R² Score", "value": f"{best_reg['test_r2']:+.4f}", "note": "Relative to Mean Variance", "tone": "teal"},
            {"label": "Median Absolute Error", "value": f"{best_reg['test_median_ae']:.4f} min", "note": "Typical Flight Error", "tone": "teal"},
        ])

        callout(
            f"**Selection Rationale:** {best_reg['selection_criterion']} "
            f"The L1 loss formulation models the conditional median delay, achieving a median absolute error of just {best_reg['test_median_ae']:.2f} minutes.",
            tone="info",
        )

        col_diag_r1, col_diag_r2 = st.columns(2, gap="large")
        p_act = ARTIFACTS / "ppt_charts" / "reg_04_actual_vs_predicted.png"
        p_dist = ARTIFACTS / "ppt_charts" / "reg_05_residual_distribution.png"
        with col_diag_r1:
            st.markdown("#### Actual vs. Predicted Delay")
            if p_act.exists():
                st.image(str(p_act), caption="Actual vs. Predicted Delay (45° Reference Line)", use_column_width=True)
        with col_diag_r2:
            st.markdown("#### Residual Error Distribution")
            if p_dist.exists():
                st.image(str(p_dist), caption="Residual Distribution (Actual - Predicted, Clipped ±60m)", use_column_width=True)

# ==============================================================================
# TAB 3: EXPERIMENT TRACKING LEDGER
# ==============================================================================
with tabs[2]:
    section("Production Experiment Tracking Ledger", "Audit trail of the 8 production machine learning models.")
    task_filter = st.radio("Filter Experiments by Task Type:", ["All (8 Models)", "Classification (4 Models)", "Regression (4 Models)"], horizontal=True)
    if task_filter == "All (8 Models)":
        st.dataframe(exp_df, use_container_width=True, hide_index=True)
    elif "Classification" in task_filter:
        st.dataframe(exp_df[exp_df["Task Type"] == "Classification"], use_container_width=True, hide_index=True)
    else:
        st.dataframe(exp_df[exp_df["Task Type"] == "Regression"], use_container_width=True, hide_index=True)
