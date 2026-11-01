import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.app_helper import initialize_app_state
from src import ui_data as data
from src.ui_theme import page_header, section, kpi_grid, callout, note
from src.regression import (
    load_all_regressor_records,
    get_regressor_record_by_name,
    predict_delay_duration,
)
from src.visualizations import (
    plot_model_metric_bars,
    plot_feature_importance_horizontal,
    plot_actual_vs_predicted_scatter,
    plot_residuals_distribution,
    plot_residuals_vs_predicted,
    plot_residual_qq,
)
from src.model_evaluation import (
    get_regression_comparison_table,
    get_best_models_summary,
)

st.set_page_config(
    page_title="Regression Analysis | Flight Delay Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_app_state()
df = st.session_state.df
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
PPT_CHARTS = ARTIFACTS / "ppt_charts"

page_header(
    "Regression Analysis: Departure Delay Duration",
    "Continuous Delay Duration Estimation (Minutes)",
    "Evaluates the top 4 regression algorithms designed to predict exact departure delay in minutes. "
    "All models optimize robust L1 loss (MAE minimization) to handle the long right-tail of severe delays without data distortion.",
    eyebrow="4 Regression Models",
)

best_models = get_best_models_summary()
best_reg = best_models.get("regression", {})
all_reg_records = load_all_regressor_records()
reg_df = get_regression_comparison_table()

# Load test set prediction sample (10,000 real out-of-time test flights)
reg_preds = data.regression_predictions()

# Champion Banner
if best_reg:
    callout(
        f"**Champion Regressor: {best_reg['name']}** — {best_reg['selection_criterion']} "
        f"Test MAE: **{best_reg['test_mae']:.4f} min** (-{best_reg['mae_improvement_pct']:.1f}% vs training mean baseline) | "
        f"Median AE: **{best_reg['test_median_ae']:.4f} min** | RMSE: **{best_reg['test_rmse']:.4f} min** | "
        f"Explained Variance: **{best_reg['test_explained_variance']:.4f}**.",
        tone="info",
    )

tabs = st.tabs([
    "1. Model Inspector & Diagnostics",
    "2. 4 Models Comparison",
    "3. Residual Diagnostics & Error Distributions",
    "4. Feature Importance & Attribution",
    "5. Delay Duration Estimator",
])

# Helper function to get prediction column for selected regressor
def _get_reg_pred_column(model_name: str) -> str:
    m_lower = model_name.lower()
    if "hist" in m_lower:
        return "Pred_HistGradientBoosting_L1_min"
    elif "xgb" in m_lower:
        return "Pred_XGBoost_L1_min"
    elif "gradient" in m_lower and "hist" not in m_lower:
        return "Pred_Gradient_Boosting_L1_min"
    return "Pred_LightGBM_L1_min"

# ==============================================================================
# TAB 1: MODEL INSPECTOR
# ==============================================================================
with tabs[0]:
    section("Regression Model Inspector", "Detailed error metrics, actual vs. predicted diagnostics, and residual distributions for the selected model.")
    model_names = [r["model"] for r in all_reg_records] if all_reg_records else reg_df["Model"].tolist()
    sel_reg_name = st.selectbox("Select Regression Model to Inspect:", model_names, index=0)
    rec = get_regressor_record_by_name(sel_reg_name)

    if rec:
        tm = rec["test"]
        vm = rec["val"]

        # 5 Core KPI Cards
        kpi_grid([
            {"label": "MAE (Mean Absolute Error)", "value": f"{tm['mae']:.4f} min", "note": f"Val: {vm['mae']:.4f} min", "tone": "blue"},
            {"label": "RMSE (Root Mean Squared Error)", "value": f"{tm['rmse']:.4f} min", "note": f"Val: {vm['rmse']:.4f} min", "tone": "orange"},
            {"label": "R² Score", "value": f"{tm['r2']:+.4f}", "note": f"Val: {vm['r2']:+.4f}", "tone": "teal"},
            {"label": "Median Absolute Error", "value": f"{tm['median_ae']:.4f} min", "note": f"Val: {vm['median_ae']:.4f} min", "tone": "teal"},
            {"label": "Explained Variance", "value": f"{tm.get('explained_variance', 0.02):+.4f}", "note": f"Val: {vm.get('explained_variance', 0.02):+.4f}", "tone": "blue"},
        ])

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Dynamic Residual Diagnostics for Selected Model
        if not reg_preds.empty and "DEP_DELAY" in reg_preds.columns:
            pred_col = _get_reg_pred_column(sel_reg_name)
            if pred_col in reg_preds.columns:
                y_true = reg_preds["DEP_DELAY"].values
                y_pred = reg_preds[pred_col].values
                residuals = y_true - y_pred

                col_res1, col_res2 = st.columns(2, gap="large")
                with col_res1:
                    st.plotly_chart(plot_actual_vs_predicted_scatter(y_true, y_pred), use_container_width=True)
                with col_res2:
                    st.plotly_chart(plot_residuals_distribution(residuals), use_container_width=True)

        note(
            f"The L1 loss function directly estimates the conditional median of the departure delay distribution, "
            f"achieving a median error of {tm['median_ae']:.2f} minutes on 632,306 untouched test flights."
        )

# ==============================================================================
# TAB 2: 4 MODELS COMPARISON
# ==============================================================================
with tabs[1]:
    section("4 Regression Models Comparison", "Side-by-side benchmark across the top 4 regression models.")
    display_cols = ["Rank", "Model", "MAE (minutes)", "RMSE (minutes)", "R² Score", "Median Absolute Error", "Status"]
    valid_cols = [c for c in display_cols if c in reg_df.columns]
    st.dataframe(reg_df[valid_cols], use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(plot_model_metric_bars(reg_df, "MAE (minutes)", "MAE Comparison (Lower is Better)", "MAE (minutes)"), use_container_width=True)
    with c2:
        st.plotly_chart(plot_model_metric_bars(reg_df, "Median Absolute Error", "Median Absolute Error (Lower is Better)", "Median AE (minutes)"), use_container_width=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.plotly_chart(plot_model_metric_bars(reg_df, "RMSE (minutes)", "RMSE Comparison", "RMSE (minutes)"), use_container_width=True)
    with c4:
        st.plotly_chart(plot_model_metric_bars(reg_df, "R² Score", "R² Score Comparison", "R² Score"), use_container_width=True)

# ==============================================================================
# TAB 3: RESIDUAL DIAGNOSTICS & DISTRIBUTIONS
# ==============================================================================
with tabs[2]:
    section("Residual Diagnostics & Error Distributions", "Comprehensive analysis of model errors, residual patterns, and distribution normality.")

    # Interactive diagnostics based on selected or champion model
    if not reg_preds.empty and "DEP_DELAY" in reg_preds.columns:
        pred_col = _get_reg_pred_column(sel_reg_name if "sel_reg_name" in locals() else "LightGBM Regressor")
        if pred_col in reg_preds.columns:
            y_true = reg_preds["DEP_DELAY"].values
            y_pred = reg_preds[pred_col].values
            residuals = y_true - y_pred

            st.markdown("#### Interactive Diagnostics (Selected Model Predictions)")
            r_diag1, r_diag2 = st.columns(2, gap="large")
            with r_diag1:
                st.plotly_chart(plot_residuals_vs_predicted(y_pred, residuals), use_container_width=True)
            with r_diag2:
                st.plotly_chart(plot_residual_qq(residuals), use_container_width=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    section("Publication Error Diagnostics & Performance Analysis", "Full-sample test evaluations across 632,306 flights.")

    # Verified existing charts from artifacts/ppt_charts/
    p_act = PPT_CHARTS / "reg_04_actual_vs_predicted.png"
    p_dist = PPT_CHARTS / "reg_05_residual_distribution.png"
    p_rvp = PPT_CHARTS / "reg_06_residual_vs_predicted.png"
    p_err = PPT_CHARTS / "reg_07_error_distribution.png"
    p_band = PPT_CHARTS / "reg_08_performance_by_delay_range.png"
    p_qq = PPT_CHARTS / "reg_12_residual_qq_and_shap.png"

    g1, g2 = st.columns(2, gap="large")
    with g1:
        if p_act.exists():
            st.image(str(p_act), caption="Actual vs. Predicted Delay (45° Reference Line)", use_column_width=True)
        if p_rvp.exists():
            st.image(str(p_rvp), caption="Residuals vs. Predicted Delay (Homoscedasticity Check)", use_column_width=True)
        if p_band.exists():
            st.image(str(p_band), caption="MAE Across Delay Ranges (≤ 60m, ≤ 120m, All Flights)", use_column_width=True)

    with g2:
        if p_dist.exists():
            st.image(str(p_dist), caption="Residual Error Distribution (Clipped ±60 min)", use_column_width=True)
        if p_err.exists():
            st.image(str(p_err), caption="Cumulative Error Distribution Across Models", use_column_width=True)
        if p_qq.exists():
            st.image(str(p_qq), caption="Residual Normal Q-Q Plot and Tail Departure", use_column_width=True)

# ==============================================================================
# TAB 4: FEATURE IMPORTANCE
# ==============================================================================
with tabs[3]:
    section("Regression Feature Importance", "Feature attribution for continuous delay duration.")
    fi_reg_csv = ARTIFACTS / "feature_importance_regression.csv"
    if fi_reg_csv.exists():
        fi_reg = pd.read_csv(fi_reg_csv)
        val_col = "Mean_Abs_TreeSHAP_Min" if "Mean_Abs_TreeSHAP_Min" in fi_reg.columns else "importance_pct"
        st.plotly_chart(
            plot_feature_importance_horizontal(
                fi_reg, val_col,
                "Top 20 Regression Features by Impact on Delay Duration",
                "Mean |TreeSHAP| (minutes)" if val_col == "Mean_Abs_TreeSHAP_Min" else "Feature Gain (%)",
            ),
            use_container_width=True,
        )
        st.markdown("#### Complete Feature Importance Ledger")
        st.dataframe(fi_reg.head(25), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 5: DELAY ESTIMATOR
# ==============================================================================
with tabs[4]:
    section("Interactive Delay Duration Estimator", "Predict continuous departure delay in minutes using LightGBM Regressor.")
    r_sim1, r_sim2, r_sim3 = st.columns(3, gap="large")
    airlines_list = sorted(df["AIRLINE"].dropna().unique().tolist())
    origins_list = sorted(df["ORIGIN"].dropna().unique().tolist())
    dests_list = sorted(df["DEST"].dropna().unique().tolist())

    with r_sim1:
        sel_airline_r = st.selectbox("Operating Airline:", airlines_list, index=0, key="r_air")
        sel_origin_r = st.selectbox("Origin Airport (IATA):", origins_list, index=origins_list.index("ORD") if "ORD" in origins_list else 0, key="r_orig")
        sel_dest_r = st.selectbox("Destination Airport (IATA):", dests_list, index=dests_list.index("LAX") if "LAX" in dests_list else 1, key="r_dest")

    with r_sim2:
        sel_hour_r = st.slider("Scheduled Departure Hour (0–23):", 0, 23, 18, key="r_hr")
        sel_min_r = st.slider("Scheduled Minute (0–59):", 0, 55, 45, step=5, key="r_min")
        sel_dow_r = st.selectbox("Day of Week:", ["Monday (0)", "Tuesday (1)", "Wednesday (2)", "Thursday (3)", "Friday (4)", "Saturday (5)", "Sunday (6)"], index=4, key="r_dow")

    with r_sim3:
        sel_month_r = st.slider("Month of Travel (1–12):", 1, 12, 7, key="r_mo")
        sel_dist_r = st.number_input("Flight Distance (miles):", min_value=50.0, max_value=5000.0, value=1744.0, step=50.0, key="r_dist")
        sel_elapsed_r = st.number_input("Scheduled Duration (minutes):", min_value=20.0, max_value=700.0, value=255.0, step=10.0, key="r_elap")

    dow_int_r = int(sel_dow_r.split("(")[1].rstrip(")"))
    mock_reg_input = pd.DataFrame([{
        "AIRLINE": sel_airline_r,
        "ORIGIN": sel_origin_r,
        "DEST": sel_dest_r,
        "DISTANCE": float(sel_dist_r),
        "CRS_ELAPSED_TIME": float(sel_elapsed_r),
        "YEAR": 2023,
        "MONTH": int(sel_month_r),
        "DAY_OF_WEEK": dow_int_r,
        "SCHEDULED_HOUR": int(sel_hour_r),
        "SCHEDULED_MINUTE": int(sel_min_r),
        "ROUTE": f"{sel_origin_r}_{sel_dest_r}",
        "AIRLINE_ROUTE": f"{sel_airline_r}_{sel_origin_r}_{sel_dest_r}",
    }])

    if st.button("Calculate Predicted Departure Delay", type="primary", use_container_width=True):
        pred_r = predict_delay_duration(mock_reg_input)
        d_val = pred_r["predicted_delay_minutes"]
        tone = "teal" if d_val <= 0 else ("blue" if d_val < 15 else "orange")
        kpi_grid([
            {"label": "Predicted Departure Delay", "value": f"{d_val:+.1f} min", "note": "Point Estimate (Conditional Median)", "tone": tone},
            {"label": "Delay Category", "value": pred_r["category"], "note": "Operational Classification", "tone": tone},
            {"label": "Delayed Threshold (≥ 15 min)", "value": "Yes" if pred_r["is_delayed_threshold"] else "No", "note": "FAA Standard", "tone": tone},
            {"label": "Model Used", "value": "LightGBM Regressor", "note": "Champion Regressor (L1 loss)", "tone": "blue"},
        ])
