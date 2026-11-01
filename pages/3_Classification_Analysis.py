import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.app_helper import initialize_app_state
from src import ui_data as data
from src.ui_theme import page_header, section, kpi_grid, callout, note
from src.classification import (
    load_all_classifier_records,
    get_classifier_record_by_name,
    predict_flight_delay,
)
from src.visualizations import (
    plot_confusion_matrix,
    plot_feature_importance_horizontal,
    plot_model_metric_bars,
    plot_predicted_probability_distribution,
    plot_roc_curve_interactive,
    plot_pr_curve_interactive,
)
from src.model_evaluation import (
    get_classification_comparison_table,
    get_best_models_summary,
)

st.set_page_config(
    page_title="Classification Analysis | Flight Delay Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_app_state()
df = st.session_state.df
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
PPT_CHARTS = ARTIFACTS / "ppt_charts"

page_header(
    "Classification Analysis: Flight Delay Prediction",
    "Pre-Departure Disruption Prediction (Delayed ≥ 15 Minutes)",
    "Evaluates the top 4 classification algorithms on natural, unmanipulated flight operations data. "
    "All models are trained with chronological validation and evaluated on the untouched out-of-time test partition (632,306 flights).",
    eyebrow="4 Classification Models",
)

best_models = get_best_models_summary()
best_cls = best_models.get("classification", {})
all_records = load_all_classifier_records()
comp_df = get_classification_comparison_table()

# Load test set prediction sample (10,000 real out-of-time test flights)
clf_preds = data.classification_predictions()

# Helper function to get probability column for selected classifier
def _get_clf_prob_column(model_name: str) -> str:
    m_lower = model_name.lower()
    if "hist" in m_lower:
        return "Prob_HistGradientBoosting"
    elif "light" in m_lower:
        return "Prob_LightGBM"
    elif "logistic" in m_lower:
        return "Prob_Logistic_Regression"
    return "Prob_XGBoost"

# Champion Callout
if best_cls:
    callout(
        f"**Champion Classifier: {best_cls['name']}** — {best_cls['selection_criterion']} "
        f"Test F1: **{best_cls['test_f1_decimal']:.4f}** | ROC-AUC: **{best_cls['test_roc_auc_decimal']:.4f}** | "
        f"Recall: **{best_cls['test_recall']:.2f}%** | Precision: **{best_cls['test_precision']:.2f}%** | "
        f"Accuracy: **{best_cls['test_accuracy']:.2f}%**.",
        tone="success",
    )

tabs = st.tabs([
    "1. Model Inspector & Diagnostics",
    "2. 4 Models Comparison",
    "3. ROC, PR & Threshold Curves",
    "4. Feature Importance & Selection",
    "5. Flight Delay Simulator",
])

# ==============================================================================
# TAB 1: MODEL INSPECTOR
# ==============================================================================
with tabs[0]:
    section("Classification Model Inspector", "Detailed performance evaluation, confusion matrix, and probability separation for the selected model.")
    model_names = [r["model"] for r in all_records] if all_records else comp_df["Model"].tolist()
    sel_model_name = st.selectbox("Select Classification Model to Inspect:", model_names, index=0)
    rec = get_classifier_record_by_name(sel_model_name)

    if rec:
        tm = rec["test"]
        vm = rec["val"]
        thr = float(tm.get("threshold", 0.2372))

        # 6 Core KPI Cards
        kpi_grid([
            {"label": "Accuracy", "value": f"{tm['accuracy']*100:.2f}%", "note": f"Val: {vm['accuracy']*100:.2f}%", "tone": "blue"},
            {"label": "Precision", "value": f"{tm['precision']*100:.2f}%", "note": f"Val: {vm['precision']*100:.2f}%", "tone": "teal"},
            {"label": "Recall", "value": f"{tm['recall']*100:.2f}%", "note": f"Val: {vm['recall']*100:.2f}%", "tone": "orange"},
            {"label": "F1 Score", "value": f"{tm['f1']:.4f}", "note": f"Val: {vm['f1']:.4f}", "tone": "teal"},
            {"label": "ROC-AUC", "value": f"{tm['roc_auc']:.4f}", "note": f"Val: {vm['roc_auc']:.4f}", "tone": "blue"},
            {"label": "PR-AUC", "value": f"{tm['pr_auc']:.4f}", "note": f"Val: {vm['pr_auc']:.4f}", "tone": "teal"},
        ])

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        col_cm, col_prob = st.columns([1.0, 1.15], gap="large")
        with col_cm:
            st.plotly_chart(plot_confusion_matrix(tm["confusion_matrix"]), use_container_width=True)

        with col_prob:
            # Probability distribution of predictions
            if not clf_preds.empty and "DELAYED" in clf_preds.columns:
                p_col = _get_clf_prob_column(sel_model_name)
                if p_col in clf_preds.columns:
                    y_true = clf_preds["DELAYED"].values
                    y_prob = clf_preds[p_col].values
                    st.plotly_chart(
                        plot_predicted_probability_distribution(y_true, y_prob, threshold=thr, model_name=sel_model_name),
                        use_container_width=True,
                    )

        # Classification Report Table
        cm_mat = tm["confusion_matrix"]
        tn, fp = int(cm_mat[0][0]), int(cm_mat[0][1])
        fn, tp = int(cm_mat[1][0]), int(cm_mat[1][1])
        n_tot = tn + fp + fn + tp
        n_neg = tn + fp
        n_pos = fn + tp
        prec_0 = tn / max(tn + fn, 1)
        rec_0 = tn / max(n_neg, 1)
        f1_0 = 2 * prec_0 * rec_0 / max(prec_0 + rec_0, 1e-12)

        st.markdown("#### Classification Report (Out-of-Time Test Set)")
        rep_df = pd.DataFrame([
            {"Class": "0 — On-Time (< 15 min)", "Precision": f"{prec_0*100:.2f}%", "Recall": f"{rec_0*100:.2f}%", "F1 Score": f"{f1_0:.4f}", "Support": f"{n_neg:,}"},
            {"Class": "1 — Delayed (≥ 15 min)", "Precision": f"{tm['precision']*100:.2f}%", "Recall": f"{tm['recall']*100:.2f}%", "F1 Score": f"{tm['f1']:.4f}", "Support": f"{n_pos:,}"},
            {"Class": "Macro Average", "Precision": f"{(prec_0 + tm['precision'])/2*100:.2f}%", "Recall": f"{tm['balanced_accuracy']*100:.2f}%", "F1 Score": f"{(f1_0 + tm['f1'])/2:.4f}", "Support": f"{n_tot:,}"},
        ])
        st.dataframe(rep_df, use_container_width=True, hide_index=True)
        note(
            f"Evaluated on {n_tot:,} flights with natural operational distribution ({n_pos:,} delayed flights = {n_pos/n_tot*100:.2f}%). "
            f"Decision threshold: τ* = {thr:.3f}."
        )

# ==============================================================================
# TAB 2: 4 MODELS COMPARISON
# ==============================================================================
with tabs[1]:
    section("4 Classification Models Comparison", "Side-by-side benchmark across the top 4 classification algorithms.")
    display_cols = ["Rank", "Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Status"]
    valid_cols = [c for c in display_cols if c in comp_df.columns]
    st.dataframe(comp_df[valid_cols], use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(plot_model_metric_bars(comp_df, "F1 Score", "F1 Score Comparison (Delayed Class)", "F1 Score"), use_container_width=True)
    with c2:
        st.plotly_chart(plot_model_metric_bars(comp_df, "Recall", "Recall Comparison (Detection Sensitivity)", "Recall (%)"), use_container_width=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.plotly_chart(plot_model_metric_bars(comp_df, "Accuracy", "Overall Accuracy Comparison", "Accuracy (%)"), use_container_width=True)
    with c4:
        st.plotly_chart(plot_model_metric_bars(comp_df, "ROC-AUC", "ROC-AUC Discrimination Comparison", "ROC-AUC"), use_container_width=True)

# ==============================================================================
# TAB 3: ROC, PR & THRESHOLD OPTIMIZATION
# ==============================================================================
with tabs[2]:
    section("ROC, Precision-Recall & Threshold Analysis", "Interactive discrimination curves and validation-based threshold optimization.")

    # Interactive ROC & PR Curves for selected model
    if not clf_preds.empty and "DELAYED" in clf_preds.columns:
        p_col = _get_clf_prob_column(sel_model_name if "sel_model_name" in locals() else "XGBoost Classifier")
        if p_col in clf_preds.columns:
            y_true = clf_preds["DELAYED"].values
            y_prob = clf_preds[p_col].values
            curr_model = sel_model_name if "sel_model_name" in locals() else "XGBoost Classifier"

            st.markdown(f"#### Interactive Curves ({curr_model})")
            curv1, curv2 = st.columns(2, gap="large")
            with curv1:
                st.plotly_chart(plot_roc_curve_interactive(y_true, y_prob, model_name=curr_model), use_container_width=True)
            with curv2:
                st.plotly_chart(plot_pr_curve_interactive(y_true, y_prob, model_name=curr_model), use_container_width=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    section("Publication Multi-Model Curves & Threshold Optimization", "Full out-of-time test benchmark.")

    p_roc = PPT_CHARTS / "clf_06_roc_curves.png"
    p_pr = PPT_CHARTS / "clf_07_pr_curves.png"
    p_thr = PPT_CHARTS / "clf_10_threshold_analysis.png"
    p_dist = PPT_CHARTS / "clf_01_class_distribution.png"

    r_c1, r_c2 = st.columns(2, gap="large")
    with r_c1:
        if p_roc.exists():
            st.image(str(p_roc), caption="Multi-Model ROC Curves Comparison", use_column_width=True)
        if p_thr.exists():
            st.image(str(p_thr), caption="Decision Threshold Sweep (Precision, Recall, F1 vs. Threshold)", use_column_width=True)

    with r_c2:
        if p_pr.exists():
            st.image(str(p_pr), caption="Multi-Model Precision-Recall Curves Comparison", use_column_width=True)
        if p_dist.exists():
            st.image(str(p_dist), caption="Natural Operational Class Distribution", use_column_width=True)

# ==============================================================================
# TAB 4: FEATURE IMPORTANCE
# ==============================================================================
with tabs[3]:
    section("Feature Importance & Selection", "Attribution analysis based on TreeSHAP and Mutual Information.")
    fs_csv = ARTIFACTS / "feature_selection_classification.csv"
    if fs_csv.exists():
        fs_df = pd.read_csv(fs_csv)
        s_c1, s_c2 = st.columns(2, gap="large")
        with s_c1:
            st.plotly_chart(
                plot_feature_importance_horizontal(
                    fs_df, "Mean_Abs_TreeSHAP_LogOdds",
                    "Top 20 Features by Exact TreeSHAP (Log-Odds Impact)",
                    "Mean |TreeSHAP| (log-odds)",
                ),
                use_container_width=True,
            )
        with s_c2:
            st.plotly_chart(
                plot_feature_importance_horizontal(
                    fs_df.sort_values("Mutual_Information", ascending=False),
                    "Mutual_Information",
                    "Top 20 Features by Mutual Information with Delay",
                    "Mutual Information (nats)",
                ),
                use_container_width=True,
            )
        st.markdown("#### Complete Feature Selection Table")
        st.dataframe(fs_df.head(25), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 5: INTERACTIVE SIMULATOR
# ==============================================================================
with tabs[4]:
    section("Interactive Flight Delay Simulator", "Simulate flight departure delay using the Champion XGBoost Classifier.")
    sim_col1, sim_col2, sim_col3 = st.columns(3, gap="large")
    airlines_list = sorted(df["AIRLINE"].dropna().unique().tolist())
    origins_list = sorted(df["ORIGIN"].dropna().unique().tolist())
    dests_list = sorted(df["DEST"].dropna().unique().tolist())

    with sim_col1:
        sel_airline = st.selectbox("Operating Airline:", airlines_list, index=0)
        sel_origin = st.selectbox("Origin Airport (IATA):", origins_list, index=origins_list.index("ORD") if "ORD" in origins_list else 0)
        sel_dest = st.selectbox("Destination Airport (IATA):", dests_list, index=dests_list.index("LGA") if "LGA" in dests_list else 1)

    with sim_col2:
        sel_hour = st.slider("Scheduled Departure Hour (0–23):", 0, 23, 18)
        sel_min = st.slider("Scheduled Minute (0–59):", 0, 55, 30, step=5)
        sel_dow_str = st.selectbox("Day of Week:", ["Monday (0)", "Tuesday (1)", "Wednesday (2)", "Thursday (3)", "Friday (4)", "Saturday (5)", "Sunday (6)"], index=4)
        sel_dow = int(sel_dow_str.split("(")[1].rstrip(")"))

    with sim_col3:
        sel_month = st.slider("Departure Month (1–12):", 1, 12, 7)
        sel_dist = st.number_input("Flight Distance (miles):", min_value=50.0, max_value=5000.0, value=733.0, step=25.0)
        sel_elapsed = st.number_input("Scheduled Flight Duration (minutes):", min_value=20.0, max_value=700.0, value=140.0, step=5.0)

    if st.button("Run Pre-Departure Disruption Prediction", type="primary", use_container_width=True):
        pred = predict_flight_delay(
            flight_data=None,
            airline=sel_airline,
            origin=sel_origin,
            dest=sel_dest,
            month=sel_month,
            day_of_week=sel_dow,
            scheduled_hour=sel_hour,
            scheduled_minute=sel_min,
            distance=sel_dist,
            crs_elapsed_time=sel_elapsed,
        )
        tone = "orange" if pred["predicted_class"] == 1 else "teal"
        kpi_grid([
            {"label": "Predicted Status", "value": pred["status_label"], "note": f"Threshold τ* = {pred['threshold']:.3f}", "tone": tone},
            {"label": "Calibrated Delay Probability", "value": f"{pred['delay_probability']*100:.1f}%", "note": f"Risk Tier: {pred['risk_level']}", "tone": tone},
            {"label": "Model Used", "value": "XGBoost Classifier", "note": "Champion Classifier", "tone": "blue"},
            {"label": "Recommended Action", "value": "Turnaround Buffer Alert" if pred["predicted_class"] == 1 else "Standard Pushback", "note": "Operational Guidance", "tone": tone},
        ])
