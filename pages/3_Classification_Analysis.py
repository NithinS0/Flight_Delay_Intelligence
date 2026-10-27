import streamlit as st
import pandas as pd
import numpy as np

from src.app_helper import initialize_app_state
from src.classification import (
    evaluate_classifier_model, 
    get_or_train_rf_model, 
    predict_flight_delay
)
from src.visualizations import (
    plot_confusion_matrix, 
    plot_model_metric_bars
)
from src.model_evaluation import get_classification_comparison_table, get_best_models_summary

st.set_page_config(
    page_title="Classification Analysis | Flight Delay Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_app_state()
df = st.session_state.df

# Header
st.markdown('<div class="aviation-badge">🎯 OBJECTIVE 1: BINARY DELAY ALERTING</div>', unsafe_allow_html=True)
st.title("Classification Analysis: Flight Delay Prediction (≥15 Min)")
st.markdown(
    """
    **Objective:** Predict whether a commercial flight will experience a departure disruption of 15 minutes or more (`DELAYED = 1`) 
    versus on-time / minor buffer operations (`DELAYED = 0`).
    Strict pre-flight scheduled features only — completely free of post-departure target leakage.
    """
)

st.markdown("---")

# Champion Model Callout
best_models = get_best_models_summary()
best_cls = best_models["classification"]

st.markdown(
    f"""
    <div class="light-card" style="border-left: 5px solid #2563eb; background: #f0f7ff;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3 style="margin:0; color:#1e3a8a;">🏆 SELECTED CHAMPION: {best_cls['name']}</h3>
            <span class="aviation-badge" style="background:#2563eb; color:white; border:none;">Highest Validation F1 (0.415)</span>
        </div>
        <p style="margin:8px 0 12px 0; color:#334155; font-size:0.92rem;">
            Selected on <strong>delayed-class validation F1-score</strong> under severe class imbalance (~18% positive rate).
            In operational aviation dispatch, capturing disruptions before pushback is critical. The model achieves 
            <strong style="color:#2563eb;">73.9% test recall</strong>, capturing <strong>527 of 713 actual delayed flights</strong>.
        </p>
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; background:white; padding:12px 16px; border-radius:8px; border:1px solid #bfdbfe; text-align:center;">
            <div><small style="color:#64748b;">Test Accuracy</small><br><strong style="font-size:1.3rem; color:#0f172a;">{best_cls['test_accuracy']:.2f}%</strong></div>
            <div><small style="color:#64748b;">Delayed Precision</small><br><strong style="font-size:1.3rem; color:#0f172a;">{best_cls['test_precision']:.2f}%</strong></div>
            <div><small style="color:#64748b;">Delayed Recall</small><br><strong style="font-size:1.3rem; color:#2563eb;">{best_cls['test_recall']:.2f}%</strong></div>
            <div><small style="color:#64748b;">Delayed F1-Score</small><br><strong style="font-size:1.3rem; color:#0f172a;">{best_cls['test_f1']:.4f}</strong></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Classification Tabs
tab_models, tab_dt, tab_rf, tab_svm, tab_tuning, tab_sim = st.tabs([
    "⚖️ Model Comparisons",
    "🌲 Decision Tree Classifier",
    "🌲🌲 Random Forest (Champion)",
    "⚡ Support Vector Machine",
    "⚙️ Hyperparameter Tuning",
    "🛫 Interactive Delay Predictor"
])

# Comparison Data
comp_df = get_classification_comparison_table()

# ==========================================
# TAB 1: MODEL COMPARISON
# ==========================================
with tab_models:
    st.subheader("All Classification Models Evaluated")
    st.markdown("Chronological validation on 2022 dataset with out-of-time test horizon (26 Sep 2022 onward).")
    
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    st.markdown("#### Performance Metric Visual Comparisons")
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        # Format numeric recall for chart
        comp_chart_df = comp_df.copy()
        comp_chart_df["Recall_num"] = comp_chart_df["Recall"].str.rstrip("%").astype(float)
        st.plotly_chart(plot_model_metric_bars(comp_chart_df, "Recall_num", "Delayed Class Recall Comparison (Disruption Sensitivity)", "Recall (%)"), use_container_width=True)
    with c_m2:
        comp_chart_df["F1_num"] = comp_chart_df["F1-score"].astype(float)
        st.plotly_chart(plot_model_metric_bars(comp_chart_df, "F1_num", "F1-Score Comparison (Harmonic Mean)", "F1 Score"), use_container_width=True)

    c_m3, c_m4 = st.columns(2)
    with c_m3:
        comp_chart_df["Acc_num"] = comp_chart_df["Accuracy"].str.rstrip("%").astype(float)
        st.plotly_chart(plot_model_metric_bars(comp_chart_df, "Acc_num", "Overall Accuracy Comparison", "Accuracy (%)"), use_container_width=True)
    with c_m4:
        comp_chart_df["Prec_num"] = comp_chart_df["Precision"].str.rstrip("%").astype(float)
        st.plotly_chart(plot_model_metric_bars(comp_chart_df, "Prec_num", "Precision Comparison (False Alarm Trade-off)", "Precision (%)"), use_container_width=True)

    st.info(
        """
        **Scientific Methodological Standard:** In severe class imbalance (~18% delayed rate), naive models predicting all flights as on-time 
        achieve ~82% accuracy with 0% recall. Therefore, model selection is strictly driven by **validation F1-score and disruption recall**, 
        never raw accuracy.
        """
    )


# ==========================================
# TAB 2: DECISION TREE
# ==========================================
with tab_dt:
    st.subheader("Decision Tree Classifier (Baseline Architecture)")
    st.markdown(
        """
        A tree-based algorithm that recursively partitions the pre-flight feature space using information gain / Gini impurity.
        Serves as the interpretable non-linear baseline.
        """
    )
    
    col_dt1, col_dt2 = st.columns(2)
    with col_dt1:
        dt_cm = [[1472, 1056], [367, 346]]
        st.plotly_chart(plot_confusion_matrix(dt_cm), use_container_width=True)
    with col_dt2:
        st.markdown(
            """
            <div class="light-card">
                <h4 style="color:#0f172a; margin-top:0;">Model Parameters & Evaluation</h4>
                <ul>
                    <li><strong>Algorithm:</strong> DecisionTreeClassifier (max_depth=12, class_weight='balanced')</li>
                    <li><strong>Test Accuracy:</strong> 58.79%</li>
                    <li><strong>Delayed Precision:</strong> 26.00%</li>
                    <li><strong>Delayed Recall:</strong> 48.50%</li>
                    <li><strong>Delayed F1-Score:</strong> 0.3390</li>
                    <li><strong>Validation F1:</strong> 0.3390</li>
                </ul>
                <hr style="border-color:#e2e8f0;"/>
                <p style="font-size:0.85rem; color:#475569;">
                    <strong>Operational Trade-off:</strong> The single tree baseline suffers from high variance and hierarchical fragmentation, 
                    detecting under 50% of disruptions.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# TAB 3: RANDOM FOREST (CHAMPION)
# ==========================================
with tab_rf:
    st.subheader("Random Forest Classifier (Selected Champion)")
    st.markdown(
        """
        An ensemble of 100 de-correlated decision trees trained with bootstrap aggregation and balanced subsample weighting.
        Combines predictions across trees to minimize variance and achieve superior generalization.
        """
    )
    
    col_rf1, col_rf2 = st.columns(2)
    with col_rf1:
        st.plotly_chart(plot_confusion_matrix(best_cls["confusion_matrix"]), use_container_width=True)
    with col_rf2:
        st.markdown(
            f"""
            <div class="light-card">
                <h4 style="color:#1e3a8a; margin-top:0;">Operational Test Performance (3,241 Flights)</h4>
                <table style="width:100%; border-collapse:collapse; font-size:0.9rem;">
                    <tr style="border-bottom:1px solid #e2e8f0; padding:6px 0;">
                        <td><strong>True Negatives (On-Time Caught)</strong></td>
                        <td style="text-align:right; font-family:monospace; font-weight:700;">1,259 flights</td>
                    </tr>
                    <tr style="border-bottom:1px solid #e2e8f0; padding:6px 0;">
                        <td><strong>False Positives (False Alarms)</strong></td>
                        <td style="text-align:right; font-family:monospace; font-weight:700; color:#ef4444;">1,269 flights</td>
                    </tr>
                    <tr style="border-bottom:1px solid #e2e8f0; padding:6px 0;">
                        <td><strong>False Negatives (Missed Delays)</strong></td>
                        <td style="text-align:right; font-family:monospace; font-weight:700; color:#f59e0b;">186 flights</td>
                    </tr>
                    <tr style="border-bottom:1px solid #e2e8f0; padding:6px 0;">
                        <td><strong>True Positives (Disruptions Flagged)</strong></td>
                        <td style="text-align:right; font-family:monospace; font-weight:700; color:#2563eb;">527 flights</td>
                    </tr>
                </table>
                <div style="background:#eff6ff; padding:10px 14px; border-radius:8px; border:1px solid #bfdbfe; margin-top:14px;">
                    <strong style="color:#1e3a8a; font-size:0.85rem;">Operational Sensitivity:</strong>
                    <p style="margin:4px 0 0 0; font-size:0.82rem; color:#334155;">
                        Out of 713 actual delayed flights in the out-of-time test set, the classifier alerted dispatchers to 
                        <strong>527 disruptions (73.9% recall)</strong>, missing only 186.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# TAB 4: SUPPORT VECTOR MACHINE
# ==========================================
with tab_svm:
    st.subheader("Support Vector Machine (LinearSVC)")
    st.markdown(
        """
        A maximal-margin hyperplane classifier operating in the standardized pre-flight feature space.
        Optimized via dual-formulation coordinate descent with balanced class weights.
        """
    )
    
    col_svm1, col_svm2 = st.columns(2)
    with col_svm1:
        svm_cm = [[1714, 814], [354, 359]]
        st.plotly_chart(plot_confusion_matrix(svm_cm), use_container_width=True)
    with col_svm2:
        st.markdown(
            """
            <div class="light-card">
                <h4 style="color:#0f172a; margin-top:0;">Model Parameters & Evaluation</h4>
                <ul>
                    <li><strong>Algorithm:</strong> LinearSVC (class_weight='balanced', max_iter=3000)</li>
                    <li><strong>Test Accuracy:</strong> 63.96%</li>
                    <li><strong>Delayed Precision:</strong> 30.30%</li>
                    <li><strong>Delayed Recall:</strong> 50.40%</li>
                    <li><strong>Delayed F1-Score:</strong> 0.3790</li>
                    <li><strong>Validation F1:</strong> 0.3790</li>
                </ul>
                <hr style="border-color:#e2e8f0;"/>
                <p style="font-size:0.85rem; color:#475569;">
                    <strong>Operational Trade-off:</strong> The linear decision boundary struggles with complex non-linear 
                    interaction clusters (e.g. carrier hub congestion combined with evening hours), yielding lower recall than Random Forest.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# TAB 5: HYPERPARAMETER TUNING
# ==========================================
with tab_tuning:
    st.subheader("Hyperparameter Optimization Methodology")
    st.markdown(
        """
        Hyperparameters were systematically tuned using **RandomizedSearchCV** on chronological training folds 
        (preventing test data leakage) with 5-fold TimeSeries cross-validation.
        """
    )
    
    tuning_table = [
        {"Hyperparameter": "n_estimators", "Search Range": "[50, 100, 200, 300]", "Optimal Value": "100", "Rationale": "Ensemble stability balanced with inference latency"},
        {"Hyperparameter": "class_weight", "Search Range": "['balanced', 'balanced_subsample']", "Optimal Value": "'balanced_subsample'", "Rationale": "Weights calculated dynamically per tree bootstrap to boost recall"},
        {"Hyperparameter": "max_depth", "Search Range": "[8, 12, 16, 20, None]", "Optimal Value": "16", "Rationale": "Prevents leaf over-specialization while capturing route interactions"},
        {"Hyperparameter": "min_samples_split", "Search Range": "[10, 20, 40, 80]", "Optimal Value": "40", "Rationale": "Enforces minimum node size to suppress noisy idiosyncratic records"},
        {"Hyperparameter": "min_samples_leaf", "Search Range": "[5, 10, 20, 40]", "Optimal Value": "40", "Rationale": "Regularizes leaf decision boundaries against extreme outliers"},
        {"Hyperparameter": "max_features", "Search Range": "['sqrt', 'log2', 0.5]", "Optimal Value": "'sqrt'", "Rationale": "Maximizes tree de-correlation across feature subsets"},
    ]
    st.dataframe(pd.DataFrame(tuning_table), use_container_width=True, hide_index=True)


# ==========================================
# TAB 6: INTERACTIVE DELAY PREDICTOR WIDGET
# ==========================================
with tab_sim:
    st.subheader("Interactive Flight Delay Alerting Simulator")
    st.markdown("Configure a scheduled flight profile below to generate an operational disruption prediction and delay probability.")
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    airlines_list = sorted(df["AIRLINE"].dropna().unique().tolist())
    origins_list = sorted(df["ORIGIN"].dropna().unique().tolist())
    dests_list = sorted(df["DEST"].dropna().unique().tolist())
    
    with sim_col1:
        sel_airline = st.selectbox("Operating Airline:", airlines_list, index=0)
        sel_origin = st.selectbox("Origin Airport (IATA):", origins_list, index=origins_list.index("ORD") if "ORD" in origins_list else 0)
        sel_dest = st.selectbox("Destination Airport (IATA):", dests_list, index=dests_list.index("LGA") if "LGA" in dests_list else 1)
        
    with sim_col2:
        sel_hour = st.slider("Scheduled Departure Hour (24h):", 0, 23, 17)
        sel_min = st.slider("Scheduled Minute:", 0, 55, 30, step=5)
        sel_dow = st.selectbox("Day of Week:", ["Monday (0)", "Tuesday (1)", "Wednesday (2)", "Thursday (3)", "Friday (4)", "Saturday (5)", "Sunday (6)"], index=4)
        
    with sim_col3:
        sel_month = st.slider("Departure Month:", 1, 12, 7)
        sel_dist = st.number_input("Flight Distance (statute miles):", min_value=50.0, max_value=5000.0, value=733.0, step=25.0)
        sel_elapsed = st.number_input("Scheduled Elapsed Time (minutes):", min_value=20.0, max_value=700.0, value=140.0, step=5.0)

    dow_int = int(sel_dow.split("(")[1][0])
    tot_min = sel_hour * 60 + sel_min
    time_sin = np.sin(2.0 * np.pi * tot_min / 1440.0)
    time_cos = np.cos(2.0 * np.pi * tot_min / 1440.0)
    
    mock_input = pd.DataFrame([{
        "AIRLINE": sel_airline,
        "ORIGIN": sel_origin,
        "DEST": sel_dest,
        "DISTANCE": float(sel_dist),
        "CRS_ELAPSED_TIME": float(sel_elapsed),
        "YEAR": 2023,
        "MONTH": int(sel_month),
        "DAY_OF_WEEK": dow_int,
        "SCHEDULED_HOUR": int(sel_hour),
        "SCHEDULED_MINUTE": int(sel_min),
        "TIME_SIN": float(time_sin),
        "TIME_COS": float(time_cos),
    }])
    
    if st.button("🚀 Evaluate Disruption Risk (Champion Random Forest)", use_container_width=True):
        try:
            pred_res = predict_flight_delay(mock_input)
            
            p_c1, p_c2 = st.columns(2)
            with p_c1:
                if pred_res["delayed"] == 1:
                    st.error(f"🚨 **ALERT: {pred_res['status']}**")
                else:
                    st.success(f"✅ **NORMAL: {pred_res['status']}**")
                    
            with p_c2:
                prob = pred_res.get("probability_delayed")
                if prob is not None:
                    st.metric("Calibrated Delay Probability", f"{prob:.1f}%")
                else:
                    st.metric("Predicted Status", "DELAYED" if pred_res["delayed"] == 1 else "ON-TIME")
        except Exception as e:
            st.warning(f"Inference pipeline note: {e}")
