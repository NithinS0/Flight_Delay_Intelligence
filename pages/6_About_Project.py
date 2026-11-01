import streamlit as st
from src.app_helper import initialize_app_state
from src.ui_theme import page_header, section, callout

st.set_page_config(
    page_title="About the Project | Flight Delay Intelligence",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_app_state()

page_header(
    "About the Flight Delay Intelligence System",
    "Methodology, Statistical Theory and System Architecture",
    "Comprehensive scientific and architectural documentation covering dataset governance, causal pre-departure feature engineering, "
    "parametric and non-parametric statistical inference, chronological validation, and production machine learning models.",
    eyebrow="Documentation",
)

col_intro1, col_intro2 = st.columns(2, gap="large")
with col_intro1:
    section("1. Project Introduction & Problem Statement")
    st.markdown(
        """
        Commercial aviation networks operate on tightly coupled aircraft rotation schedules, crew duty-time limits, and airport slot capacities.
        When a flight departs 15 or more minutes late (the **FAA / DOT formal delay threshold**), delays cascade across downstream legs, missed passenger connections, and terminal gate assignments.
        
        **Core Research Objective:** Predict both:
        1. **Binary flight delay status ($\ge 15$ minutes, FAA Standard)** using classification.
        2. **Continuous departure delay duration in minutes (`DEP_DELAY`)** using regression.

        All models use **strictly pre-departure schedule and causal historical network state features** across **3,000,000 commercial flights** (`2,920,859` completed flights, Jan 2019 – Aug 2023), without leaking post-pushback or same-day future information.
        """
    )

with col_intro2:
    section("2. Dataset Governance & Causal Feature Set")
    st.markdown(
        """
        - **Raw Dataset (`flight_data.csv`):** `3,000,000` flights $\\times$ `32` columns (`575.4 MB`), spanning `18` US carriers, `378` airports, and `7,785` routes (`2019-01-01` to `2023-08-31`).
        - **Cleaned Population:** `2,920,859` non-cancelled flights (`516,401` delayed $\ge 15$ min = `17.68%`).
        - **Natural Class Distribution:** The genuine operational class balance is strictly preserved (17.68% delayed flights) across training, validation, and testing partitions without synthetic data modification.
        - **Zero-Leakage Historical Engine:** 58 strictly prior-day rolling delay statistics (`H_*`) computed strictly prior to flight date (`< FL_DATE`) across 1-day, 7-day, 30-day, and cumulative windows with Empirical-Bayes shrinkage ($m = 50$).
        """
    )

section("3. Selected Machine Learning Architecture Suites", "Selected strictly via chronological validation.")
col_alg1, col_alg2, col_alg3 = st.columns(3, gap="large")

with col_alg1:
    st.markdown("#### 📊 Statistical Inference Suite")
    st.markdown(
        """
        - **Descriptive Statistics:** Mean, Median, Mode, 10% Trimmed Mean, Weighted Means, Variance, SD, IQR, MAD, MedAD, CV, Percentiles P1–P99, Skewness (`11.49`), Kurtosis (`243.90`).
        - **Correlation & Collinearity:** Pearson $r$, Spearman $\\rho$, Kendall $\\tau$, and **Variance Inflation Factor (VIF)**.
        - **14 Hypothesis Tests:** Student's $t$, Welch's $t$, Mann-Whitney $U$, Paired $t$, Wilcoxon Signed-Rank, Chi-Square, Fisher's Exact, Shapiro-Wilk, D'Agostino-Pearson $K^2$, KS-test with **Benjamini-Hochberg FDR** correction.
        - **ANOVA Suite:** One-Way ANOVA, Welch's ANOVA, Kruskal-Wallis, **Cluster-Robust Block-Mean ANOVA**, Two-Way ANOVA (`DAY_PART × IS_WEEKEND`), and **Games-Howell Post-Hoc**.
        """
    )

with col_alg2:
    st.markdown("#### 🎯 4 Best Classification Models")
    st.markdown(
        """
        Selected based on validation delayed-class F1-score with validation-optimized decision thresholds:
        1. **XGBoost Classifier (Champion):** Extreme Gradient Boosting with validation-tuned threshold $\\tau^* = 0.2372$ (Test F1 = `0.4522`, ROC-AUC = `0.7068`, Recall = `59.27%`).
        2. **HistGradientBoosting Classifier:** Scikit-learn histogram-binned boosting (Test F1 = `0.4525`, ROC-AUC = `0.7055`).
        3. **LightGBM Classifier:** Highly efficient leaf-wise gradient boosting (Test F1 = `0.4511`, ROC-AUC = `0.7049`).
        4. **Logistic Regression:** Regularized linear probabilistic baseline with OneHotEncoding and StandardScaling (Test F1 = `0.4505`, ROC-AUC = `0.7028`).
        """
    )

with col_alg3:
    st.markdown("#### 📈 4 Best Regression Models")
    st.markdown(
        """
        Selected based on validation Mean Absolute Error (MAE) under robust L1 loss optimization:
        1. **LightGBM Regressor (Champion):** L1-loss minimization predicting conditional median delay (Test MAE = `18.5561 min`, Median AE = `5.1581 min`, RMSE = `57.4162 min`).
        2. **HistGradientBoosting Regressor:** L1-loss boosting with native missing-value support (Test MAE = `18.5962 min`, Median AE = `5.1490 min`).
        3. **XGBoost Regressor:** Robust gradient boosting regressor (Test MAE = `18.6308 min`, Median AE = `5.1655 min`).
        4. **Gradient Boosting Regressor:** Scikit-learn gradient boosting regressor with L1 loss (Test MAE = `18.7290 min`, Median AE = `5.3890 min`).
        """
    )

section("4. End-to-End System Architecture Diagram")
architecture_dot = """
digraph FlightDelayIntelligence {
    graph [rankdir=TB, bgcolor="transparent", nodesep=0.35, ranksep=0.45, fontname="Helvetica"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10.5, margin="0.18,0.10", penwidth=1.4];
    edge [color="#64748b", penwidth=1.4, arrowsize=0.75];

    data_source [label="RAW DATASET: flight_data.csv\\n(3,000,000 Rows × 32 Columns | 2019–2023)", fillcolor="#e0f2fe", color="#0284c7", fontcolor="#0369a1"];

    subgraph cluster_prep {
        label="Data Preparation & Feature Engineering";
        style="dashed"; color="#cbd5e1"; fontcolor="#475569"; fontsize=10;
        clean [label="Cleaning & Quality Audit\\n(2,920,859 Flights | Natural Class Distribution | No Synthetic Sampling)", fillcolor="#ecfeff", color="#06b6d4", fontcolor="#0e7490"];
        causal [label="Causal History Engine (< FL_DATE)\\n(58 Prior-Day Rolling Features + Empirical Bayes Shrinkage)", fillcolor="#ecfeff", color="#06b6d4", fontcolor="#0e7490"];
        split [label="Chronological Split\\nFit: 1.78M | Validation: 504k | Test: 632k (Untouched Out-of-Time)", fillcolor="#f1f5f9", color="#64748b", fontcolor="#334155"];
    }

    subgraph cluster_stats {
        label="Statistical Intelligence Suite";
        style="dashed"; color="#86efac"; fontcolor="#15803d"; fontsize=10;
        desc [label="Descriptive, Tail & VIF Analysis\\n(Weighted Means, MAD, P1–P99, VIF)", fillcolor="#f0fdf4", color="#16a34a", fontcolor="#15803d"];
        hyp [label="14 Hypothesis Tests + ANOVA Suite\\n(BH-FDR, Welch/Kruskal/Cluster ANOVA, Games-Howell)", fillcolor="#f0fdf4", color="#16a34a", fontcolor="#15803d"];
    }

    subgraph cluster_clf {
        label="4 Best Classification Models";
        style="dashed"; color="#93c5fd"; fontcolor="#1d4ed8"; fontsize=10;
        clf_models [label="Top 4 Classifiers:\\n1. XGBoost Classifier (Champion)\\n2. HistGradientBoosting Classifier\\n3. LightGBM Classifier\\n4. Logistic Regression", fillcolor="#eff6ff", color="#2563eb", fontcolor="#1e3a8a"];
    }

    subgraph cluster_reg {
        label="4 Best Regression Models";
        style="dashed"; color="#d8b4fe"; fontcolor="#7e22ce"; fontsize=10;
        reg_models [label="Top 4 Regressors (L1 Loss):\\n1. LightGBM Regressor (Champion)\\n2. HistGradientBoosting Regressor\\n3. XGBoost Regressor\\n4. Gradient Boosting Regressor", fillcolor="#faf5ff", color="#9333ea", fontcolor="#6b21a8"];
    }

    ui [label="Streamlit Professional Analytics Dashboard\\n(6 Interactive Modules, Clean Light Theme, Real Inference Simulators)", fillcolor="#f8fafc", color="#334155", fontcolor="#0f172a", penwidth=2.0];

    data_source -> clean -> causal -> split;
    causal -> desc -> hyp -> ui;
    split -> clf_models -> ui;
    split -> reg_models -> ui;
}
"""
st.graphviz_chart(architecture_dot)

col_fut1, col_fut2 = st.columns(2, gap="large")
with col_fut1:
    section("5. Methodological Limitations")
    st.markdown(
        """
        - **Unobserved Day-of-Operation Shocks:** Pre-departure timetable and prior-day historical features cannot observe real-time convective thunderstorm cells or immediate FAA ground stops occurring right before pushback.
        - **Clustering & Hub Correlation:** Flights departing the same hub on the same afternoon share airport capacity constraints; we account for this by incorporating cluster-robust statistical tests.
        - **Extreme Delay Outliers:** Genuine long delays (> 500 minutes) are retained in the dataset; optimizing L1 loss (MAE) targets the conditional median delay, preventing single extreme delays from destabilizing the regression predictions.
        """
    )

with col_fut2:
    section("6. Future Scope & Operational Enhancements")
    st.markdown(
        """
        - **Real-Time METAR / TAF Weather Feeds:** Ingest surface observations (ceiling, visibility, gusts) at $T - 2\\text{ hours}$ before scheduled pushback.
        - **Tail-Number Rotation Graphs:** Trace inbound aircraft rotations (`TAIL_NUM`) to predict upstream delay propagation into downstream flight legs.
        - **Live Air Traffic Control Advisories:** Stream FAA System Command Center ground delay program (GDP) status directly into feature inputs.
        - **Conformal Prediction Intervals:** Provide calibrated 90% confidence bands ($[\\hat{q}_{0.05}, \\hat{q}_{0.95}]$) alongside point duration estimates.
        """
    )
