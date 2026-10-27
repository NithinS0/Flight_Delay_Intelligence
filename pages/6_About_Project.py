import streamlit as st
from src.app_helper import initialize_app_state

st.set_page_config(
    page_title="About the Project | Flight Delay Intelligence",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_app_state()

# Page Header
st.markdown('<div class="aviation-badge">ℹ️ METHODOLOGY & ARCHITECTURE</div>', unsafe_allow_html=True)
st.title("About the Flight Delay Intelligence Platform")
st.markdown(
    """
    Complete scientific and architectural documentation of the Flight Delay Intelligence machine learning system, 
    detailing research objectives, algorithmic foundations, chronological validation methodology, 
    end-to-end system architecture, operational limitations, and roadmap enhancements.
    """
)

st.markdown("---")

# Section A, B, C: Overview, Problem Statement & Objectives
col_intro1, col_intro2 = st.columns(2)

with col_intro1:
    st.subheader("Section A – Project Introduction")
    st.markdown(
        """
        **Flight Delay Intelligence** is an end-to-end machine learning and statistical analysis platform that uses historical 
        airline flight records to identify departure delay patterns and predict flight delays. The platform is designed 
        specifically for aviation operational decision support, providing both **early-warning threshold alerts (Classification)** 
        and **continuous delay duration forecasts (Regression)**.
        """
    )
    
    st.subheader("Section B – Problem Statement")
    st.markdown(
        """
        Flight disruptions impose immense financial and operational costs on commercial aviation, disrupting passenger 
        itineraries, escalating crew overtime costs, creating terminal gate congestion, and triggering cascading delays 
        across airline networks. Predicting delays prior to gate pushback enables airlines and airport operations centers 
        to proactively manage turnaround buffers, re-route connecting passenger luggage, and optimize gate assignments.
        """
    )

with col_intro2:
    st.subheader("Section C – Project Objectives")
    st.markdown(
        """
        - **Analyse Historical Flight Delay Patterns:** Perform comprehensive empirical data analysis on multi-year flight operations.
        - **Understand Airline & Airport Performance:** Quantify operational delay variances across US commercial carriers and airport hubs.
        - **Identify Time-Based Delay Trends:** Characterize diurnal hourly buildup, weekday variances, and seasonal monthly peaks.
        - **Classify Binary Disruption Alerts:** Train classifiers to identify flights at risk of delay ≥ 15 minutes before gate departure.
        - **Predict Departure Delay Duration:** Estimate numerical delay duration in minutes to support buffer planning.
        - **Compare Machine Learning Algorithms:** Systematically benchmark multiple architectures (Trees, Ensembles, SVMs, Boosting).
        - **Evaluate Models with Rigorous Metrics:** Measure operational recall, F1, MAE, and RMSE under strict zero target leakage.
        """
    )

st.markdown("---")

# Section D: Algorithms Used
st.subheader("Section D – Algorithms Used")
st.markdown("Detailed breakdown of machine learning algorithms implemented for both operational objectives:")

col_alg1, col_alg2 = st.columns(2)

with col_alg1:
    st.markdown("#### 🎯 Classification Algorithms (Disruption Alerting)")
    st.markdown(
        """
        1. **Decision Tree Classifier:**
           - *Mechanism:* A non-parametric tree algorithm that creates recursive decision rules by maximizing Gini impurity reduction across scheduled pre-flight features.
           - *Role:* Serves as the fast, highly interpretable baseline model.
           
        2. **Random Forest Classifier (Selected Champion):**
           - *Mechanism:* An ensemble of 100 de-correlated decision trees trained with bootstrap aggregation (bagging) and dynamic `balanced_subsample` class weighting.
           - *Role:* Delivers superior operational generalization and achieves 73.9% recall on delayed flights under severe class imbalance.
           
        3. **Support Vector Machine (LinearSVC):**
           - *Mechanism:* A maximal-margin classification algorithm that computes a linear separating hyperplane in the scaled feature space.
           - *Role:* Evaluates whether linear boundaries are sufficient for separating delayed and on-time flight operations.
        """
    )

with col_alg2:
    st.markdown("#### 📈 Regression Algorithms (Duration Estimation)")
    st.markdown(
        """
        1. **Linear Regression (Ridge L2):**
           - *Mechanism:* Fits a linear combination of normalized pre-flight features with L2 penalty to estimate continuous minutes.
           - *Role:* Provides the parametric linear benchmark against a simple historical mean predictor.
           
        2. **Random Forest Regressor:**
           - *Mechanism:* Combines predictions from multiple decision trees by averaging predictions at tree terminal leaves.
           - *Role:* Captures non-linear feature interactions without target scaling.
           
        3. **Gradient Boosting Regressor:**
           - *Mechanism:* Builds an ensemble of shallow trees sequentially, where each successive tree fits the negative gradient (residuals) of the preceding ensemble.
           - *Role:* Minimizes prediction residuals through iterative gradient descent.
           
        4. **Advanced Transformed Gradient Boosting (Selected Champion):**
           - *Mechanism:* Couples gradient boosting with advanced route interactions (`ROUTE`, `AIRLINE_ROUTE`, `DAY_PART`, `IS_WEEKEND`) and a **signed logarithmic target transformation** `sign(y)·log1p(|y|)`.
           - *Role:* Compresses severe positive right-tail skewness, reducing test MAE by 15.95% over baseline.
        """
    )

st.markdown("---")

# Section E: Validation Methodology & Zero Leakage
st.subheader("Section E – Validation Methodology & Zero Target Leakage")
st.markdown(
    """
    A foundational principle of this project is **strict methodological integrity** to ensure real-world operational validity:
    
    - **Chronological Data Splitting:** Flight operations occur sequentially in time. Random train-test splitting leaks future downstream weather and congestion patterns into past training folds. All models are trained strictly on historical flights (before 26 September 2022) and evaluated out-of-time (from 26 September 2022 onward).
    - **TimeSeries Validation:** Hyperparameter tuning with `RandomizedSearchCV` employs rolling `TimeSeriesSplit` cross-validation on the training set, guaranteeing that validation folds strictly follow training folds.
    - **Zero Target Leakage Guarantee:** All input features are restricted to scheduled pre-flight parameters available hours before departure (`AIRLINE`, `ORIGIN`, `DEST`, `CRS_DEP_TIME`, `DISTANCE`, `CRS_ELAPSED_TIME`, and cyclical temporal sine/cosine transforms). Actual departure times, arrival delays, taxi times, wheels on/off, and cancellation codes are strictly excluded.
    - **Pipeline Preprocessing:** One-hot encoders and standard scalers are fitted solely on training splits, with `handle_unknown="ignore"` to safely handle unseen airports in production inference.
    """
)

st.markdown("---")

# Section F: Evaluation Metrics
st.subheader("Section F – Evaluation Metrics")

col_met1, col_met2 = st.columns(2)
with col_met1:
    st.markdown("#### Classification Metrics")
    st.markdown(
        """
        - **Accuracy:** Overall proportion of correct predictions across all flights. *Caveat: Highly misleading in imbalanced datasets where predicting always on-time yields ~82% accuracy.*
        - **Precision:** $\\frac{TP}{TP + FP}$ — The proportion of flights flagged as delayed that actually experienced a delay. Reflects the false alarm rate.
        - **Recall (Sensitivity):** $\\frac{TP}{TP + FN}$ — The proportion of actual delayed flights successfully detected prior to gate departure. In aviation, high recall is critical for dispatch triage.
        - **F1-Score:** The harmonic mean of precision and recall: $2 \\cdot \\frac{P \\cdot R}{P + R}$. The primary criterion for champion classifier selection.
        """
    )

with col_met2:
    st.markdown("#### Regression Metrics")
    st.markdown(
        """
        - **Mean Absolute Error (MAE):** $\\frac{1}{n} \\sum |y_i - \\hat{y}_i|$ — Average absolute deviation in minutes between predicted and actual departure delay. Primary operational metric.
        - **Root Mean Squared Error (RMSE):** $\\sqrt{\\frac{1}{n} \\sum (y_i - \\hat{y}_i)^2}$ — Heavily weights extreme error outliers due to squaring.
        - **Coefficient of Determination (R²):** $1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2}$ — Measures performance relative to a naive mean baseline. In heavy-tailed distributions with black-swan groundings, R² is near-zero or negative and must not be interpreted as an accuracy percentage.
        """
    )

st.markdown("---")

# ==========================================
# SECTION G: SYSTEM ARCHITECTURE DIAGRAM
# ==========================================
st.subheader("Section G – System Architecture Diagram")
st.markdown(
    """
    End-to-end technical data flow and modeling lifecycle diagram.
    Rendered dynamically using Graphviz with color-differentiated functional subsystems.
    """
)

# Graphviz architecture diagram definition
architecture_dot = """
digraph FlightDelayIntelligence {
    graph [rankdir=TB, bgcolor="transparent", nodesep=0.4, ranksep=0.5, fontname="Helvetica"];
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11, margin="0.2,0.12", penwidth=1.5];
    edge [color="#64748b", penwidth=1.5, arrowsize=0.8];

    // Data Source (Light Blue)
    data_source [label="FLIGHT DATA CSV\\n(flight_data.csv)", fillcolor="#e0f2fe", color="#0284c7", fontcolor="#0369a1"];

    // Data Loading & Preprocessing (Light Cyan)
    subgraph cluster_preprocessing {
        label="Data Ingestion & Cleaning";
        style="dashed";
        color="#cbd5e1";
        fontcolor="#64748b";
        fontsize=10;

        data_load [label="DATA LOADING\\n(Column Validation & Type Optimization)", fillcolor="#ecfeff", color="#06b6d4", fontcolor="#0e7490"];
        data_clean [label="DATA PREPROCESSING\\n(Remove Cancellations, Retain Diversions, Preserve Extremes)", fillcolor="#ecfeff", color="#06b6d4", fontcolor="#0e7490"];
        feat_eng [label="FEATURE ENGINEERING\\n(Cyclical Time, Routes, Day Parts, Target Definitions)", fillcolor="#ecfeff", color="#06b6d4", fontcolor="#0e7490"];
    }

    // Branch 1: Statistical Analysis & EDA (Light Green)
    subgraph cluster_eda {
        label="Exploratory Analysis Phase";
        style="dashed";
        color="#cbd5e1";
        fontcolor="#64748b";
        fontsize=10;

        stat_analysis [label="STATISTICAL ANALYSIS\\n(Descriptive Stats, Skewness, Kurtosis, Outlier Audits)", fillcolor="#f0fdf4", color="#16a34a", fontcolor="#15803d"];
        eda_vis [label="EDA & VISUALIZATIONS\\n(Airlines, Airports, Corridors, Diurnal & Monthly Trends)", fillcolor="#f0fdf4", color="#16a34a", fontcolor="#15803d"];
    }

    // Branch 2: Chronological Splitting & Machine Learning
    chrono_split [label="CHRONOLOGICAL DATA SPLIT\\n(Train: < 26 Sep 2022 | Test: ≥ 26 Sep 2022)", fillcolor="#f1f5f9", color="#64748b", fontcolor="#334155"];
    train_val_test [label="TRAINING / VALIDATION / TEST FOLDS\\n(TimeSeriesSplit Cross-Validation | Zero Leakage)", fillcolor="#f1f5f9", color="#64748b", fontcolor="#334155"];

    // Objective 1: Classification (Light Orange)
    subgraph cluster_classification {
        label="Classification Pipeline (Target: DELAYED ≥ 15m)";
        style="dashed";
        color="#fdba74";
        fontcolor="#c2410c";
        fontsize=10;

        cls_branch [label="CLASSIFICATION PIPELINES\\n(OneHotEncoder + StandardScaler + Class Weights)", fillcolor="#fff7ed", color="#ea580c", fontcolor="#c2410c"];
        dt_model [label="Decision Tree\\n(Baseline)", fillcolor="#fff7ed", color="#ea580c", fontcolor="#c2410c"];
        rf_model [label="Random Forest\\n(Balanced Subsample)", fillcolor="#fff7ed", color="#ea580c", fontcolor="#c2410c"];
        svm_model [label="Linear SVM\\n(Max Margin)", fillcolor="#fff7ed", color="#ea580c", fontcolor="#c2410c"];
        cls_tuning [label="HYPERPARAMETER TUNING\\n(RandomizedSearchCV F1 Optimization)", fillcolor="#fff7ed", color="#ea580c", fontcolor="#c2410c"];
        best_cls [label="BEST CLASSIFIER\\n(Random Forest: 73.9% Recall, F1 0.42)", fillcolor="#ffedd5", color="#c2410c", fontcolor="#9a3412", penwidth=2.2];
    }

    // Objective 2: Regression (Light Purple)
    subgraph cluster_regression {
        label="Regression Pipeline (Target: DEP_DELAY Continuous Minutes)";
        style="dashed";
        color="#d8b4fe";
        fontcolor="#7e22ce";
        fontsize=10;

        reg_branch [label="REGRESSION PIPELINES\\n(OneHotEncoder + StandardScaler + Target Transform)", fillcolor="#faf5ff", color="#9333ea", fontcolor="#7e22ce"];
        lr_model [label="Linear Regression\\n(Ridge Baseline)", fillcolor="#faf5ff", color="#9333ea", fontcolor="#7e22ce"];
        rf_reg_model [label="Random Forest\\nRegressor", fillcolor="#faf5ff", color="#9333ea", fontcolor="#7e22ce"];
        gb_model [label="Gradient Boosting\\n(Transformed Target)", fillcolor="#faf5ff", color="#9333ea", fontcolor="#7e22ce"];
        reg_tuning [label="HYPERPARAMETER TUNING\\n(RandomizedSearchCV MAE Optimization)", fillcolor="#faf5ff", color="#9333ea", fontcolor="#7e22ce"];
        best_reg [label="BEST REGRESSOR\\n(Transformed GBR: MAE 17.51m, +15.95%)", fillcolor="#f3e8ff", color="#7e22ce", fontcolor="#6b21a8", penwidth=2.2];
    }

    // Evaluation Subsystem (Light Yellow)
    subgraph cluster_eval {
        label="Evaluation & Synthesis";
        style="dashed";
        color="#fde047";
        fontcolor="#a16207";
        fontsize=10;

        model_eval [label="MODEL EVALUATION\\n(Out-of-Time Test Set: Confusion Matrix, MAE vs Baseline)", fillcolor="#fefce8", color="#ca8a04", fontcolor="#854d0e"];
        comp_dash [label="COMPARISON DASHBOARD\\n(Multi-Model Benchmarks, Metric Visualizations)", fillcolor="#fefce8", color="#ca8a04", fontcolor="#854d0e"];
    }

    // Interactive Streamlit UI (Light Grey)
    streamlit_app [label="STREAMLIT MULTIPAGE DASHBOARD\\n(Light Theme, Interactive Charts, Real-Time Prediction Widgets)", fillcolor="#f8fafc", color="#475569", fontcolor="#0f172a", penwidth=2.5];

    // Pipeline Connections
    data_source -> data_load;
    data_load -> data_clean;
    data_clean -> feat_eng;

    feat_eng -> stat_analysis;
    stat_analysis -> eda_vis;

    feat_eng -> chrono_split;
    chrono_split -> train_val_test;

    train_val_test -> cls_branch;
    cls_branch -> dt_model;
    cls_branch -> rf_model;
    cls_branch -> svm_model;
    dt_model -> cls_tuning;
    rf_model -> cls_tuning;
    svm_model -> cls_tuning;
    cls_tuning -> best_cls;

    train_val_test -> reg_branch;
    reg_branch -> lr_model;
    reg_branch -> rf_reg_model;
    reg_branch -> gb_model;
    lr_model -> reg_tuning;
    rf_reg_model -> reg_tuning;
    gb_model -> reg_tuning;
    reg_tuning -> best_reg;

    best_cls -> model_eval;
    best_reg -> model_eval;

    model_eval -> comp_dash;
    eda_vis -> streamlit_app;
    comp_dash -> streamlit_app;
}
"""

st.graphviz_chart(architecture_dot)

st.markdown(
    """
    #### Architecture Stage Explanations:
    - **1. Data Ingestion & Preprocessing (Light Cyan):** Automatically locates `flight_data.csv`, validates columns, excludes cancelled flights, retains non-cancelled diverted flights, and preserves genuine extreme delay observations.
    - **2. Feature Engineering (Light Cyan):** Creates cyclical scheduled-time sine/cosine vectors, date components (`YEAR`, `MONTH`, `DAY_OF_WEEK`), interaction features (`ROUTE`, `AIRLINE_ROUTE`, `DAY_PART`), and binary targets (`DELAYED ≥ 15m`).
    - **3. Statistical Analysis & EDA (Light Green):** Conducts comprehensive descriptive statistical analysis (Mean, Median, Mode, Std, Variance, Min, Max, Q1–Q3, IQR, Skewness, Kurtosis) and macro visualizations.
    - **4. Chronological Splitting (Grey):** Partitions dataset temporally into historical training data and future untouched test data to prevent target leakage.
    - **5. Classification Subsystem (Light Orange):** Evaluates Decision Tree, Random Forest, and Linear SVM with class weighting, tuning hyperparameters via validation F1 to select the champion classifier.
    - **6. Regression Subsystem (Light Purple):** Evaluates Ridge Linear Regression, Random Forest Regressor, and Gradient Boosting with signed logarithmic target transformation, selecting the champion based on lowest validation MAE.
    - **7. Model Evaluation (Light Yellow):** Measures final test set metrics, confusion matrices, and percentage MAE improvement against baseline mean prediction.
    - **8. Streamlit Multipage UI (Light Grey):** Delivers a responsive, modern light-themed dashboard with real-time prediction widgets.
    """
)

st.markdown("---")

# Section H & I: Limitations & Future Enhancements
col_fut1, col_fut2 = st.columns(2)

with col_fut1:
    st.subheader("Section H – Project Limitations")
    st.markdown(
        """
        - **Black-Swan Outliers:** Extreme mechanical groundings, national airspace ground stops, and severe convective weather clusters cannot be predicted using scheduled timetable information alone.
        - **Classification False Alarm Trade-off:** Optimizing for high disruption recall (73.9%) requires setting conservative decision thresholds that yield false positives (1,269 in test set).
        - **Negative Regression R²:** Severe right-tail delay variance makes squared error metrics penalize scheduled-only forecasts, resulting in negative R² despite 15.95% MAE improvements.
        - **Stationarity Shifts:** Airline route networks, fleet compositions, and air traffic control procedures change over time, requiring continuous periodic re-training.
        """
    )

with col_fut2:
    st.subheader("Section I – Future Enhancements")
    st.markdown(
        """
        - **Real-Time Weather Data Integration:** Incorporate METAR/TAF weather feeds, convective storm outlooks, and wind shear warnings.
        - **Air Traffic & Airport Congestion:** Integrate live FAA National Airspace System (NAS) ground delay program status and runway acceptance rates.
        - **Aircraft Turnaround Tail-Tracking:** Track inbound aircraft tail numbers to compute physical inbound aircraft delay propagation.
        - **Deep Sequential Ensembles:** Implement LightGBM, XGBoost, CatBoost, and Temporal Convolutional Networks (TCN) for sequential schedule representations.
        - **Continuous Model Monitoring:** Implement MLOps drift monitoring (Evidently AI) to detect concept drift in seasonal delay distributions.
        """
    )
