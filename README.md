# FLIGHT DELAY INTELLIGENCE
## Statistical Analysis, Machine Learning Classification and Regression System
### Streamlit Multipage Production Analytics Platform

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.22+-3F4F75.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.2+-150458.svg)

An end-to-end, interactive aviation analytics and machine learning dashboard built with **Python**, **Streamlit**, **Scikit-Learn**, and **Plotly**. The system automatically ingests historical commercial flight records (`flight_data.csv`), performs internal preprocessing and feature engineering, conducts comprehensive statistical analysis / EDA, and benchmarks multiple classification and regression algorithms under strict **zero target leakage** methodology.

---

## 1. Clean Light-Themed User Interface

The application features a custom Streamlit light theme design system:
- **Base Background:** Crisp white (`#ffffff`).
- **Section & Card Backgrounds:** Subtle soft light grey (`#f8fafc`, `#f1f5f9`).
- **Typography:** Dark navy (`#0f172a`, `#1e293b`) using modern system fonts.
- **Accents:** Blue primary accents (`#2563eb`) with secondary cyan (`#0ea5e9`, `#06b6d4`), amber (`#f59e0b`), and emerald (`#10b981`).
- **Metric Cards:** Rounded cards (`border-radius: 12px`) with soft shadows and hover interactions.
- **Consistent Sidebar:** Aviation branding icon, live dataset metrics, and intuitive 6-menu navigation.

---

## 2. Six-Menu Multipage Dashboard Structure

The platform is organized into six dedicated menus matching exact specifications:

1. **Dashboard Overview (`1_Dashboard_Overview.py`):**
   - Executive header and problem statement.
   - 8 dynamically computed KPI cards (Total flights, delayed flights, non-delayed flights, delayed %, airlines, origin airports, destination airports, average departure delay).
   - Macro visualizations: Delayed vs. on-time donut chart, monthly delay trend, airline average delay rankings, diurnal hourly delay curve.
   - Champion model summary cards and quick navigation buttons.

2. **Statistical Analysis & EDA (`2_Statistical_Analysis.py`) — Highest Priority Menu:**
   - **Section A – Dataset Overview:** Total records, columns, date range, carrier counts, airport counts, unique routes, missing values, duplicates.
   - **Section B – Descriptive Statistics:** Count, Mean, Median, Mode, Standard Deviation, Variance, Min, Max, Q1, Q2, Q3, IQR, Skewness, Kurtosis, Tukey fence outliers.
   - **Section C – Delay Distribution:** Departure delay histogram with 15m threshold, boxplot with outlier whiskers, smooth KDE probability density curve, donut chart, and 5 operational delay categories (Early, On-time/minor, Moderate, High, Severe).
   - **Section D – Airline Performance:** Punctuality rankings, average delay bar chart, delayed percentage chart, flight volume market share, and an interactive carrier selector.
   - **Section E – Airport Analysis:** Origin and destination gateway rankings with configurable minimum flight sample volume slider and volume vs. delay scatter.
   - **Section F – Route Analysis:** Origin-destination corridor analysis (`ROUTE = ORIGIN → DEST`), top delayed corridors, most punctual corridors, and sample volume thresholds.
   - **Section G – Time-Based Dynamics:** Scheduled hour diurnal curve, weekday dynamics, monthly seasonality, multi-year progression, and day part comparison (Morning, Afternoon, Evening, Night).
   - **Section H – Correlation Analysis:** Pearson correlation matrix heatmap for scheduled parameters with causality disclaimer.
   - **Section I – Dynamic Insights:** Automated insights generated from computed metrics.

3. **Classification Analysis (`3_Classification_Analysis.py`):**
   - **Objective:** Predict binary delay disruption (`DELAYED = 1` if departure delay ≥ 15m, `0` otherwise).
   - **Models Evaluated:** Decision Tree Classifier, Random Forest Classifier, Support Vector Machine (LinearSVC).
   - **Selected Champion:** **Random Forest Classifier** (`class_weight='balanced_subsample'`, `max_depth=16`, `min_samples_split=40`, `min_samples_leaf=40`).
   - **Verified Test Metrics:** **Recall: 73.90%** (detecting 527 of 713 actual disruptions in out-of-time test set), **Precision: 29.30%**, **F1-Score: 0.4200**, **Accuracy: 55.11%**, Confusion Matrix: `[[1259, 1269], [186, 527]]`.
   - **Interactive Predictor:** Real-time disruption inference widget.

4. **Regression Analysis (`4_Regression_Analysis.py`):**
   - **Objective:** Estimate continuous departure delay duration in minutes (`DEP_DELAY`).
   - **Models Evaluated:** Linear Regression (Ridge L2), Random Forest Regressor, Gradient Boosting Regressor, and Advanced Transformed Gradient Boosting Regressor.
   - **Selected Champion:** **Advanced Transformed Gradient Boosting Regressor** (`func=sign(y)·log1p(|y|)`).
   - **Verified Test Metrics:** **MAE: 17.51 minutes** (**+15.95% error reduction** over baseline mean of 20.83 minutes), **RMSE: 47.43 minutes**, **R²: -0.0542**.
   - **Visualizations:** Actual vs. predicted scatter with 45° identity line, residual distribution histogram, and residuals vs. predicted plot.
   - **Scientific R² Discussion:** Comprehensive analysis explaining why extreme black-swan outliers (>120m) cause negative R² despite strong MAE performance.
   - **Interactive Duration Estimator:** Real-time delay duration calculation widget.

5. **Model Comparison (`5_Model_Comparison.py`):**
   - Complete classification comparison table and comparative metric bar charts (Accuracy, Precision, Recall, F1).
   - Complete regression comparison table and comparative metric bar charts (MAE, RMSE, R²).
   - Side-by-side champion summaries.
   - Dual-objective synthesis matrix explaining why classification and regression address distinct aviation dispatch requirements.

6. **About the Project (`6_About_Project.py`):**
   - Comprehensive documentation: Introduction, Problem Statement, Objectives, Algorithm details, and Chronological validation methodology.
   - **System Architecture Diagram:** Rendered dynamically using Graphviz (`st.graphviz_chart`) with color-coded subsystems (light blue, light cyan, light green, light orange, light purple, light yellow, light grey).
   - Operational limitations and future enhancement roadmap.

---

## 3. System Architecture

```
FLIGHT DATA CSV (flight_data.csv)
      │
      ▼
DATA LOADING (data_loader.py: validation & type optimization)
      │
      ▼
DATA PREPROCESSING (preprocessing.py: cancellations removed, diversions kept)
      │
      ▼
FEATURE ENGINEERING (feature_engineering.py: cyclical time, routes, day parts)
      │
      ├─────────────────────────────┬─────────────────────────────┐
      ▼                             ▼                             ▼
STATISTICAL ANALYSIS & EDA    CHRONOLOGICAL SPLIT         MODEL EVALUATION
(Descriptive stats, skew,     (Train: < 26 Sep 2022       (Confusion matrix, MAE,
 airlines, airports, routes)   Test: ≥ 26 Sep 2022)        RMSE, baseline comparison)
                                    │                             │
                      ┌─────────────┴─────────────┐               ▼
                      ▼                           ▼      COMPARISON DASHBOARD
                CLASSIFICATION               REGRESSION           │
              (DT, RF Champion, SVM)      (LR, RF, Trans GBR)     ▼
                      │                           │      STREAMLIT UI
                      └─────────────┬─────────────┘      (Light theme, interactive
                                    ▼                     charts & prediction widgets)
                              CHAMPION MODELS
```

---

## 4. Methodological Standards & Zero Leakage

1. **Strict Pre-Flight Features Only:** Input features are limited to timetable parameters known prior to departure (`AIRLINE`, `ORIGIN`, `DEST`, `DISTANCE`, `CRS_ELAPSED_TIME`, `YEAR`, `MONTH`, `DAY_OF_WEEK`, `SCHEDULED_HOUR`, `SCHEDULED_MINUTE`, `TIME_SIN`, `TIME_COS`, `ROUTE`, `AIRLINE_ROUTE`, `DAY_PART`, `IS_WEEKEND`).
2. **Exclusion of Post-Flight Information:** Actual departure time, arrival delay, taxi times, wheels on/off, and cancellation codes are strictly excluded.
3. **Chronological Splitting:**
   - Training Set: Before 26 September 2022.
   - Validation Set: TimeSeriesSplit subsets of training horizon.
   - Test Set: 26 September 2022 onward (untouched out-of-time benchmark).
4. **Outlier Preservation:** Extreme delays (up to 2,966 minutes) represent real operational black-swan events and are retained to mirror genuine aviation operations.
5. **Metric Separation:** Classification accuracy and regression MAE are never conflated.

---

## 5. Quickstart & Execution

### Prerequisites
Python 3.11+ with pip.

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
Ensure `flight_data.csv` is located in the project root or inside the `data/` folder, then execute:

```bash
streamlit run app.py
```

The application will launch at `http://localhost:8501`.

### Running Unit Tests
```bash
pytest
```
All unit tests in `tests/` validate leakage safety, cyclical feature transformations, and inference pipelines.
