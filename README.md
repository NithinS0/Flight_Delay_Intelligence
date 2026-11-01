# Flight Delay Prediction and Statistical Intelligence System
## End-to-End Aviation Data Science, Statistical Inference & Machine Learning Platform

![Python](https://img.shields.io/badge/Python-3.11-14305c.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-0f8f87.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3+-3b6aa8.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.22+-3F4F75.svg)

An enterprise-grade aviation statistical intelligence and machine learning platform built with **Python**, **Pandas**, **NumPy**, **SciPy**, **Scikit-Learn**, **XGBoost**, **LightGBM**, **Plotly**, and **Streamlit**. 

The system analyzes **3,000,000 raw commercial flight records** (**2,920,859 completed flights**, Jan 2019 – Aug 2023) from the **US Department of Transportation (USDOT) / Bureau of Transportation Statistics (BTS)**, engineers **74 zero-leakage pre-departure features** (including 58 causal prior-day rolling network statistics), executes **14 formal hypothesis tests** with Benjamini-Hochberg False Discovery Rate (FDR) control and a full **ANOVA suite**, and deploys the **Top 4 Classification Models** and **Top 4 Regression Models** inside a clean, professional 6-page Streamlit web application.

---

## Table of Contents
1. [Executive Summary & Benchmark Highlights](#1-executive-summary--benchmark-highlights)
2. [Problem Statement & Operational Motivation](#2-problem-statement--operational-motivation)
3. [Dataset Provenance & Population Governance](#3-dataset-provenance--population-governance)
4. [Zero-Leakage Causal Feature Engineering (74 Features)](#4-zero-leakage-causal-feature-engineering-74-features)
5. [Chronological Time-Series Validation Protocol](#5-chronological-time-series-validation-protocol)
6. [Advanced Statistical Intelligence & Hypothesis Testing](#6-advanced-statistical-intelligence--hypothesis-testing)
7. [Machine Learning Classification Pipeline (Top 4 Models)](#7-machine-learning-classification-pipeline-top-4-models)
8. [Machine Learning Regression Pipeline (Top 4 Models)](#8-machine-learning-regression-pipeline-top-4-models)
9. [Six-Page Streamlit Dashboard Architecture](#9-six-page-streamlit-dashboard-architecture)
10. [Complete Project Directory Structure](#10-complete-project-directory-structure)
11. [Installation, Quickstart & Testing](#11-installation-quickstart--testing)
12. [Methodological Limitations & Future Scope](#12-methodological-limitations--future-scope)

---

## 1. Executive Summary & Benchmark Highlights

 evaluated on the untouched **Out-of-Time Test Partition (`632,306` flights, Sep 2022 – Aug 2023)**:

| Task Domain | Selected Champion Model | Primary Test Metrics | Operational Significance |
| :--- | :--- | :--- | :--- |
| **Binary Classification**<br>*(Target: `DELAYED ≥ 15 min`, FAA Standard)* | **XGBoost Classifier**<br>*(Validation-tuned threshold $\tau^* = 0.2372$)* | **F1-Score: `0.4522`**<br>**ROC-AUC: `0.7068`**<br>**Recall: `59.27%`** \| **Precision: `36.56%`**<br>**Accuracy: `68.28%`** \| **PR-AUC: `0.4106`** | Successfully detects **82,801 delayed flights** prior to pushback without artificial synthetic oversampling, preserving genuine operational probabilities. |
| **Continuous Regression**<br>*(Target: `DEP_DELAY` in Minutes)* | **LightGBM Regressor**<br>*(Robust L1 / Conditional Median Objective)* | **MAE: `18.5561 min`**<br>**Median AE: `5.1581 min`**<br>**RMSE: `57.4162 min`**<br>**`-18.09%` Error vs. Mean Baseline** | Achieves an **18.09% MAE reduction** below the training mean baseline (`22.65 min`) and predicts over 50% of commercial flights within **~5.16 minutes** of actual pushback. |
| **Statistical Intelligence**<br>*(2,920,859 Completed Flights)* | **14 BH-FDR Hypothesis Tests + Cluster-Robust ANOVA** | **Mean: `+10.10 min`** \| **Median: `-2.00 min`**<br>**Skewness: `+11.49`** \| **Kurtosis: `+243.94`**<br>**Paired Diurnal $d_z = 1.0234$** \| **Evening OR $= 3.67\times$** | Proves extreme non-Gaussian leptokurtic tail behavior and quantifies that evening departures face **$3.67\times$ higher odds** of disruption than morning flights. |

---

## 2. Problem Statement & Operational Motivation

Commercial aviation networks operate on tightly coupled aircraft rotation schedules, flight crew duty-time limits, and airport slot capacities. When a flight departs 15 or more minutes late (the formal **FAA / DOT delay threshold**), disruptions cascade across downstream flight legs, passenger connections, and terminal gate assignments.

Airline operations dispatchers require two distinct pre-departure predictions at $T - 2$ hours:
1. **Binary Disruption Alerting (Classification):** *Will this flight depart 15 minutes or more late (`DELAYED = 1`)?*
2. **Continuous Delay Duration Estimation (Regression):** *How many minutes early or late will the aircraft push back (`DEP_DELAY` in minutes)?*

### Resolving Methodological Flaws in Prior Studies
* **Zero Post-Departure Leakage:** All post-pushback columns (`DEP_TIME`, `TAXI_OUT`, `WHEELS_OFF`, `WHEELS_ON`, `TAXI_IN`, `ARR_TIME`, `ARR_DELAY`, `ELAPSED_TIME`, `AIR_TIME`, and post-flight delay cause breakdowns) are strictly excluded from model inputs.
* **Causal Prior-Day Rolling History ($t < \text{FL\_DATE}$):** Historical carrier, airport, and route delay statistics are aggregated strictly over calendar dates *prior* to the flight date, preventing same-day or future lookahead leakage.
* **Natural Class Distribution Preserved:** Synthetic oversampling techniques (SMOTE, ADASYN, random oversampling/undersampling) are completely excluded so predicted probabilities reflect genuine operational base rates (`17.68%` delayed vs. `82.32%` on-time).

---

## 3. Dataset Provenance & Population Governance

* **Source:** United States Department of Transportation (USDOT) / Bureau of Transportation Statistics (BTS) *TranStats Airline On-Time Performance Reporting Database* (14 CFR Part 234).
* **Primary Dataset File:** `flight_data.csv` (`614.1 MB`, `3,000,000` rows $\times$ `32` raw columns).
* **Observation Window:** January 1, 2019 – August 31, 2023 (`56 consecutive calendar months`).
* **Network Coverage:** `18` US commercial airlines, `378` origin/destination airports, `7,785` directional flight routes, and `13,015` carrier–route pairs.

### Data Cleaning & Governance Audit
1. **Duplicate Removal:** `72` duplicate flight keys (`0.002%`) identified on `(FL_DATE, AIRLINE_CODE, FL_NUMBER, ORIGIN, DEST, CRS_DEP_TIME)` and removed.
2. **Cancelled Flights Excluded:** `79,140` cancelled flights (`2.64%`) excluded because pushback never occurred.
3. **Diverted Flights Retained:** `8,261` diverted flights (`0.28%`) retained because gate departure occurred and departure delay was physically measured.
4. **Extreme Delays Preserved:** Genuine severe delays up to **`+2,966.0 minutes` (~`49.4 hours`)** are preserved without artificial clipping.
5. **Missing Delay Reasons Policy (`NaN` Retained):** Delay cause columns (`DELAY_DUE_CARRIER`, `WEATHER`, `NAS`, `SECURITY`, `LATE_AIRCRAFT`) are structurally `NaN` on `81.9%` of rows (non-delayed flights) and are never zero-imputed or used as predictors.
6. **Cleaned Analytical Population:** **`2,920,859` completed flights** (`516,401` delayed $\ge 15$ min = `17.68%`; `2,404,458` on-time = `82.32%`).

---

## 4. Zero-Leakage Causal Feature Engineering (`74 Features`)

 Features are organized into three progressive tiers:

1. **Core Schedule Features (`BASE`, 8 features):**
   * `AIRLINE`, `ORIGIN`, `DEST`, `DISTANCE`, `CRS_ELAPSED_TIME`, `MONTH`, `DAY_OF_WEEK`, `SCHEDULED_HOUR`.
2. **Engineered Calendar & Route Features (`ENG`, 16 features):**
   * `ROUTE` (`ORIGIN_DEST`), `AIRLINE_ROUTE`, `DAY_PART` (`Morning`, `Afternoon`, `Evening`, `Night`), `IS_WEEKEND`, `SCHEDULED_MINUTE`, and cyclical encodings (`TIME_SIN`, `TIME_COS`, `MONTH_SIN`, `MONTH_COS`, `DOW_SIN`, `DOW_COS`).
3. **Causal Prior-Day Network State Features (`HIST`, 74 total features — adds 58 historical features):**
   * Computed strictly over prior calendar days ($t < \text{FL\_DATE}$) across **1-day (`1D`)**, **7-day (`7D`)**, **30-day (`30D`)**, and **expanding cumulative (`ALL`)** windows.
   * Tracks disruption rate (`RATE_*`), mean delay (`MEAN_*`), and sample volume (`N_*`) across 7 operational groupings: Global NAS network, Operating Airline, Origin Airport, Destination Airport, Route Corridor, Origin $\times$ Departure Hour, and Airline $\times$ Route.
   * Applies **Empirical Bayes Shrinkage** ($m = 50$) to regularize low-volume routes toward the global prior.

---

## 5. Chronological Time-Series Validation Protocol

To mirror real-world production deployment and prevent temporal leakage, the `2,920,859` flights are partitioned strictly by calendar date:

| Partition | Calendar Date Range | Completed Flights | Share | Role in Pipeline |
| :--- | :--- | :---: | :---: | :--- |
| **Fit / Train Split** | `2019-01-01` → `2021-12-24` | `1,784,112` | `61.08%` | Model fitting & 3-fold `TimeSeriesSplit` hyperparameter tuning |
| **Validation Split** | `2021-12-25` → `2022-09-25` | `504,441` | `17.27%` | Model selection & decision threshold optimization ($\tau^*$) |
| **Out-of-Time Test Split** | `2022-09-26` → `2023-08-31` | `632,306` | `21.65%` | Untouched final benchmark evaluation (evaluated once) |
| **Total Population** | `2019-01-01` → `2023-08-31` | **`2,920,859`** | **`100.0%`** | Complete cleaned dataset |

---

## 6. Advanced Statistical Intelligence & Hypothesis Testing

### 6.1 Descriptive & Distributional Parameters (`N = 2,920,859`)
* **Central Tendency:** Mean $\mu = +10.0957$ min (`95% CI: [10.0393, 10.1521]`), Median $Q_2 = -2.0000$ min, Mode $= -5.0000$ min, 10% Trimmed Mean $= +1.42$ min.
* **Dispersion & Robust Spread:** Standard Deviation $\sigma = 49.1777$ min, Variance $\sigma^2 = 2,418.44$ min², $\text{IQR} = 12.00$ min ($Q_1 = -6.0$, $Q_3 = +6.0$), Median Absolute Deviation ($\text{MedAD}$) $= 4.00$ min, Mean Absolute Deviation ($\text{MAD}$) $= 21.35$ min.
* **Tail & Shape:** Skewness $\gamma_1 = +11.4906$, Excess Kurtosis $\gamma_2 = +243.9439$, $P_{90} = 36$ min, $P_{95} = 72$ min, $P_{99} = 191$ min, Range $= [-90, +2,966]$ min (`64.86%` of flights depart on time or early $\le 0$ min).

### 6.2 Variance Inflation Factor (VIF) Collinearity Audit
* `CRS_ELAPSED_TIME` ($\text{VIF} = 30.028$) and `DISTANCE` ($\text{VIF} = 29.989$) exhibit severe linear collinearity ($r = 0.983$). Removing `CRS_ELAPSED_TIME` reduces `DISTANCE` VIF to **`1.029`**. All historical `H_*` features exhibit low VIF ($< 4.5$).

### 6.3 14 Formal Hypothesis Tests (Benjamini-Hochberg FDR Controlled at $\alpha = 0.05$)

| ID | Test & Research Hypothesis | Statistic | BH Adjusted $p$ | Effect Size | Decision |
| :---: | :--- | :--- | :---: | :--- | :---: |
| **HT-01** | One-Sample $t$-test ($\mu = 0$ min) | $t = 350.8533$ | $< 10^{-300}$ | Cohen's $d = 0.2053$ | **Reject $H_0$** |
| **HT-02** | One-Sample $t$-test ($\mu = 15$ min) | $t = -170.4359$ | $< 10^{-300}$ | Cohen's $d = -0.0997$ | **Reject $H_0$** |
| **HT-03** | One-Sample Wilcoxon Signed-Rank ($m = 0$) | $W = 5.482 \times 10^8$ | $8.80 \times 10^{-8}$ | Rank-Biserial $r = -0.2871$ | **Reject $H_0$** |
| **HT-04** | Two-Sample $t$-test (Evening vs. Morning) | $t = 147.9410$ | $< 10^{-300}$ | Cohen's $d = 0.2306$ | **Reject $H_0$** |
| **HT-05** | Welch's $t$-test (Weekend vs. Weekday) | $t = 13.2233$ | $6.97 \times 10^{-40}$ | Cohen's $d = 0.0176$ | **Reject $H_0$** |
| **HT-06** | Mann-Whitney $U$ (Evening vs. Morning) | $U = 3.822 \times 10^9$ | $< 10^{-300}$ | Rank-Biserial $r = 0.1943$ | **Reject $H_0$** |
| **HT-07** | Paired $t$-test ($N = 1,704$ Matched Days) | $t = 42.2460$ | $2.53 \times 10^{-267}$ | **Paired $d_z = 1.0234$** | **Reject $H_0$** |
| **HT-08** | Paired Wilcoxon ($N = 1,704$ Matched Days) | $W = 56,315.0$ | $1.86 \times 10^{-238}$ | **Rank-Biserial $r = 0.7735$** | **Reject $H_0$** |
| **HT-09** | Pearson $\chi^2$ (Airline $\times$ Delayed, $df=17$) | $\chi^2 = 32,732.28$ | $< 10^{-300}$ | Cramér's $V = 0.1059$ | **Reject $H_0$** |
| **HT-10** | Pearson $\chi^2$ (Day Part $\times$ Delayed, $df=3$) | $\chi^2 = 72,233.58$ | $< 10^{-300}$ | Cramér's $V = 0.1573$ | **Reject $H_0$** |
| **HT-11** | Fisher's Exact Test (Evening vs. Morning Delay) | $\text{OR} = 3.6749$ | $< 10^{-300}$ | **Odds Ratio $= 3.6749\times$** | **Reject $H_0$** |
| **HT-12** | Shapiro-Wilk Normality Test ($n = 5,000$) | $W = 0.3595$ | $1.35 \times 10^{-85}$ | $1 - W = 0.6405$ | **Reject $H_0$** |
| **HT-13** | D'Agostino-Pearson Omnibus $K^2$ Test | $K^2 = 4.940 \times 10^6$ | $< 10^{-300}$ | Skewness $\gamma_1 = 11.49$ | **Reject $H_0$** |
| **HT-14** | Kolmogorov-Smirnov Normality Test ($n = 50,000$) | $D = 0.3052$ | $< 10^{-300}$ | KS Distance $D = 0.3052$ | **Reject $H_0$** |

### 6.4 ANOVA & Post-Hoc Suite
* **One-Way, Welch's, Kruskal-Wallis & Daily Cluster-Robust Block-Mean ANOVA:**
  * *Scheduled Day Part ($k=4$):* $F = 7,965.87$, Welch $F = 7,833.27$, Daily Cluster-Robust $F = 416.91$ ($p < 10^{-15}$), $\eta^2 = 0.0081$.
  * *Operating Airline ($k=18$):* $F = 651.10$, Welch $F = 730.82$, Daily Cluster-Robust $F = 106.64$ ($p < 10^{-15}$), $\eta^2 = 0.0038$.
  * *Calendar Month ($k=12$):* $F = 896.39$, Welch $F = 876.45$, Daily Cluster-Robust $F = 18.27$ ($p < 10^{-15}$), $\eta^2 = 0.0034$.
* **Two-Way Factorial ANOVA (`DAY_PART × IS_WEEKEND`):** Day-Part main effect ($F = 7,966.46$), Weekend main effect ($F = 184.78$), Interaction ($F = 11.74$, $p = 2.4 \times 10^{-7}$).
* **Games-Howell Post-Hoc:** All pairwise day-part comparisons significant ($p < 0.001$), led by Evening vs. Morning ($+13.64$ min).

---

## 7. Machine Learning Classification Pipeline (Top 4 Models)

The top 4 classification algorithms were selected strictly via **Validation Delayed-Class F1-Score** with validation-optimized decision thresholds ($\tau^*$) and evaluated on the untouched **632,306-flight Test Set**:

| Rank | Classification Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Validation F1 | Threshold $\tau^*$ | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **XGBoost Classifier** | **68.28%** | **36.56%** | **59.27%** | **0.4522** | **0.7068** | **0.4106** | **0.4510** | `0.2372` | **Champion Classifier** |
| 2 | HistGradientBoosting Classifier | 67.98% | 36.36% | 59.89% | 0.4525 | 0.7055 | 0.4090 | 0.4484 | `0.2330` | Top 4 Selected |
| 3 | LightGBM Classifier | 68.79% | 36.88% | 58.06% | 0.4511 | 0.7049 | 0.4095 | 0.4479 | `0.2361` | Top 4 Selected |
| 4 | Logistic Regression | 66.62% | 35.40% | 61.92% | 0.4505 | 0.7028 | 0.4061 | 0.4472 | `0.5868` | Top 4 Selected |

* **Champion Confusion Matrix (`632,306` Test Flights):** True Negatives $= 348,925$ | False Positives $= 143,688$ | False Negatives $= 56,892$ | True Positives $= 82,801$ (**Specificity $= 70.83\%$**, **Balanced Accuracy $= 65.05\%$**, **MCC $= 0.2605$**).

---

## 8. Machine Learning Regression Pipeline (Top 4 Models)

Because `DEP_DELAY` has extreme right skewness ($\gamma_1 = 11.49$), all 4 selected regression algorithms optimize **robust L1 loss (`absolute_error`)**, estimating the conditional median delay to minimize **Mean Absolute Error (MAE)**:

| Rank | Regression Model | MAE (minutes) | RMSE (minutes) | $R^2$ Score | Median AE (minutes) | Validation MAE | Explained Variance | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **LightGBM Regressor (L1)** | **18.5561** | **57.4162** | **-0.0295** | **5.1581** | **17.8043** | **0.0245** | **Champion Regressor** |
| 2 | HistGradientBoosting Regressor (L1) | 18.5962 | 57.5214 | -0.0333 | 5.1490 | 17.8662 | 0.0223 | Top 4 Selected |
| 3 | XGBoost Regressor (L1) | 18.6308 | 57.6149 | -0.0367 | 5.1655 | 17.8945 | 0.0202 | Top 4 Selected |
| 4 | Gradient Boosting Regressor (L1) | 18.7290 | 57.6410 | -0.0376 | 5.3890 | 17.9840 | 0.0183 | Top 4 Selected |

* **Why $R^2$ is Slightly Negative Under L1 Optimization:** $R^2$ measures squared deviations around the unconditional mean ($+10.10$ min). Because L1 regression accurately tracks the conditional median ($-2.00$ min), it incurs a systematic mean-offset penalty when squared against extreme positive outliers ($+2,966$ min), while cutting **MAE by 18.09%** below the mean baseline (`22.6540 min`) and achieving a **Median Absolute Error of `5.1581 minutes`**.

---

## 9. Six-Page Streamlit Dashboard Architecture

The frontend is built entirely in **Streamlit** with a unified professional light theme (`src/ui_theme.py`), pure white content panels, navy (`#14305c`) and teal (`#0f8f87`) accents, wrapped chart titles, and zero text overlap:

1. **Dashboard Overview (`pages/1_Dashboard_Overview.py`):**
   * 6 KPI cards (`2,920,859` flights, `516,401` delayed, `17.68%` delay rate, `+10.10m` mean / `-2.0m` median delay, `18` airlines), monthly flight volume and delay trend series, delay distribution donut/histogram, airline delay bar chart, operational delay bands, and Champion Model summary cards.
2. **Statistical Analysis & EDA (`pages/2_Statistical_Analysis.py`):**
   * 8 interactive tabs: Dataset Quality Report, Descriptive Statistics (Mean/Median/Mode/MAD/MedAD/CV/P1–P99/CIs), Distribution Diagnostics (Histogram, Boxplot, KDE, Violin, ECDF, Q-Q Normality), Airline/Airport/Route Rankings, Pearson/Spearman/Kendall Correlations & VIF Collinearity, 14 BH-FDR Hypothesis Tests + Live Custom Two-Group Tester, ANOVA & Games-Howell Post-Hoc Suite, and Outlier & Temporal Trend Analysis.
3. **Classification Analysis (`pages/3_Classification_Analysis.py`):**
   * Dropdown selector restricted to the **4 Best Classification Models**, 6 KPI cards (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC), interactive Confusion Matrix, **Interactive Predicted Probability Distribution** (On-Time vs. Delayed with threshold $\tau^*$), Per-Class Classification Report, 4-Model comparison bar charts, interactive & publication ROC/PR/Threshold curves, Exact TreeSHAP & Mutual Information feature rankings, and a live **Pre-Departure Flight Disruption Simulator**.
4. **Regression Analysis (`pages/4_Regression_Analysis.py`):**
   * Dropdown selector restricted to the **4 Best Regression Models**, 5 KPI cards (MAE, RMSE, $R^2$, Median AE, Explained Variance), **Interactive Actual vs. Predicted Scatter Plot**, **Interactive Residual Error Distribution Histogram**, **Interactive Residuals vs. Predicted Plot**, **Interactive Normal Q-Q Plot**, publication error diagnostics across delay ranges, TreeSHAP feature importance, and a live **Delay Duration Estimator**.
5. **Model Comparison (`pages/5_Model_Comparison.py`):**
   * **Tab 1 (Classification Comparison):** Clean 4-model leaderboard (`Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC`), 4 comparison bar charts, and a dedicated **Best Classification Model** section with confusion matrix and ROC/PR curves.
   * **Tab 2 (Regression Comparison):** Clean 4-model leaderboard (`Model | MAE (minutes) | RMSE (minutes) | R² Score | Median Absolute Error`), 4 comparison bar charts, and a dedicated **Best Regression Model** section with actual vs. predicted and residual distribution plots.
   * **Tab 3 (Experiment Tracking Ledger):** Filterable audit table of the 8 production models.
6. **About the Project (`pages/6_About_Project.py`):**
   * Academic documentation covering problem statement, dataset governance, statistical and ML architectures, Graphviz end-to-end system architecture diagram, methodological limitations, and future scope.

---

## 10. Complete Project Directory Structure

```text
Flight_Delay_Intelligence/
├── app.py                                      # Main Streamlit entrypoint (routes to Dashboard Overview)
├── requirements.txt                            # Python package dependencies
├── dictionary.html                             # Official USDOT / BTS TranStats data dictionary
├── .streamlit/
│   └── config.toml                             # Streamlit light theme & server configuration
├── pages/                                      # 6 Multipage Streamlit Dashboard Modules
│   ├── 1_Dashboard_Overview.py
│   ├── 2_Statistical_Analysis.py
│   ├── 3_Classification_Analysis.py
│   ├── 4_Regression_Analysis.py
│   ├── 5_Model_Comparison.py
│   └── 6_About_Project.py
├── src/                                        # Core Data Science, Statistical & UI Library
│   ├── app_helper.py                           # Session state & dataset initialization
│   ├── data_loader.py                          # Raw CSV ingestion & schema validation
│   ├── preprocessing.py                        # Cleaning, deduplication & cancellation filtering
│   ├── feature_engineering.py                  # Cyclical time, route & day-part engineering
│   ├── history_features.py                     # Causal prior-day rolling history engine (t < FL_DATE)
│   ├── statistical_analysis.py                 # Descriptive, group & temporal statistical functions
│   ├── hypothesis_testing.py                   # 14 BH-FDR hypothesis tests & custom 2-group tester
│   ├── anova_analysis.py                       # One-Way, Welch, Kruskal, Cluster-Robust & 2-Way ANOVA
│   ├── eda.py                                  # EDA artifact loaders & outlier analysis
│   ├── classification.py                       # Top 4 classification pipelines & live inference
│   ├── regression.py                           # Top 4 L1 regression pipelines & live inference
│   ├── model_evaluation.py                     # Leaderboard formatting & experiment tracking
│   ├── visualizations.py                       # Plotly interactive chart builders
│   ├── ui_data.py                              # Streamlit cached data access layer
│   └── ui_theme.py                             # Light theme CSS, KPI cards & layout components
├── models/                                     # Persisted Production Model Bundles (.joblib)
│   ├── history_snapshot.joblib                 # Causal historical lookup snapshot for inference
│   ├── classification/                         # XGBoost, HistGradientBoosting, LightGBM, LogisticRegression
│   └── regression/                             # LightGBM_L1, HistGradientBoosting_L1, XGBoost_L1, GBR_L1
├── artifacts/                                  # Verified Evaluation Outputs, Metrics & 40 PPT Charts
│   ├── final_classification_metrics.csv
│   ├── final_regression_metrics.csv
│   ├── best_classification_model.json
│   ├── best_regression_model.json
│   ├── model_selection_summary.json
│   ├── classification_predictions.csv          # 10,000 out-of-time test predictions (4 classifiers)
│   ├── regression_predictions.csv              # 10,000 out-of-time test predictions (4 regressors)
│   ├── metrics/                                # JSON statistical & evaluation artifacts
│   └── ppt_charts/                             # 40 high-DPI publication PNG charts (eda_*, clf_*, reg_*)
├── project_report/                             # Academic Report & PowerPoint Reference Suite
│   ├── Flight_Delay_Prediction_Project_Report.md
│   ├── PPT_Slide_Content.md
│   ├── PPT_Visual_Reference.md
│   ├── Project_Key_Metrics.md
│   └── Presentation_Speaker_Notes.md
├── experiments/                                # Reproducible training & evaluation scripts
└── tests/                                      # Automated pytest verification suite
    └── test_intelligence_system.py
```

---

## 11. Installation, Quickstart & Testing

### Prerequisites
* **Python 3.11+** with `pip`.

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/NithinS0/Flight_Delay_Intelligence.git
cd Flight_Delay_Intelligence
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard
```bash
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to explore all 6 interactive modules.

### 3. Run Automated Verification Tests
```bash
pytest
```

---

## 12. Methodological Limitations & Future Scope

### Current Limitations
* **Unobserved Day-of-Operation Weather Shocks:** Pre-departure timetables and prior-day historical features cannot observe sudden convective thunderstorms, microbursts, or immediate FAA ground stops occurring right before pushback.
* **Spatial-Temporal Hub Clustering:** Flights departing the same hub on the same day share weather and ATC capacity shocks; naive flight-level $p$-values overstate precision, which is why our ANOVA suite pairs standard tests with **Daily Cluster-Robust Block-Mean ANOVA**.

### Future Enhancements
* **Live NOAA METAR / TAF Weather Telemetry:** Join hourly airport surface weather observations (ceiling, visibility, crosswind gusts, thunderstorm indicators) at $T - 2$ hours.
* **Tail-Number (`TAIL_NUM`) Rotation Graphs:** Construct directed acyclic graphs of daily aircraft rotations to propagate upstream inbound leg delays directly into downstream departures.
* **Calibrated Conformal Prediction Intervals:** Pair point regression estimates with calibrated 90% prediction intervals ($[\hat{q}_{0.05}, \hat{q}_{0.95}]$) for gate buffer risk management.
