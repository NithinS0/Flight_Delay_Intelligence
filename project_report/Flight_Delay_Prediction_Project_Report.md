# Flight Delay Prediction and Statistical Intelligence System
## Comprehensive Technical Project Report

**Author / Project Team:** Final Year Project Team  
**Domain:** Aviation Data Science, Statistical Intelligence & Machine Learning  
**Core Technologies:** Python 3.11, Pandas, NumPy, Scikit-Learn, SciPy, LightGBM, XGBoost, Plotly, Streamlit  
**Dataset:** United States Department of Transportation (USDOT) / Bureau of Transportation Statistics (BTS) Airline On-Time Performance Database (2019–2023)  
**Date of Report:** November 2026  

---

## Table of Contents
1. [Chapter 1 – Introduction](#chapter-1--introduction)
2. [Chapter 2 – Dataset Description & Provenance](#chapter-2--dataset-description--provenance)
3. [Chapter 3 – Data Preprocessing and Exploratory Data Analysis (EDA)](#chapter-3--data-preprocessing-and-exploratory-data-analysis-eda)
4. [Chapter 4 – Advanced Statistical Intelligence & Hypothesis Testing](#chapter-4--advanced-statistical-intelligence--hypothesis-testing)
5. [Chapter 5 – Classification Modeling & Disruption Alerting](#chapter-5--classification-modeling--disruption-alerting)
6. [Chapter 6 – Regression Modeling & Delay Duration Estimation](#chapter-6--regression-modeling--delay-duration-estimation)
7. [Chapter 7 – Model Comparison & Benchmark Ledger](#chapter-7--model-comparison--benchmark-ledger)
8. [Chapter 8 – System Architecture & Interactive Streamlit Platform](#chapter-8--system-architecture--interactive-streamlit-platform)
9. [Chapter 9 – Results, Operational Insights & Discussion](#chapter-9--results-operational-insights--discussion)
10. [Chapter 10 – Conclusion, Methodological Limitations & Future Scope](#chapter-10--conclusion-methodological-limitations--future-scope)
11. [References](#references)

---

## Chapter 1 – Introduction

### 1.1 Background of Airline Flight Delays
Commercial aviation represents one of the most critical backbones of international and domestic commerce. The Federal Aviation Administration (FAA) and the Bureau of Transportation Statistics (BTS) track millions of flights annually across the United States National Airspace System (NAS). In high-density commercial airspace, airlines operate with tightly linked schedules where flight crews, passenger connections, aircraft physical airframes, and ground handling teams must synchronize with minute-level precision. When a flight is delayed, the operational and financial consequences propagate across the entire transportation network.

### 1.2 Problem Statement
Flight departure delays are traditionally difficult to anticipate before scheduled departure because they emerge from interconnected factors: airport congestion, convective weather, ground crew constraints, and upstream rotational delay propagation. Many legacy academic implementations suffer from critical methodological flaws:
1. **Target and Post-Departure Information Leakage:** Incorporating post-pushback variables (such as actual departure time, taxi-out time, wheels-off time, or arrival delay) to "predict" departure delay.
2. **Artificial Data Manipulation:** Altering genuine flight distributions via synthetic balancing (e.g., SMOTE or ADASYN), which distorts real-world base rates and leads to inflated false-alarm rates during deployment.
3. **Improper Time-Series Splitting:** Utilizing random cross-validation on time-series flight records, training on future flights to predict historical flights.
4. **Simplistic Statistical Investigation:** Limiting exploratory analysis to superficial means and standard deviations while ignoring the profound right-skewed, leptokurtic nature of flight delay distributions.

### 1.3 Motivation
An airline dispatcher or airport operations manager requires two critical pre-departure answers at $T - 2$ hours:
* **Binary Disruption Classification:** *Will this flight experience an operational delay of 15 minutes or more ($\ge 15$ min, FAA standard threshold)?*
* **Continuous Duration Regression:** *If delayed, approximately how many minutes will pushback be postponed?*

Solving both tasks using strictly pre-departure, causal historical network features without data leakage provides genuine operational utility for gate reassignment, passenger notification, and turnaround buffer optimization.

### 1.4 Aim of the Project
To design, implement, empirically evaluate, and deploy a comprehensive **Flight Delay Prediction and Statistical Intelligence System** that combines rigorous parametric and non-parametric statistical inference with state-of-the-art machine learning classification and regression pipelines, deployed through an interactive, multipage Streamlit platform.

### 1.5 Project Objectives
* **Objective 1 (Data Governance):** Ingest and clean 3,000,000 raw US commercial flight records (2019–2023), preserving genuine black-swan operational delays up to +2,966 minutes while eliminating duplicate and cancelled flights.
* **Objective 2 (Causal Feature Engineering):** Engineer 74 zero-leakage pre-departure features, including 58 causal prior-day rolling statistics ($t < \text{FL\_DATE}$) with Empirical Bayes shrinkage.
* **Objective 3 (Statistical Rigor):** Implement 14 formal hypothesis tests with Benjamini-Hochberg False Discovery Rate (FDR) control, variance inflation factor (VIF) collinearity diagnostics, and a full ANOVA suite (One-Way, Welch's, Kruskal-Wallis, Cluster-Robust Block-Mean, and Two-Way ANOVA).
* **Objective 4 (Classification Pipeline):** Evaluate classification algorithms using chronological validation, selecting the top four architectures based on delayed-class F1-score with validation-optimized decision thresholds ($\tau^*$).
* **Objective 5 (Regression Pipeline):** Develop and evaluate regression models optimizing robust L1 loss (MAE minimization) to estimate median departure delays, selecting the top four architectures.
* **Objective 6 (Production Dashboard):** Deploy an interactive, clean light-themed multipage Streamlit dashboard enabling real-time flight delay simulations and transparent model audits.

### 1.6 Scope and Boundaries
* **Included:** US commercial domestic passenger flights from January 1, 2019 to August 31, 2023 (56 months). Features strictly restricted to schedule, calendar, route, carrier, and causal historical statistics available prior to departure.
* **Excluded:** International flights, general aviation, military flights, and real-time radar / live METAR telemetry not present in the historical BTS database.

---

## Chapter 2 – Dataset Description & Provenance

### 2.1 Data Source and Provenance
The underlying data originates from the **United States Department of Transportation (USDOT) / Bureau of Transportation Statistics (BTS)** *TranStats Airline On-Time Performance Reporting Database*. Under Federal Regulation 14 CFR Part 234, certified US air carriers that account for at least 0.5% of domestic passenger revenues are legally mandated to report detailed flight performance records.

### 2.2 Dataset Dimensions
* **Raw Dataset File:** `flight_data.csv` (Size: `614.1 MB`).
* **Raw Records:** `3,000,000` flight rows $\times$ `32` raw columns.
* **Cleaned Population:** `2,920,859` non-cancelled commercial passenger flights.
* **Date Range:** January 1, 2019 to August 31, 2023 (`56 consecutive calendar months`).

### 2.3 Attribute Descriptions and Data Types
The 32 raw columns in `flight_data.csv` encompass three distinct operational categories:

1. **Scheduled Timetable Attributes (Permissible Inputs):**
   * `FL_DATE` (Date, `object`): Flight date (`YYYY-MM-DD`).
   * `AIRLINE` / `AIRLINE_CODE` (`object`): Operating carrier name and IATA code (e.g., Delta Air Lines Inc., `DL`).
   * `DOT_CODE` (`int64`): Unique DOT carrier identification number.
   * `FL_NUMBER` (`int64`): Commercial flight flight number.
   * `ORIGIN` / `ORIGIN_CITY` (`object`): Origin airport 3-letter IATA code and city name.
   * `DEST` / `DEST_CITY` (`object`): Destination airport 3-letter IATA code and city name.
   * `CRS_DEP_TIME` (`int64`): Scheduled departure time in local standard time (`HHMM`).
   * `CRS_ARR_TIME` (`int64`): Scheduled arrival time in local standard time (`HHMM`).
   * `CRS_ELAPSED_TIME` (`float64`): Scheduled gate-to-gate block duration in minutes.
   * `DISTANCE` (`float64`): Great-circle flight distance in statute miles.

2. **Target Variables (Ground Truth):**
   * `DEP_DELAY` (`float64`): Continuous departure delay in minutes ($\text{Actual Departure} - \text{Scheduled Departure}$). Positive values denote late pushback; negative values indicate early departure.
   * `DELAYED` (Derived Binary, `int8`): FAA standard disruption indicator: $1$ if $\text{DEP\_DELAY} \ge 15.0\text{ min}$, and $0$ if $\text{DEP\_DELAY} < 15.0\text{ min}$.

3. **Post-Departure Variables (Strictly Excluded to Prevent Data Leakage):**
   * `DEP_TIME`, `TAXI_OUT`, `WHEELS_OFF`, `WHEELS_ON`, `TAXI_IN`, `ARR_TIME`, `ARR_DELAY`, `ELAPSED_TIME`, `AIR_TIME`.
   * Delay cause breakdowns: `DELAY_DUE_CARRIER`, `DELAY_DUE_WEATHER`, `DELAY_DUE_NAS`, `DELAY_DUE_SECURITY`, `DELAY_DUE_LATE_AIRCRAFT`.

### 2.4 Dataset Limitations
* **Missing Delay Explanations:** Delay-cause breakdown columns are structurally missing (`NaN`) on 81.9% of records (all non-delayed flights). They cannot be imputed with zeros because non-delayed flights do not have delay causes.
* **Unobserved Intra-Day Micro-Events:** Sudden ground stops, localized lightning, or mechanical failures occurring minutes before pushback are not captured in historical schedule tables.

---

## Chapter 3 – Data Preprocessing and Exploratory Data Analysis (EDA)

### 3.1 Data Cleaning Decisions & Audit
1. **Duplicate Detection:** A multi-column uniqueness key (`FL_DATE`, `AIRLINE_CODE`, `FL_NUMBER`, `ORIGIN`, `DEST`, `CRS_DEP_TIME`) identified `72 duplicate records` (`0.002%`), which were removed.
2. **Cancelled Flight Policy:** `79,140 cancelled flights` (`2.64%`) were excluded because pushback never occurred, rendering departure delay physically non-existent.
3. **Diverted Flight Policy:** `8,261 diverted flights` (`0.28%`) were retained because departure occurred normally, providing genuine departure delay measurements.
4. **Extreme Outlier Preservation:** Delays up to `+2,966 minutes` (~49.4 hours) were preserved. In aviation, extreme delays represent genuine operational disruptions (e.g., blizzards, mechanical groundings). Truncating them would introduce artificial data bias.

### 3.2 Feature Engineering Architecture
To capture temporal dynamics and network conditions without lookahead leakage, 74 features were engineered:
* **Cyclical Temporal Encodings:** $\sin$ and $\cos$ transformations of departure minute ($1,440$ min/day), month ($12$ months/year), and day of week ($7$ days/week).
* **Network Graph Attributes:** Route corridors (`ORIGIN_DEST`), carrier-corridors (`AIRLINE_ORIGIN_DEST`), and four day-parts (`Morning [05:00–11:59]`, `Afternoon [12:00–16:59]`, `Evening [17:00–20:59]`, `Night [21:00–04:59]`).
* **58 Causal Prior-Day Historical Statistics (`H_*`):** Rolling 1-day, 7-day, 30-day, and cumulative metrics calculated strictly on flights departing *prior to* the target flight date ($t < \text{FL\_DATE}$). Features incorporate Empirical Bayes smoothing ($m = 50$) to stabilize estimates for low-frequency routes.

### 3.3 Exploratory Data Analysis & Key Findings
* **Prevalence of Early Departures:** `64.86%` of flights depart on schedule or early ($\le 0$ min), with the modal departure occurring at `-5.0 minutes`.
* **Diurnal Delay Accumulation:** Delay probability rises monotonically throughout the operating day: Morning flights experience only `10.82%` delay rate (mean `+4.18 min`), while Evening flights peak at `26.24%` delay rate (mean `+17.82 min`).
* **Carrier Heterogeneity:** Airline average delays range from `+4.11 minutes` (Hawaiian Airlines) to `+16.48 minutes` (JetBlue Airways).

---

## Chapter 4 – Advanced Statistical Intelligence & Hypothesis Testing

### 4.1 Descriptive Statistics Suite
Comprehensive analysis across the `2,920,859` completed flights revealed the following distributional characteristics:
* **Central Tendency:** Mean $\mu = +10.0957$ min, Median $M = -2.0000$ min, Mode $= -5.0000$ min.
* **Dispersion:** Standard Deviation $\sigma = 49.1777$ min, Variance $\sigma^2 = 2,418.44$ min², Interquartile Range $\text{IQR} = 12.00$ min ($Q_1 = -6.0$, $Q_3 = +6.0$).
* **Robust Measures:** Median Absolute Deviation ($\text{MedAD}$) $= 4.00$ min, Mean Absolute Deviation ($\text{MAD}$) $= 21.35$ min.
* **Shape Parameters:** Skewness $\gamma_1 = +11.4906$ (extreme right asymmetry), Excess Kurtosis $\gamma_2 = +243.9439$ (heavy leptokurtic tail).

### 4.2 Collinearity & Variance Inflation Factor (VIF) Analysis
* `DISTANCE` ($\text{VIF} = 29.989$) and `CRS_ELAPSED_TIME` ($\text{VIF} = 30.028$) exhibit severe multicollinearity due to near-perfect physical correlation ($r = 0.983$). Dropping `CRS_ELAPSED_TIME` collapses distance VIF to $1.029$.
* `SCHEDULED_HOUR` ($\text{VIF} = 9.411$) and cyclical `TIME_SIN` ($\text{VIF} = 7.437$) demonstrate moderate shared diurnal signal.
* All engineered historical features (`H_*`) exhibit low collinearity ($\text{VIF} < 4.5$).

### 4.3 14 Formal Hypothesis Tests (Benjamini-Hochberg FDR Controlled)
Every test below was evaluated under two-tailed conditions with $\alpha = 0.05$ and adjusted via Benjamini-Hochberg procedure:

1. **HT-01 (One-Sample $t$-test vs 0 min):** $t = 350.85$, $p < 10^{-300}$. Reject $H_0$. True mean departure delay is significantly positive ($\mu = +10.10$ min, $d = 0.205$).
2. **HT-02 (One-Sample $t$-test vs 15 min):** $t = -170.44$, $p < 10^{-300}$. Reject $H_0$. Overall flight population mean delay is significantly below the FAA disruption threshold.
3. **HT-03 (One-Sample Wilcoxon Signed-Rank vs 0 min):** $W = 5.48 \times 10^8$, $p = 8.80 \times 10^{-8}$. Reject $H_0$. Median delay is significantly below zero ($m = -2.0$ min, Rank-Biserial $r = -0.287$).
4. **HT-04 (Two-Sample $t$-test: Evening vs Morning):** $t = 147.94$, $p < 10^{-300}$. Reject $H_0$. Evening flights experience significantly higher mean delay ($17.82$m vs $4.18$m, $d = 0.231$).
5. **HT-05 (Welch's $t$-test: Weekend vs Weekday):** Welch $t = 13.22$, $p = 6.97 \times 10^{-40}$. Reject $H_0$. Weekday mean delay ($10.23$m) is significantly higher than weekend mean delay ($9.37$m).
6. **HT-06 (Mann-Whitney $U$: Evening vs Morning):** $U = 3.82 \times 10^9$, $p < 10^{-300}$. Reject $H_0$. Complete non-parametric distributional shift toward delays in evening operations ($r = 0.194$).
7. **HT-07 (Paired $t$-test: $N=1,704$ Matched Calendar Days):** Paired $t = 42.25$, $p = 2.53 \times 10^{-267}$. Reject $H_0$. Controlling for daily network shocks, evening delay exceeds morning delay by an average of $+13.62$ min ($d_z = 1.023$, large effect).
8. **HT-08 (Paired Wilcoxon: $N=1,704$ Matched Days):** $W = 56,315.0$, $p = 1.86 \times 10^{-238}$. Reject $H_0$. Evening median delay exceeds morning median on over 94% of calendar days ($r = 0.774$).
9. **HT-09 (Pearson $\chi^2$: Airline vs Delay, $df=17$):** $\chi^2 = 32,732.28$, $p < 10^{-300}$. Reject $H_0$. Delay occurrence is strongly dependent on carrier (Cramér's $V = 0.106$).
10. **HT-10 (Pearson $\chi^2$: Day Part vs Delay, $df=3$):** $\chi^2 = 72,233.58$, $p < 10^{-300}$. Reject $H_0$. Disruption rate strongly depends on scheduled time of day (Cramér's $V = 0.157$).
11. **HT-11 (Fisher's Exact Test: Evening vs Morning Disruption):** Odds Ratio $= 3.6749$, $p < 10^{-300}$. Reject $H_0$. Flights scheduled in the evening have $3.67\times$ higher odds of being delayed $\ge 15$ minutes compared to morning flights.
12. **HT-12 (Shapiro-Wilk Normality Test, $n=5,000$):** $W = 0.3595$, $p = 1.35 \times 10^{-85}$. Reject $H_0$. Severe non-normality ($1 - W = 0.6405$).
13. **HT-13 (D'Agostino-Pearson $K^2$ Test):** $K^2 = 4,940,303.01$, $p < 10^{-300}$. Reject $H_0$. Significant skewness and excess kurtosis confirm non-Gaussian behavior.
14. **HT-14 (Kolmogorov-Smirnov Test, $n=50,000$):** $D = 0.3052$, $p < 10^{-300}$. Reject $H_0$. Empirical cumulative distribution differs significantly from normal distribution.

### 4.4 ANOVA Suite
* **One-Way ANOVA (Day Part, $k=4$):** $F = 7,965.87$, Welch's $F = 7,833.27$, Cluster-Robust $F = 416.91$ ($p < 10^{-15}$), $\eta^2 = 0.0081$.
* **One-Way ANOVA (Carrier, $k=18$):** $F = 651.10$, Welch's $F = 730.82$, Cluster-Robust $F = 106.64$ ($p < 10^{-15}$), $\eta^2 = 0.0038$.
* **Two-Way ANOVA (`DAY_PART × IS_WEEKEND`):** Day-part main effect dominates ($F = 7,966.46$, Partial $\eta^2 = 0.00812$), while weekend effect ($F = 184.78$) and interaction effect ($F = 11.74$) are secondary.

---

## Chapter 5 – Classification

### 5.1 Classification Objective & Target Definition
The classification pipeline alerts operational dispatchers to impending flight disruptions prior to pushback. Target variable:
$$\text{DELAYED} = \begin{cases} 1 & \text{if } \text{DEP\_DELAY} \ge 15.0\text{ minutes} \\ 0 & \text{if } \text{DEP\_DELAY} < 15.0\text{ minutes} \end{cases}$$

### 5.2 Chronological Validation Methodology
To guarantee true generalization, models were tuned using 3-fold `TimeSeriesSplit` across the fit partition (`2019-01-01` to `2021-12-24`, 1,784,112 flights) and evaluated for selection on the validation partition (`2021-12-25` to `2022-09-25`, 504,441 flights). The primary model selection metric was **Delayed-Class F1-Score**, supported by Recall and PR-AUC. The single final evaluation was performed on the untouched out-of-time test partition (`2022-09-26` to `2023-08-31`, 632,306 flights).

### 5.3 Best Four Classification Algorithms Selected
1. **XGBoost Classifier (Champion Classifier):** Selected based on highest validation F1 (`0.4510`), validation Recall (`59.16%`), and validation PR-AUC (`0.4213`) with optimal decision threshold $\tau^* = 0.2372$.
2. **HistGradientBoosting Classifier:** Scikit-learn histogram-binned tree boosting (Validation F1 = `0.4484`, $\tau^* = 0.2330$).
3. **LightGBM Classifier:** Fast gradient boosting using leaf-wise tree growth (Validation F1 = `0.4479`, $\tau^* = 0.2361$).
4. **Logistic Regression:** L2-regularized linear baseline with OneHotEncoding and StandardScaling (Validation F1 = `0.4472`, $\tau^* = 0.5868$).

### 5.4 Final Test Set Performance (632,306 Untouched Flights)

| Classification Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Balanced Accuracy | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost Classifier** | **68.28%** | **36.56%** | **59.27%** | **0.4522** | **0.7068** | **0.4106** | **65.05%** | **Champion Classifier** |
| **HistGradientBoosting Classifier** | 67.98% | 36.36% | 59.89% | 0.4525 | 0.7055 | 0.4090 | 65.08% | Top 4 Candidate |
| **LightGBM Classifier** | 68.79% | 36.88% | 58.06% | 0.4511 | 0.7049 | 0.4095 | 64.94% | Top 4 Candidate |
| **Logistic Regression** | 66.62% | 35.40% | 61.92% | 0.4505 | 0.7028 | 0.4061 | 64.94% | Top 4 Candidate |

### 5.5 Champion Confusion Matrix
On the 632,306 out-of-time test flights:
* **True Negatives (TN):** `348,925` (correct on-time predictions)
* **False Positives (FP):** `143,688` (false disruption alarms)
* **False Negatives (FN):** `56,892` (missed delayed flights)
* **True Positives (TP):** `82,801` (successfully detected disruptions)
* **Specificity (TNR):** `70.83%`
* **Matthews Correlation Coefficient (MCC):** `0.2605`

---

## Chapter 6 – Regression

### 6.1 Regression Objective & Target Variable
Predict continuous departure delay in minutes (`DEP_DELAY`, ranging from `-90 min` to `+2,966 min`). Regression models are evaluated on numerical deviation metrics and **do not report classification accuracy as a percentage**.

### 6.2 The L1 Loss (MAE Minimization) Formulation
Under extreme right skewness ($\gamma_1 = 11.49$), traditional L2 loss (Mean Squared Error) attempts to model the conditional mean $E[Y \mid X] \approx +10.10$ min. However, because 64.86% of flights depart early or on time, predicting near the mean incurs massive error on normal flights. 
By formulating regression models with **L1 loss** ($\text{objective} = \text{absolute\_error}$), models estimate the **conditional median** $\text{Median}(Y \mid X) \approx -2.0$ min, directly minimizing Mean Absolute Error (MAE).

### 6.3 Best Four Regression Algorithms Selected
1. **LightGBM Regressor (L1 Loss, Champion Regressor):** Selected on lowest validation MAE (`17.8043 min`).
2. **HistGradientBoosting Regressor (L1 Loss):** Validation MAE = `17.8662 min`.
3. **XGBoost Regressor (L1 Loss):** Validation MAE = `17.8945 min`.
4. **Gradient Boosting Regressor (L1 Loss):** Validation MAE = `17.9840 min`.

### 6.4 Final Test Set Performance (632,306 Untouched Flights)

| Regression Model | MAE (min) | RMSE (min) | $R^2$ Score | Median AE (min) | Explained Variance | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM Regressor** | **18.5561** | **57.4162** | **-0.0295** | **5.1581** | **0.0245** | **Champion Regressor** |
| **HistGradientBoosting Regressor** | 18.5962 | 57.5214 | -0.0333 | 5.1490 | 0.0223 | Top 4 Candidate |
| **XGBoost Regressor** | 18.6308 | 57.6149 | -0.0367 | 5.1655 | 0.0202 | Top 4 Candidate |
| **Gradient Boosting Regressor** | 18.7290 | 57.6410 | -0.0376 | 5.3890 | 0.0183 | Top 4 Candidate |

### 6.5 Prediction-vs-Actual and Residual Interpretation
* **Baseline Comparison:** The training mean baseline yields test MAE of `22.6540 min`. The champion LightGBM regressor achieves an **`18.09%` error reduction**.
* **Median Absolute Error of `5.1581 min`:** For more than 316,000 test flights, the predicted delay is within ~5 minutes of the actual pushback timestamp.
* **Explanation of Negative $R^2$:** $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ penalizes squared deviations from the mean. Because L1 regression correctly predicts near the median ($-2.0$ min), it incurs a negative squared bias offset against the $+10.10$ min test mean, which is mathematically expected when optimizing L1 loss under extreme positive skewness.

---

## Chapter 7 – Model Comparison

### 7.1 Classification Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier (Champion)** | **68.28%** | **36.56%** | **59.27%** | **0.4522** | **0.7068** | **0.4106** |
| HistGradientBoosting Classifier | 67.98% | 36.36% | 59.89% | 0.4525 | 0.7055 | 0.4090 |
| LightGBM Classifier | 68.79% | 36.88% | 58.06% | 0.4511 | 0.7049 | 0.4095 |
| Logistic Regression | 66.62% | 35.40% | 61.92% | 0.4505 | 0.7028 | 0.4061 |

* **Selection Rationale:** XGBoost demonstrated superior ranking discrimination (ROC-AUC `0.7068`, PR-AUC `0.4106`) and the highest validation F1-score (`0.4510`), maintaining balanced recall (`59.27%`) and precision (`36.56%`).

### 7.2 Regression Comparison

| Model | MAE (minutes) | RMSE (minutes) | $R^2$ Score | Median Absolute Error |
| :--- | :---: | :---: | :---: | :---: |
| **LightGBM Regressor (Champion)** | **18.5561** | **57.4162** | **-0.0295** | **5.1581** |
| HistGradientBoosting Regressor | 18.5962 | 57.5214 | -0.0333 | 5.1490 |
| XGBoost Regressor | 18.6308 | 57.6149 | -0.0367 | 5.1655 |
| Gradient Boosting Regressor | 18.7290 | 57.6410 | -0.0376 | 5.3890 |

* **Selection Rationale:** LightGBM achieved the lowest out-of-time MAE (`18.5561 min`), lowest RMSE (`57.4162 min`), and highest explained variance (`0.0245`) with the fastest training convergence (7.58 seconds).

---

## Chapter 8 – System Architecture and Dashboard

### 8.1 System Architecture Overview
The end-to-end architecture is organized into five modular functional layers:
1. **Data Ingestion & Hygiene Layer:** Ingests raw `flight_data.csv`, audits duplicates, cleans non-cancelled flights, and preserves genuine extreme delay outliers.
2. **Causal History Feature Engine (`history_features.py`):** Computes rolling 1D, 7D, 30D, and cumulative network statistics prior to departure date ($t < \text{FL\_DATE}$), serializing snapshot matrices to `models/history_snapshot.joblib`.
3. **Statistical Inference Engine (`eda.py`, `hypothesis_testing.py`):** Calculates descriptive metrics, runs 14 FDR-controlled hypothesis tests, and executes the ANOVA suite.
4. **Machine Learning Pipeline Layer (`classification.py`, `regression.py`):** Implements time-series splitting, hyperparameter optimization, threshold calibration, and joblib model persistence.
5. **Streamlit Multi-Page Presentation Layer (`app.py`, `pages/`):** A responsive, unified light-themed web interface for visualization, inspection, and live simulation.

### 8.2 Description of the Six Dashboard Pages
* **Page 1: Dashboard Overview (`1_Dashboard_Overview.py`):** Executive KPI cards (2.92M flights, 17.68% delay rate, 18 airlines), monthly flight volumes, delay trends, airline delay rankings, and selected champion model cards.
* **Page 2: Statistical Analysis & EDA (`2_Statistical_Analysis.py`):** 8 interactive tabs covering Dataset Quality, Descriptive Statistics, Distribution Diagnostics (KDE/Violin/ECDF/Q-Q), Airline & Airport Performance, Correlation & VIF, 14 Hypothesis Tests, ANOVA & Games-Howell Post-Hoc, and Outlier Audits.
* **Page 3: Classification Analysis (`3_Classification_Analysis.py`):** Model inspector for the top 4 classifiers with 6 KPI metrics, interactive Confusion Matrix, **Predicted Probability Distribution** (separating On-Time vs. Delayed flights with threshold $\tau^*$), ROC and PR curves, TreeSHAP feature importance, and an interactive **Pre-Departure Flight Disruption Simulator**.
* **Page 4: Regression Analysis (`4_Regression_Analysis.py`):** Model inspector for the top 4 regressors with 5 KPI metrics, **Interactive Actual vs. Predicted Scatter Plot** (with 45° reference line), **Interactive Residual Distribution Histogram**, **Residuals vs. Predicted Plot**, **Normal Q-Q Plot**, publication error diagnostics, TreeSHAP gain rankings, and an interactive **Delay Duration Estimator**.
* **Page 5: Model Comparison (`5_Model_Comparison.py`):** Clean benchmark leaderboards and 4 comparative bar charts for Classification (F1, Recall, Accuracy, ROC-AUC) and Regression (MAE, RMSE, $R^2$, Median AE), dedicated Best Model sections, and the 8-model Production Experiment Tracking Ledger.
* **Page 6: About the Project (`6_About_Project.py`):** Formal academic documentation, causal engineering principles, Graphviz system architecture diagram, methodological limitations, and future research roadmap.

---

## Chapter 9 – Results, Operational Insights & Discussion

### 9.1 Important Statistical Findings
* Commercial flight departure delays are heavily non-Gaussian, characterized by positive skewness ($\gamma_1 = 11.49$) and massive leptokurtosis ($\gamma_2 = 243.94$). Median departure delay is negative ($-2.0$ min), with $64.86\%$ of flights departing early.
* Time of day is the single strongest operational predictor of delay risk ($F = 7,965.87$, $\text{OR} = 3.67$ for evening vs. morning), driven by cumulative rotational delay propagation throughout the air traffic network.

### 9.2 Practical Significance for Airline Operations
* **Proactive Gate and Crew Buffering:** Detecting $59.27\%$ of delayed flights prior to pushback enables ground dispatchers to hold connecting gates or adjust crew duty rotations before passengers board.
* **Customer Communication:** Utilizing the median absolute error of $5.16$ minutes allows carriers to provide realistic delay estimates rather than arbitrary rolling 15-minute pushbacks.

---

## Chapter 10 – Conclusion, Methodological Limitations & Future Scope

### 10.1 Overall Project Achievements
The project successfully developed a leakage-free, statistically verified flight delay prediction system across 2,920,859 US commercial flights. By restricting active algorithms to the best four classification and best four regression models and eliminating synthetic class-balancing distortions, the system maintains strict operational realism and achieves high discriminative power ($0.7068$ ROC-AUC) and tight median precision ($5.16$ min Median AE).

### 10.2 Methodological Limitations
1. **Unobserved Micro-Weather Events:** Pre-departure timetable data lacks real-time localized thunderstorm telemetry or sudden convective weather reroutes occurring immediately prior to departure.
2. **Hub Clustering Effects:** Flights departing the same airport on the same afternoon share airspace constraints, requiring daily block-mean cluster-robust adjustments.

### 10.3 Future Improvements
* **Real-Time Weather Integration:** Ingest live NOAA METAR and TAF surface observations at $T - 2$ hours.
* **Aircraft Rotation Tracking:** Parse physical aircraft tail numbers (`TAIL_NUM`) to track inbound inbound leg delays directly into downstream departures.
* **Conformal Prediction Intervals:** Provide calibrated 90% confidence intervals ($[\hat{q}_{0.05}, \hat{q}_{0.95}]$) alongside point duration predictions.

---

## References

1. U.S. Department of Transportation, Bureau of Transportation Statistics, "Airline On-Time Performance Data," *TranStats Database*, 2024. [Online]. Available: https://www.transtats.bts.gov/
2. Federal Aviation Administration, "Aviation Capacity Outlook and Delay Definitions," U.S. Department of Transportation, Washington, D.C., Order 7031.2C, 2020.
3. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining (KDD)*, 2016, pp. 785–794.
4. G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 3146–3154.
5. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
6. Y. Benjamini and Y. Hochberg, "Controlling the false discovery rate: A practical and powerful approach to multiple testing," *Journal of the Royal Statistical Society: Series B (Methodological)*, vol. 57, no. 1, pp. 289–300, 1995.
7. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 4765–4774.
8. P. A. Games and J. F. Howell, "Pairwise multiple comparison procedures with unequal N's and/or variances: A Monte Carlo study," *Journal of Educational Statistics*, vol. 1, no. 2, pp. 113–125, 1976.

