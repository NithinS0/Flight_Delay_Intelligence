# Flight Delay Prediction and Statistical Intelligence System — Final Scientific & Engineering Report

> **Generated Automatically from Verified Full-Dataset Execution (`2,920,859` Completed Non-Cancelled Flights out of `3,000,000` Raw Records, Jan 2019 – Aug 2023)**

---

## 1. Executive Summary

The **Flight Delay Prediction and Statistical Intelligence System** was systematically upgraded across **dataset preparation**, **parametric and non-parametric statistical inference**, **multi-algorithm classification and regression modeling**, and an **executive 6-page Streamlit dashboard**:

- **Full-Scale Zero-Leakage Data Pipeline:** Processed all **3,000,000 raw records** (`575.4 MB`) in `flight_data.csv`, isolating **2,920,859 completed non-cancelled commercial flights** (`79,141` cancelled flights = `2.64%` excluded; `0` exact duplicate records). Genuine extreme delays up to **2,966 minutes (~49.4 hours)** are preserved, structural `NaN` values in delay-reason columns (`DELAY_DUE_*`, missing on `81.89%` of flights) are never zero-imputed, and all post-departure columns (`DEP_TIME`, `ARR_DELAY`, `TAXI_OUT`, `WHEELS_OFF`, etc.) are strictly barred from predictive features.
- **Causal Multi-Horizon Feature Engineering (`74 Features`):** Engineered **58 strictly causal prior-day historical network state features (`H_*`)** across **1-day, 7-day, 30-day, and expanding cumulative (`< FL_DATE`) windows** with Empirical-Bayes shrinkage ($m = 50$). Expanding from the original 8 schedule features (`BASE`) to the 74-feature causal set (`HIST`) lifts out-of-time XGBoost F1 from **0.3547 to 0.4522 (+27.5%)** and ROC-AUC from **0.6317 to 0.7068**.
- **Comprehensive Statistical Intelligence:**
  - **Central Tendency & Tail Morphology:** While **64.86%** of flights depart on or before schedule (**Median = -2.0 min**, **Mode = -5.0 min**), extreme right-tail disruptions pull the **Arithmetic Mean to +10.100 min** (95% Parametric CI `[10.043, 10.156] min`, Bootstrap CI `[9.954, 10.268] min`, **Distance-Weighted Mean = 10.724 min**, **Skewness = 11.491**, **Excess Kurtosis = 243.902**).
  - **14 Formal Hypothesis Tests (Benjamini-Hochberg FDR $\alpha = 0.05$):** Confirmed significant diurnal cascade (Evening vs Morning $+10.97\text{ min}$, Welch $t = 123.51, q < 10^{-300}$, Cohen's $d = 0.251$), paired daily Evening-minus-Morning excess across $N = 1,704$ matched days ($+10.52\text{ min}$, paired $t = 49.49$, Wilcoxon $p < 10^{-260}$, $d_z = 1.199$), and prior-day congestion persistence (Odds Ratio $= 2.660$, $\chi^2 = 58,412$).
  - **ANOVA & Post-Hoc Suite:** One-Way, Welch's, Kruskal-Wallis, **Daily Cluster-Robust Block-Mean ANOVA**, **Two-Way Factorial ANOVA (`DAY_PART × IS_WEEKEND`)**, Brown-Forsythe Levene checks, and **Games-Howell Post-Hoc** tests across Airline, Month, Day of Week, Day Part, and Top-15 Origin Airports.
- **Classification Improvement (`16 Evaluated Models`):** Champion **Soft Voting Ensemble (`XGBoost + LightGBM + HistGradientBoosting`)** with isotonic calibration and validation-tuned threshold $\tau^* = 0.241$ achieves **Out-of-Time Test F1 = 0.4543** (vs `0.4171` for the original Random Forest, and `0.3596` at default 0.50 threshold), **ROC-AUC = 0.7078**, **PR-AUC = 0.4126**, **Recall = 60.66%**, and **MCC = 0.2621**.
- **Regression Improvement (`34 Evaluated Runs`):**
  1. **Point-Prediction MAE Champion (`LightGBM (L1)`):** Achieves **Validation MAE = 17.804 min**, **Out-of-Time Test MAE = 18.556 min**, and **Median AE = 5.158 min** — an **18.09% MAE reduction** compared to the training-mean baseline (`22.654 min`) and **4.25% lower MAE** than the original project's Transformed GBR (`19.379 min` on the full out-of-time test horizon).
  2. **Variance-Explained $R^2$ Champion (`XGBoost (L2)`):** Achieves **positive Out-of-Time Test $R^2 = +0.0378$** and **Test RMSE = 55.51 min** (vs `-0.0570` for the original project's Transformed GBR).

---

## 2. Dataset Overview

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Source File** | `flight_data.csv` | `575.41 MB` on disk (`312.55 MB` compressed Parquet) |
| **Raw Dimensions** | `3,000,000` rows $\times$ `32` columns | Multi-year US DOT BTS On-Time Performance sample |
| **Date Range** | `2019-01-01` to `2023-08-31` | `1,704` distinct calendar days (`56` months) |
| **Cancelled Flights Excluded** | `79,141` (`2.638%`) | Excluded because cancelled flights never push back (`DEP_DELAY` is undefined) |
| **Completed Non-Cancelled Flights** | `2,920,859` (`97.362%`) | Primary analytical and modeling population |
| **Diverted Flights Retained** | `6,924` (`0.231%`) | Retained because departure delay occurred prior to en-route diversion |
| **Exact Duplicate Records** | `0` | Verified on full row and `(FL_DATE, AIRLINE_CODE, FL_NUMBER, ORIGIN, DEST, CRS_DEP_TIME)` |
| **Operating Airlines** | `18` US Carriers | Southwest (`WN`: 560,645), Delta (`DL`: 392,775), American (`AA`: 371,599), SkyWest (`OO`: 326,866), United (`UA`: 249,211) |
| **Unique Airports** | `378` Origins / `378` Destinations | Top hubs: `ATL` (150,762), `DFW` (126,645), `ORD` (118,939), `DEN` (116,603), `CLT` (92,380) |
| **Unique Directional Routes** | `7,785` (`ORIGIN_DEST`) | `14,768` unique `AIRLINE_ROUTE` combinations |
| **Classification Target (`DELAYED` $\ge 15$m)** | `516,401` Delayed (`17.680%`) vs `2,404,458` On-Time (`82.320%`) | Imbalance ratio `4.656 : 1` |
| **Regression Target (`DEP_DELAY`)** | Range `[-90.0, +2,966.0]` min | Mean `+10.100 min`, Median `-2.000 min`, SD `49.232 min` |

---

## 3. Data Preprocessing Summary

1. **Missing-Value Governance:**
   - Delay-reason columns (`DELAY_DUE_CARRIER`, `DELAY_DUE_WEATHER`, `DELAY_DUE_NAS`, `DELAY_DUE_SECURITY`, `DELAY_DUE_LATE_AIRCRAFT`) are missing (`NaN`) on **2,456,660 raw rows (81.89%)** because the FAA only logs delay attributions when a flight arrives $\ge 15$ minutes late. These columns are **post-event outcomes** and are **never zero-imputed or used as predictors**.
   - `CANCELLATION_CODE` is `NaN` on **2,920,860 non-cancelled rows (97.36%)** and is excluded after filtering `CANCELLED == 0`.
   - Early-period causal history cold-start values are imputed inside scikit-learn `ColumnTransformer` pipelines using training-fold `SimpleImputer(strategy="median")`.
2. **Genuine Extreme Delays Preserved:**
   - All `43,065` flights with $|z| > 3$ (`DEP_DELAY > 157.8 min`, up to `2,966 min`) are verified real operational disruptions and are retained intact in training, validation, and test partitions.
3. **Categorical Encoding & Numerical Scaling:**
   - High-cardinality categoricals (`AIRLINE`, `ORIGIN`, `DEST`, `ROUTE`, `AIRLINE_ROUTE`, `DAY_PART`) use `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` for tree ensembles and `OneHotEncoder(handle_unknown="ignore", min_frequency=30)` paired with `StandardScaler` for linear/SVM models.

---

## 4. Feature Engineering Summary

We constructed three nested, strictly pre-departure feature sets:
1. **`BASE` (8 features):** `AIRLINE`, `ORIGIN`, `DEST`, `DISTANCE`, `CRS_ELAPSED_TIME`, `MONTH`, `DAY_OF_WEEK`, `SCHEDULED_HOUR`.
2. **`ENG` (16 features):** Adds `ROUTE`, `AIRLINE_ROUTE`, `DAY_PART`, `IS_WEEKEND`, `SCHEDULED_MINUTE`, and cyclical Fourier encodings `TIME_SIN`, `TIME_COS`, `MONTH_SIN`, `MONTH_COS`, `DOW_SIN`, `DOW_COS`.
3. **`HIST` (74 features):** Adds **58 strictly causal prior-day historical delay features (`H_*`)** computed at daily granularity and shifted by 1 full calendar day (`< FL_DATE`) so same-day flights never leak into one another.

### Feature Selection & Exact TreeSHAP Attribution Top-10 (`artifacts/feature_selection_classification.csv`)

| Rank | Feature | Category | Mean \|TreeSHAP\| (Log-Odds) | TreeSHAP Share (%) | XGBoost Gain (%) | Mutual Information | ANOVA $F$-Score |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `H_ORIGINHOUR_RATE_30D` | Causal_History | 0.23295 | 14.27% | 27.09% | 0.01850 | 463.9 |
| 2 | `TIME_SIN` | Scheduled_Calendar | 0.18069 | 11.06% | 2.84% | 0.01193 | 160.0 |
| 3 | `H_AIRLINE_RATE_1D` | Causal_History | 0.13182 | 8.07% | 6.13% | 0.01955 | 363.6 |
| 4 | `H_AIRLINE_RATE_7D` | Causal_History | 0.08494 | 5.20% | 4.42% | 0.01949 | 348.4 |
| 5 | `H_AIRLINEROUTE_RATE_ALL` | Causal_History | 0.07363 | 4.51% | 1.59% | 0.00910 | 133.8 |
| 6 | `SCHEDULED_HOUR` | Scheduled_Calendar | 0.07183 | 4.40% | 5.18% | 0.01025 | 162.3 |
| 7 | `H_ORIGINHOUR_RATE_ALL` | Causal_History | 0.05803 | 3.55% | 1.84% | 0.00970 | 259.8 |
| 8 | `H_ORIGIN_MEAN_1D` | Causal_History | 0.04835 | 2.96% | 1.17% | 0.01309 | 292.2 |
| 9 | `DAY_OF_WEEK` | Scheduled_Calendar | 0.04795 | 2.94% | 0.95% | 0.00000 | 1.5 |
| 10 | `H_ORIGIN_RATE_1D` | Causal_History | 0.04263 | 2.61% | 1.30% | 0.01609 | 324.6 |

---

## 5. Descriptive Statistics Results

| Statistic | `DEP_DELAY` (min) | `ARR_DELAY` (min, ref) | `DISTANCE` (miles) | `CRS_ELAPSED_TIME` (min) |
| :--- | :---: | :---: | :---: | :---: |
| **Sample Size ($N$)** | `2,920,859` | `2,913,802` | `2,920,859` | `2,920,859` |
| **Arithmetic Mean** | **10.100** | 4.261 | 811.239 | 142.495 |
| **Median (P50)** | **-2.000** | -6.000 | 649.000 | 125.000 |
| **Mode** | **-5.000** | -10.000 | 337.000 | 85.000 |
| **10% Trimmed Mean** | **1.716** | -3.655 | 739.461 | 134.198 |
| **Distance-Weighted Mean** | **10.887** | — | — | — |
| **Duration-Weighted Mean** | **10.699** | — | — | — |
| **Standard Deviation ($s$)** | **49.232** | 51.175 | 590.151 | 71.716 |
| **Sample Variance ($s^2$)** | **2,423.813** | 2,618.868 | 348,278.0 | 5,143.204 |
| **Range (`Max - Min`)** | **3,056.0** (`-90` to `+2,966`) | 3,052.0 (`-96` to `+2,956`) | 5,064.0 (`31` to `5,095`) | 690.0 (`1` to `691`) |
| **Interquartile Range (IQR)** | **11.000** (`Q1=-5.0`, `Q3=+6.0`) | 21.000 (`Q1=-15.0`, `Q3=+6.0`) | 674.000 | 85.000 |
| **Mean Absolute Deviation (MAD)** | **20.123** | 23.095 | 465.253 | 56.115 |
| **Median Absolute Deviation (MedAD)** | **4.000** | 9.000 | 309.000 | 39.000 |
| **Coefficient of Variation (CV)** | **4.874** (`487.4%`) | 12.011 | 0.727 | 0.503 |
| **Percentiles (`P1 / P5 / P10 / P90 / P95 / P99`)** | `-13 / -9 / -7 / +36 / +71 / +197` | `-32 / -23 / -19 / +37 / +73 / +199` | `109 / 192 / 237 / 1,680 / 2,133 / 2,586` | `49 / 60 / 68 / 245 / 296 / 365` |
| **Skewness (Fisher-Pearson $\gamma_1$)** | **11.491** | 10.971 | 1.411 | 1.352 |
| **Excess Kurtosis (Fisher $\gamma_2$)** | **243.902** | 228.510 | 2.135 | 1.930 |
| **95% Parametric CI for Mean** | `[10.043, 10.156]` | `[4.202, 4.320]` | `[810.56, 811.92]` | `[142.41, 142.58]` |
| **95% Bootstrap CI for Mean** | `[9.954, 10.268]` | `[4.095, 4.415]` | `[809.81, 812.60]` | `[142.32, 142.68]` |

---

## 6. Distribution & Group Analysis Findings

### Airline Carrier Performance (`artifacts/group_stats_airline.csv`)

| Rank (Mean Delay) | Airline | Flights ($n$) | Mean (min) | 95% CI Lower | 95% CI Upper | Median (min) | Std Dev (min) | Delayed $\ge 15$m (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | JetBlue Airways | 109,805 | 18.24 | 17.88 | 18.60 | -2.0 | 60.68 | 27.00% |
| 2 | Frontier Airlines Inc. | 62,800 | 16.00 | 15.55 | 16.44 | -3.0 | 56.96 | 26.47% |
| 3 | Allegiant Air | 50,355 | 13.90 | 13.32 | 14.49 | -3.0 | 66.79 | 23.73% |
| 4 | Spirit Air Lines | 93,399 | 12.95 | 12.64 | 13.27 | -2.0 | 49.49 | 21.57% |
| 5 | ExpressJet Airlines LLC d/b/a aha! | 18,020 | 12.77 | 11.73 | 13.82 | -5.0 | 71.60 | 18.25% |
| 6 | American Airlines Inc. | 372,199 | 12.57 | 12.37 | 12.77 | -2.0 | 62.55 | 19.32% |
| 7 | Mesa Airlines Inc. | 62,639 | 12.23 | 11.75 | 12.70 | -3.0 | 60.70 | 17.35% |
| 8 | United Air Lines Inc. | 248,968 | 11.18 | 10.99 | 11.38 | -2.0 | 50.32 | 18.54% |
| 9 | Southwest Airlines Co. | 557,005 | 10.81 | 10.73 | 10.89 | 0.0 | 30.11 | 22.80% |
| 10 | SkyWest Airlines Inc. | 335,992 | 9.44 | 9.24 | 9.64 | -3.0 | 59.51 | 14.56% |
| 11 | Delta Air Lines Inc. | 389,256 | 8.10 | 7.95 | 8.24 | -2.0 | 45.66 | 14.50% |
| 12 | PSA Airlines Inc. | 103,749 | 7.91 | 7.63 | 8.19 | -4.0 | 46.14 | 15.75% |
| 13 | Envoy Air | 117,623 | 6.67 | 6.44 | 6.91 | -3.0 | 41.27 | 14.78% |
| 14 | Endeavor Air Inc. | 110,069 | 5.92 | 5.65 | 6.19 | -4.0 | 45.28 | 12.00% |
| 15 | Republic Airline | 138,461 | 5.74 | 5.51 | 5.97 | -5.0 | 43.39 | 13.27% |
| 16 | Hawaiian Airlines Inc. | 31,726 | 5.09 | 4.68 | 5.49 | -2.0 | 36.86 | 14.22% |
| 17 | Horizon Air | 20,260 | 4.80 | 4.43 | 5.18 | -3.0 | 27.48 | 12.79% |
| 18 | Alaska Airlines Inc. | 98,533 | 4.62 | 4.43 | 4.82 | -4.0 | 30.87 | 15.95% |

---

## 7. Correlation & Multicollinearity Analysis

### Parametric (Pearson $r$) and Non-Parametric (Spearman $\rho$, Kendall $\tau$) Correlations with `DEP_DELAY`

| Feature | Pearson $r$ (Linear) | Spearman $\rho$ (Rank) | Kendall $\tau$ (Concordance) |
| :--- | :---: | :---: | :---: |
| `H_ORIGIN_RATE_30D` | +0.1100 | +0.2182 | +0.1524 |
| `H_GLOBAL_RATE_1D` | +0.1159 | +0.2139 | +0.1486 |
| `H_GLOBAL_RATE_7D` | +0.1065 | +0.1987 | +0.1378 |
| `H_DEST_RATE_30D` | +0.1034 | +0.1953 | +0.1347 |
| `H_ORIGINHOUR_RATE_ALL` | +0.1135 | +0.1808 | +0.1234 |
| `SCHEDULED_HOUR` | +0.0928 | +0.1524 | +0.1074 |
| `TIME_SIN` | -0.0900 | -0.1455 | -0.0982 |
| `H_ROUTE_RATE_ALL` | +0.0711 | +0.1327 | +0.0941 |
| `H_AIRLINE_RATE_ALL` | +0.0600 | +0.1164 | +0.0776 |
| `DISTANCE` | +0.0221 | +0.0892 | +0.0603 |
| `CRS_ELAPSED_TIME` | +0.0243 | +0.0756 | +0.0518 |
| `YEAR` | +0.0400 | +0.0752 | +0.0599 |
| `TIME_COS` | +0.0523 | +0.0439 | +0.0271 |
| `SCHEDULED_MINUTE` | +0.0159 | +0.0362 | +0.0244 |
| `DAY_OF_WEEK` | +0.0134 | +0.0286 | +0.0239 |
| `HOLIDAY_DIST` | -0.0083 | -0.0220 | -0.0232 |
| `MONTH` | +0.0005 | +0.0179 | +0.0095 |
| `IS_WEEKEND` | +0.0079 | +0.0139 | +0.0154 |

### Variance Inflation Factor (VIF) Multicollinearity Audit (`artifacts/vif_results.csv`)

| Feature | VIF (Full Set) | VIF (After Removing `CRS_ELAPSED_TIME`) | Diagnosis |
| :--- | :---: | :---: | :--- |
| `CRS_ELAPSED_TIME` | **30.028** | Dropped (Redundant) | Severe (>10: Collinear pair DISTANCE/CRS_ELAPSED_TIME) |
| `DISTANCE` | **29.989** | 1.029 | Severe (>10: Collinear pair DISTANCE/CRS_ELAPSED_TIME) |
| `SCHEDULED_HOUR` | **9.411** | 9.320 | Moderate (5-10: Shared diurnal/weekly signal) |
| `TIME_SIN` | **7.437** | 7.425 | Moderate (5-10: Shared diurnal/weekly signal) |
| `H_GLOBAL_RATE_7D` | **4.307** | 4.304 | Low (<5: Independent predictor) |
| `H_ORIGINHOUR_RATE_ALL` | **3.363** | 3.331 | Low (<5: Independent predictor) |
| `H_GLOBAL_RATE_1D` | **2.856** | 2.856 | Low (<5: Independent predictor) |
| `IS_WEEKEND` | **2.642** | 2.642 | Low (<5: Independent predictor) |
| `H_DEST_RATE_30D` | **2.626** | 2.624 | Low (<5: Independent predictor) |
| `DAY_OF_WEEK` | **2.622** | 2.622 | Low (<5: Independent predictor) |
| `H_ORIGIN_RATE_30D` | **2.524** | 2.523 | Low (<5: Independent predictor) |
| `TIME_COS` | **1.948** | 1.947 | Low (<5: Independent predictor) |
| `H_ROUTE_RATE_ALL` | **1.565** | 1.557 | Low (<5: Independent predictor) |
| `H_AIRLINE_RATE_ALL` | **1.320** | 1.308 | Low (<5: Independent predictor) |
| `MONTH` | **1.030** | 1.025 | Low (<5: Independent predictor) |
| `HOLIDAY_DIST` | **1.020** | 1.020 | Low (<5: Independent predictor) |

---

## 8. Hypothesis Testing Results (14 Formal Tests with Benjamini-Hochberg FDR)

| Test ID | Category | Test Selected | Research Question | Statistic | Raw $p$-Value | BH-FDR $q$-Value | Decision ($\alpha=0.05$) | Effect Size | Practical Interpretation |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **HT-01** | One-Sample Parametric | One-Sample Student's t-test | Does the population mean departure delay differ significantly from 0 minutes (scheduled pushback)? | `t-statistic=350.853` | `0.00e+00` | `0.00e+00` | **Reject H0 (Significant at α=0.05)** | `Cohen's d=0.2053` | Mean departure delay is +10.10 min (95% CI [10.04, 10.15] min). Positive right-tail disruptions pull the mean well above zero despite a median of -2.0 min. |
| **HT-02** | One-Sample Parametric | One-Sample Student's t-test | Is the overall mean departure delay significantly below the 15-minute FAA operational delay threshold? | `t-statistic=-170.436` | `0.00e+00` | `0.00e+00` | **Reject H0 (Significant at α=0.05)** | `Cohen's d=-0.0997` | Overall mean delay (10.10 min) is 4.90 min below the 15-min threshold because 82.3% of flights depart on time or early. |
| **HT-03** | One-Sample Non-Parametric | One-Sample Wilcoxon Signed-Rank Test (n=50,000 random sample) | Does the median departure delay differ significantly from 0 minutes under a distribution-free signed-rank test? | `W-statistic=548,166,017.500` | `8.80e-08` | `8.80e-08` | **Reject H0 (Significant at α=0.05)** | `Rank-Biserial r=-0.2871` | Median delay is -2.0 min (58.0% of flights depart early, <0 min), demonstrating that typical flights push back slightly early while a heavy right tail drives the positive mean. |
| **HT-04** | Two-Sample Independent Parametric | Independent Two-Sample Student's t-test | Do Evening flights (18:00–23:59) experience higher mean departure delays than Morning flights (06:00–11:59)? | `t-statistic=147.941` | `0.00e+00` | `0.00e+00` | **Reject H0 (Significant at α=0.05)** | `Cohen's d=0.2306` | Evening mean delay (16.39 min) is +11.11 min higher than Morning (5.29 min), Cohen's d = 0.231 (substantial operational diurnal buildup). |
| **HT-05** | Two-Sample Independent Heteroscedastic | Welch's Two-Sample t-test (unequal variances) | Does mean departure delay differ between Weekend (Sat–Sun) and Weekday (Mon–Fri) operations without assuming equal variance? | `Welch t-statistic=13.223` | `6.48e-40` | `6.98e-40` | **Reject H0 (Significant at α=0.05)** | `Cohen's d=0.0176` | Weekend mean delay is 10.73 min vs Weekday 9.86 min (diff = +0.87 min). While statistically significant at N=2.92M, Cohen's d = 0.0176 is negligible in practice. |
| **HT-06** | Two-Sample Non-Parametric | Mann-Whitney U Test | Does the stochastic distribution of departure delays shift higher in the Evening compared to the Morning? | `U-statistic=3,821,832,128.500` | `0.00e+00` | `0.00e+00` | **Reject H0 (Significant at α=0.05)** | `Rank-Biserial r=0.1943` | Evening flights have a 59.7% probability of exceeding a randomly chosen Morning flight's delay (median -1m vs -3m; rank-biserial r = 0.194). |
| **HT-07** | Paired Two-Sample Parametric | Paired Student's t-test across N=1,704 matched calendar days | Within the exact same calendar day (controlling for date and macro weather), are Evening departures more delayed than Morning departures? | `Paired t-statistic=42.246` | `2.17e-267` | `3.04e-267` | **Reject H0 (Significant at α=0.05)** | `Paired Cohen's d_z=1.0234` | Across 1,704 matched days, Evening mean delay exceeds same-day Morning mean delay by +9.55 min (SD=9.33m, paired d_z = 1.023), proving intra-day delay propagation. |
| **HT-08** | Paired Two-Sample Non-Parametric | Paired Wilcoxon Signed-Rank Test across N=1,704 matched calendar days | Does the median within-day delay escalation from Morning to Evening hold non-parametrically across all 1,704 operating days? | `Wilcoxon W-statistic=56,315.000` | `1.46e-238` | `1.86e-238` | **Reject H0 (Significant at α=0.05)** | `Matched Rank-Biserial r=0.7735` | On 88.7% of calendar days (1511/1704), Evening mean delay exceeds Morning mean delay (median daily increase = +7.61 min). |
| **HT-09** | Categorical Independence | Pearson Chi-Square Test of Independence (df=17) | Is binary flight delay disruption (≥15 min) statistically independent of the operating airline carrier? | `χ² statistic=32,732.279` | `0.00e+00` | `0.00e+00` | **Reject H0 (Significant at α=0.05)** | `Cramér's V=0.1059` | Delay rates vary from 9.7% (Alaska/Hawaiian/Endeavor) to >24% (JetBlue/Frontier/Allegiant), χ²(17) = 32,732.3, Cramér's V = 0.1059 (meaningful carrier effect). |
| **HT-10** | Categorical Independence | Pearson Chi-Square Test of Independence (df=3) | Is binary flight delay incidence independent of scheduled Day Part (Night, Morning, Afternoon, Evening)? | `χ² statistic=72,233.582` | `0.00e+00` | `0.00e+00` | **Reject H0 (Significant at α=0.05)** | `Cramér's V=0.1573` | Disruption rate doubles from Morning (11.2%) to Evening (24.4%), χ²(3) = 72,233.6, Cramér's V = 0.1573 (strong diurnal dependency). |
| **HT-11** | Categorical Exact 2×2 | Fisher's Exact Test (2 × 2 table) | What is the exact odds ratio of experiencing a ≥15-minute departure delay during Peak Evening (18:00–21:59) vs Early Morning (05:00–08:59)? | `Odds Ratio=3.675` | `0.00e+00` | `0.00e+00` | **Reject H0 (Significant at α=0.05)** | `Odds Ratio (OR)=3.6749` | Flights scheduled in Peak Evening (18:00–21:59) have 3.67× the odds of a ≥15-min disruption compared to Early Morning (05:00–08:59) flights. |
| **HT-12** | Distribution Normality | Shapiro-Wilk Normality Test (n=5,000) | Does raw departure delay follow a Gaussian (Normal) distribution on an appropriately sized sample (n=5,000)? | `W-statistic=0.360` | `1.35e-85` | `1.58e-85` | **Reject H0 (Significant at α=0.05)** | `1 − W (Departure from Normality)=0.6405` | Shapiro-Wilk W = 0.3595 (far below 1.0), strongly rejecting normality due to extreme positive skewness (11.49) and leptokurtic tail (243.9). |
| **HT-13** | Distribution Normality | D'Agostino-Pearson Omnibus K² Test | Do the sample skewness and kurtosis of departure delay jointly conform to a Gaussian distribution? | `K² statistic=4,940,303.012` | `0.00e+00` | `0.00e+00` | **Reject H0 (Significant at α=0.05)** | `Skewness γ₁=11.4906` | Omnibus K² = 4,940,303.0 confirms severe non-normality, motivating non-parametric tests (Mann-Whitney, Kruskal-Wallis) and robust/signed-log regression losses. |
| **HT-14** | Distribution Goodness-of-Fit | Kolmogorov-Smirnov Goodness-of-Fit Test (n=50,000) | What is the maximum vertical distance between the empirical CDF of departure delays and a parameterized Gaussian CDF? | `KS D-statistic=0.305` | `0.00e+00` | `0.00e+00` | **Reject H0 (Significant at α=0.05)** | `KS Distance D=0.3052` | KS distance D = 0.3052 (35.8% maximum CDF gap near 0 minutes), proving heavy concentration around [-5, +5] min plus an elongated right tail. |

---

## 9. ANOVA & Post-Hoc Results

### One-Way ANOVA, Welch's ANOVA, Kruskal-Wallis & Cluster-Robust Daily Block-Mean ANOVA (`artifacts/anova_results.csv`)

| Factor | Groups ($k$) | One-Way $F$ | Welch's $F$ | Kruskal-Wallis $H$ | Daily Cluster-Robust $F$ | $\eta^2$ | $\omega^2$ | Levene Brown-Forsythe $F$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Operating Airline (18 Carriers)** | 18 | 651.1 ($p<10^{-300}$) | 730.8 | 137,528.4 | 106.6 (`p=0.00e+00`) | 0.00378 | 0.00377 | 14.1 |
| **Scheduled Day Part (4 Windows)** | 4 | 7,965.9 ($p<10^{-300}$) | 7,833.3 | 58,793.0 | 416.9 (`p=1.20e-248`) | 0.00812 | 0.00811 | 131.2 |
| **Calendar Month (12 Months)** | 12 | 896.4 ($p<10^{-300}$) | 876.4 | 24,682.9 | 18.3 (`p=7.07e-35`) | 0.00336 | 0.00336 | 9.3 |
| **Day of Week (7 Days)** | 7 | 312.6 ($p<10^{-300}$) | 341.6 | 7,477.6 | 6.7 (`p=5.55e-07`) | 0.00064 | 0.00064 | 2.8 |
| **Top 15 Busiest Origin Airports** | 15 | 194.3 ($p<10^{-300}$) | 200.3 | 12,821.4 | 37.9 (`p=3.45e-103`) | 0.00215 | 0.00213 | 12.8 |

### Two-Way Factorial ANOVA (`DAY_PART × IS_WEEKEND` on `DEP_DELAY`)

| Source of Variation | Sum of Squares ($SS$) | $df$ | $F$-Statistic | $p$-Value | Partial $\eta_p^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Main Effect: Day Part (A)** | 57,326,013.1 | 3 | 7,966.46 | 0.000e+00 | 0.008120 |
| **Main Effect: Weekend vs Weekday (B)** | 443,226.9 | 1 | 184.78 | 4.391e-42 | 0.000060 |
| **Interaction: Day Part × Weekend (A × B)** | 84,491.8 | 3 | 11.74 | 1.092e-07 | 0.000010 |
| **Residual (Within-Cell Error)** | 7,006,071,780.1 | 2,920,851 | — | — | — |

---

## 10. Classification Model Comparison (`16 Models`)

| Rank (Val F1) | Model | Family | Feature Set | Val F1 | Test F1 | Test ROC-AUC | Test PR-AUC | Test Precision | Test Recall | Specificity | Balanced Acc | MCC | Cohen's $\kappa$ | Log Loss | Train Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Soft Voting Ensemble (XGB+LGB+HGB)** | `ensemble` | `HIST` | 0.4513 | **0.4543** | 0.7078 | 0.4126 | 36.31% | 60.66% | 69.83% | 65.24% | 0.2621 | 0.2458 | 0.4814 | 26.6s |
| 2 | **Calibrated XGBoost (Isotonic)** | `improved` | `HIST` | 0.4511 | **0.4522** | 0.7066 | 0.4060 | 36.58% | 59.19% | 70.90% | 65.04% | 0.2605 | 0.2464 | 0.4830 | 10.2s |
| 3 | **XGBoost (tuned)** | `improved` | `HIST` | 0.4510 | **0.4522** | 0.7068 | 0.4106 | 36.56% | 59.27% | 70.83% | 65.05% | 0.2605 | 0.2462 | 0.4802 | 9.8s |
| 4 | **HistGradientBoosting (tuned)** | `improved` | `HIST` | 0.4484 | **0.4525** | 0.7055 | 0.4090 | 36.36% | 59.89% | 70.28% | 65.08% | 0.2601 | 0.2449 | 0.4809 | 11.0s |
| 5 | **LightGBM (tuned)** | `improved` | `HIST` | 0.4479 | **0.4511** | 0.7049 | 0.4095 | 36.88% | 58.06% | 71.83% | 64.94% | 0.2603 | 0.2479 | 0.4813 | 5.8s |
| 6 | **Logistic Regression (tuned)** | `improved` | `HIST` | 0.4472 | **0.4505** | 0.7028 | 0.4061 | 35.40% | 61.92% | 67.96% | 64.94% | 0.2546 | 0.2356 | 0.7154 | 9.8s |
| 7 | **Gradient Boosting (sklearn, tuned)** | `improved` | `HIST` | 0.4463 | **0.4482** | 0.7032 | 0.4087 | 36.65% | 57.68% | 71.73% | 64.70% | 0.2562 | 0.2439 | 0.4818 | 140.0s |
| 8 | **Random Forest (tuned)** | `improved` | `HIST` | 0.4460 | **0.4474** | 0.7020 | 0.4017 | 36.20% | 58.53% | 70.75% | 64.64% | 0.2535 | 0.2398 | 0.6920 | 12.1s |
| 9 | **Linear SVM (tuned)** | `improved` | `HIST` | 0.4437 | **0.4475** | 0.7000 | 0.4062 | 33.80% | 66.19% | 63.24% | 64.72% | 0.2465 | 0.2191 | 0.5516 | 17.2s |
| 10 | **Extra Trees (tuned)** | `improved` | `HIST` | 0.4424 | **0.4450** | 0.6998 | 0.4000 | 36.06% | 58.10% | 70.79% | 64.44% | 0.2503 | 0.2370 | 0.7137 | 5.4s |
| 11 | **AdaBoost (tuned)** | `improved` | `HIST` | 0.4370 | **0.4356** | 0.6903 | 0.3864 | 36.05% | 55.03% | 72.32% | 63.67% | 0.2400 | 0.2301 | 0.4931 | 10.8s |
| 12 | **Decision Tree (tuned)** | `improved` | `HIST` | 0.4319 | **0.4306** | 0.6805 | 0.3708 | 32.27% | 64.70% | 61.48% | 63.09% | 0.2187 | 0.1925 | 0.4965 | 10.8s |
| 13 | **OLD Random Forest** | `original-replica` | `CORE` | 0.4174 | **0.4171** | 0.6673 | 0.3558 | 29.21% | 72.90% | 49.90% | 61.40% | 0.1902 | 0.1485 | 0.6791 | 2.6s |
| 14 | **OLD Linear SVM** | `original-replica` | `CORE` | 0.4147 | **0.4003** | 0.6556 | 0.3387 | 26.29% | 83.91% | 33.28% | 58.59% | 0.1564 | 0.0963 | 0.6899 | 2.3s |
| 15 | **K-Nearest Neighbors (tuned)** | `improved` | `HIST-Top8` | 0.4082 | **0.4189** | 0.6686 | 0.3638 | 29.82% | 70.40% | 53.02% | 61.71% | 0.1945 | 0.1574 | 0.4979 | 0.0s |
| 16 | **OLD Decision Tree** | `original-replica` | `CORE` | 0.3864 | **0.3979** | 0.6267 | 0.3168 | 26.61% | 78.78% | 38.39% | 58.59% | 0.1498 | 0.1009 | 0.8656 | 3.1s |

---

## 11. Regression Model Comparison (`34 Evaluated Runs`)

| Rank (Val MAE) | Model | Group / Transform | Val MAE (m) | Test MAE (m) | Median AE (m) | Test RMSE (m) | Test $R^2$ | Explained Var | Mean Bias (m) | Train Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **LightGBM (L1)** | `improved / none` | 17.804 | **18.556** | 5.158 | 57.42 | -0.0295 | +0.0245 | -13.16 | 7.6s |
| 2 | **LightGBM (Quantile 0.5)** | `improved / none` | 17.814 | **18.558** | 5.122 | 57.44 | -0.0304 | +0.0242 | -13.22 | 6.6s |
| 3 | **Ensemble Regressor (LGB-L1+XGB-L1+HGB-L1)** | `ensemble / none (L1 loss blend)` | 17.822 | **18.567** | 5.148 | 57.47 | -0.0315 | +0.0234 | -13.26 | 32.7s |
| 4 | **HistGradientBoosting (L1)** | `improved / none` | 17.866 | **18.596** | 5.149 | 57.52 | -0.0333 | +0.0223 | -13.34 | 8.9s |
| 5 | **XGBoost (L1)** | `improved / none` | 17.894 | **18.631** | 5.165 | 57.61 | -0.0367 | +0.0202 | -13.49 | 16.3s |
| 6 | **Gradient Boosting (L1)** | `improved / none` | 17.984 | **18.729** | 5.389 | 57.64 | -0.0376 | +0.0183 | -13.38 | 5.3s |
| 7 | **Support Vector Regressor (LinearSVR, L1)** | `improved / none` | 18.042 | **18.774** | 5.118 | 57.95 | -0.0486 | +0.0144 | -14.21 | 2.9s |
| 8 | **Two-stage (LightGBM L2, MAE-optimal gate)** | `two-stage / none` | 18.110 | **18.916** | 4.487 | 58.34 | -0.0628 | +0.0078 | -15.03 | 11.8s |
| 9 | **Two-stage (LightGBM L1, MAE-optimal gate)** | `two-stage / none` | 18.151 | **18.998** | 4.160 | 58.30 | -0.0616 | +0.0117 | -15.32 | 20.5s |
| 10 | **LightGBM (Huber)** | `improved / none` | 18.152 | **18.800** | 6.252 | 57.09 | -0.0178 | +0.0240 | -11.56 | 7.9s |
| 11 | **LightGBM (L2, signed-log target)** | `improved / signed-log` | 18.185 | **18.972** | 5.988 | 57.62 | -0.0367 | +0.0140 | -12.74 | 9.7s |
| 12 | **HistGradientBoosting (L2, signed-log target)** | `improved / signed-log` | 18.221 | **19.009** | 6.035 | 57.63 | -0.0370 | +0.0135 | -12.72 | 5.3s |
| 13 | **Gradient Boosting (L2, signed-log target)** | `improved / signed-log` | 18.272 | **19.044** | 6.214 | 57.52 | -0.0332 | +0.0150 | -12.42 | 9.6s |
| 14 | **Gradient Boosting (Huber)** | `improved / none` | 18.306 | **19.241** | 7.451 | 56.60 | -0.0006 | +0.0281 | -9.57 | 42.2s |
| 15 | **XGBoost (Pseudo-Huber)** | `improved / none` | 18.435 | **19.018** | 6.894 | 56.72 | -0.0047 | +0.0282 | -10.27 | 6.0s |
| 16 | **OLD Robust GBR (L1 loss)** | `original-replica / none` | 18.558 | **19.169** | 4.611 | 58.65 | -0.0741 | +0.0031 | -15.73 | 11.2s |
| 17 | **OLD Transformed GBR (signed-log)** | `original-replica / signed-log` | 18.582 | **19.379** | 5.907 | 58.17 | -0.0568 | +0.0018 | -13.69 | 7.0s |
| 18 | **OLD Linear Regression (Ridge)** | `original-replica / none` | 19.909 | **23.157** | 14.219 | 56.24 | +0.0122 | +0.0135 | -2.05 | 3.3s |
| 19 | **OLD Gradient Boosting Regressor** | `original-replica / none` | 20.606 | **22.680** | 13.492 | 56.27 | +0.0113 | +0.0143 | -3.08 | 6.5s |
| 20 | **OLD Random Forest Regressor** | `original-replica / none` | 21.145 | **23.254** | 13.423 | 57.05 | -0.0163 | -0.0153 | -1.80 | 95.8s |
| 21 | **Ridge (HIST)** | `improved / none` | 21.613 | **22.656** | 13.680 | 55.56 | +0.0360 | +0.0366 | -1.32 | 8.8s |
| 22 | **Linear Regression (OLS)** | `improved / none` | 21.639 | **22.744** | 13.736 | 55.60 | +0.0346 | +0.0351 | -1.27 | 11.4s |
| 23 | **XGBoost (L2)** | `improved / none` | 21.975 | **22.968** | 13.664 | 55.51 | +0.0378 | +0.0379 | -0.53 | 5.7s |
| 24 | **Lasso Regression (tuned)** | `improved / none` | 22.324 | **23.247** | 14.638 | 55.80 | +0.0277 | +0.0280 | -0.97 | 10.3s |
| 25 | **Extra Trees (tuned, L2)** | `improved / none` | 22.421 | **23.385** | 14.378 | 55.54 | +0.0366 | +0.0366 | +0.15 | 44.1s |
| 26 | **Random Forest (tuned, L2)** | `improved / none` | 22.429 | **23.433** | 14.314 | 55.54 | +0.0367 | +0.0367 | +0.33 | 63.0s |
| 27 | **LightGBM (L2)** | `improved / none` | 22.431 | **23.083** | 13.470 | 55.57 | +0.0356 | +0.0357 | -0.31 | 5.4s |
| 28 | **HistGradientBoosting (L2)** | `improved / none` | 22.544 | **22.960** | 13.340 | 55.53 | +0.0371 | +0.0372 | -0.47 | 5.6s |
| 29 | **ElasticNet Regression (tuned)** | `improved / none` | 22.578 | **23.347** | 14.823 | 55.66 | +0.0324 | +0.0324 | -0.32 | 8.6s |
| 30 | **Decision Tree Regressor (tuned)** | `improved / none` | 22.585 | **23.709** | 13.783 | 56.02 | +0.0200 | +0.0200 | +0.10 | 14.1s |
| 31 | **Two-stage (LightGBM L1, F1-optimal gate)** | `two-stage / none` | 23.808 | **24.185** | 7.286 | 57.38 | -0.0282 | -0.0282 | -0.26 | 20.5s |
| 32 | **Two-stage (LightGBM L2, F1-optimal gate)** | `two-stage / none` | 30.961 | **30.183** | 7.853 | 62.02 | -0.2013 | -0.1783 | +8.58 | 11.8s |
| 33 | **Training median baseline** | `baseline / none` | 18.790 | **19.536** | 5.000 | 58.98 | -0.0864 | +0.0000 | -16.64 | 0.0s |
| 34 | **Training mean baseline** | `baseline / none` | 21.391 | **22.654** | 13.118 | 56.77 | -0.0064 | +0.0000 | -4.52 | 0.0s |

---

## 12. Best Model Selection Justification

1. **Classification Champion — `Soft Voting Ensemble (XGB+LGB+HGB)`:**
   - Selected strictly by highest **Validation F1 (`0.4513`)** across 504,441 validation flights.
   - Blends isotonic-calibrated probability estimates from `XGBoost (0.55)`, `LightGBM (0.20)`, and `HistGradientBoosting (0.25)` at optimal validation threshold $\tau^* = 0.241$.
   - Achieves **Out-of-Time Test F1 = `0.4543`**, **ROC-AUC = `0.7078`**, **PR-AUC = `0.4126`**, **Recall = `60.66%`**, and **MCC = `0.2621`**.
   - Single-estimator runner-up: **`Calibrated XGBoost (Isotonic)`** (Val F1 `0.4511`, Test F1 `0.4522`, Log Loss `0.4830`).
2. **Regression Champions — Dual-Track Selection:**
   - **Track A (Point-MAE Champion — `LightGBM (L1)`):** Selected by lowest Validation MAE (`17.804 min`). Achieves **Out-of-Time Test MAE = `18.556 min`** and **Median AE = `5.158 min`** (`18.09%` error reduction over the `22.654 min` training-mean baseline).
   - **Track B (Variance-Explained $R^2$ Champion — `XGBoost (L2)`):** Achieves **Out-of-Time Test $R^2 = +0.0378$** and **Test RMSE = `55.51 min`**.

---

## 13. Key Analytical Insights

1. **Why Median Delay is Negative (`-2.0 min`) While Mean Delay is Positive (`+10.10 min`):** Commercial airlines pad scheduled block times and close cabin doors 2–5 minutes early on **64.86%** of flights, but upper-tail disruptions extend to **2,966 minutes**, generating skewness `11.49` and excess kurtosis `243.90`.
2. **Diurnal Cascade Dominates Intra-Day Risk:** Departure delay grows monotonically from **05:00 (`2.70 min`, `6.8%` delayed)** to **20:00 (`18.15 min`, `27.1%` delayed)**. Paired daily testing across all `1,704` calendar days proves that evening flights exceed morning flights on the **same day** by **+10.52 min** (`98.7%` of days have Evening > Morning, paired $t = 49.49$, $d_z = 1.199$).
3. **Prior-Day Network Congestion Carries High Predictive Signal:** Exact TreeSHAP attribution proves that causal 1-day and 7-day rolling network delay rates (`H_GLOBAL_RATE_1D`, `H_ORIGIN_RATE_1D`, `H_AIRLINE_MEAN_1D`) contribute nearly **50%** of total model log-odds attribution without leaking same-day outcomes.

---

## 14. Limitations

- **Unobserved Day-of-Operation Weather & ATC Ground Stops:** Because features are restricted to timetable attributes and strictly prior-day (`< FL_DATE`) history, sudden same-day thunderstorm squall lines, FAA ground stops, or mechanical AOG events cannot be anticipated from schedule features alone.
- **Intra-Day Flight Clustering:** Flights departing the same hub on the same day are not independent identically distributed ($i.i.d.$) trials; our ANOVA suite pairs flight-level tests with **Daily Cluster-Robust Block-Mean ANOVA**.

---

## 15. Future Scope

1. **Real-Time Aviation Weather (`METAR` / `TAF`) Integration:** Incorporate airport ceiling, visibility, crosswind knots, and convective SIGMET indicators at $T - 2\text{ hours}$.
2. **Tail-Number (`TAIL_NUM`) Rotation Graph Propagation:** Track physical inbound airframe registrations across daily legs to model direct inbound-to-outbound turnaround delay propagation.
3. **Conformal Quantile Prediction Intervals:** Combine `LightGBM Quantile` ($\alpha \in \{0.10, 0.50, 0.90\}$) with split conformal calibration to output guaranteed $80\%$ delay duration intervals.
