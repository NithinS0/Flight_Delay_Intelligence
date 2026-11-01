# Flight Delay Prediction and Statistical Intelligence System
## Consolidated Project Key Metrics & Numerical Ledger

> **Purpose:** This reference document provides the single, verified source of truth for all numerical figures, dataset dimensions, statistical findings, and machine learning evaluation metrics across the project. Every number reflects the actual saved experiment artifacts in `artifacts/`.

---

## 1. Dataset Dimensions & Provenance

* **Dataset Source:** United States Department of Transportation (USDOT) / Bureau of Transportation Statistics (BTS) TranStats Airline On-Time Performance Reporting Database.
* **Primary Raw File:** `flight_data.csv` (Size: `614.1 MB`).
* **Raw Records:** `3,000,000` flight records $\times$ `32` raw columns.
* **Temporal Horizon:** January 1, 2019 to August 31, 2023 (`56 consecutive calendar months`).
* **Geographic & Carrier Coverage:**
  * **Operating Airlines:** `18` unique US commercial passenger air carriers.
  * **Origin Airports:** `378` unique US commercial airports.
  * **Destination Airports:** `378` unique US destination airports.
  * **Flight Corridors (Routes):** `7,785` unique directional origin–destination pairs.
  * **Airline–Route Combinations:** `13,015` unique carrier-corridor operations.

---

## 2. Data Cleaning & Population Governance

* **Duplicate Flight Keys:** `72` records identified as duplicate flights (`0.002%`) and removed.
* **Cancelled Flights Excluded:** `79,140` cancelled flights (`2.64%`) where departure never occurred (`CANCELLED == 1`).
* **Diverted Flights Retained:** `8,261` diverted flights (`0.28%`) where pushback occurred and departure delay is physically measured.
* **Cleaned Population:** `2,920,859` completed commercial flights.
* **Missing Values in Primary Schedule Features:** `0` missing values across `AIRLINE`, `ORIGIN`, `DEST`, `DISTANCE`, `CRS_DEP_TIME`, and `FL_DATE`.
* **Missing Delay Reasons (`NaN` Policy):** `DELAY_DUE_CARRIER`, `DELAY_DUE_WEATHER`, `DELAY_DUE_NAS`, `DELAY_DUE_SECURITY`, and `DELAY_DUE_LATE_AIRCRAFT` are missing on `81.9%` of rows (flights not delayed $\ge 15$ min). Retained as `NaN` and strictly excluded from predictive inputs.
* **Extreme Outlier Preservation:** Maximum recorded departure delay is `+2,966 minutes` (~`49.4 hours`). All genuine extreme delays are preserved without clipping or artificial removal.
* **Natural Class Distribution (FAA Delay Standard):**
  * **On-Time / Minor Buffer (`DELAYED = 0`, `< 15 min`):** `2,404,458` flights (`82.32%`).
  * **Delayed (`DELAYED = 1`, `≥ 15 min`):** `516,401` flights (`17.68%`).
  * **Zero Synthetic Balancing:** Natural operational class ratio (`4.66 : 1`) is preserved. SMOTE, ADASYN, and synthetic oversampling were explicitly excluded.

---

## 3. Chronological Time-Series Partitioning

To guarantee zero lookahead data leakage, all splits strictly adhere to chronological calendar dates:

| Partition | Calendar Date Range | Completed Flights | Share | Primary Function |
| :--- | :--- | :--- | :--- | :--- |
| **Fit / Train Split** | `2019-01-01` to `2021-12-24` | `1,784,112` | `61.08%` | Initial model parameter estimation & 3-fold `TimeSeriesSplit` |
| **Validation Split** | `2021-12-25` to `2022-09-25` | `504,441` | `17.27%` | Hyperparameter selection, threshold optimization ($\tau^*$), calibration |
| **Out-of-Time Test Split** | `2022-09-26` to `2023-08-31` | `632,306` | `21.65%` | Untouched, single final benchmark evaluation |
| **Total Analyzed** | `2019-01-01` to `2023-08-31` | **`2,920,859`** | **`100.0%`** | Entire analytical population |

---

## 4. Feature Engineering Architecture (`74 Features`)

* **Core Schedule Features (`BASE`, 8 features):** `AIRLINE`, `ORIGIN`, `DEST`, `DISTANCE`, `CRS_ELAPSED_TIME`, `MONTH`, `DAY_OF_WEEK`, `SCHEDULED_HOUR`.
* **Calendar & Route Features (`ENG`, 8 features):** `ROUTE`, `AIRLINE_ROUTE`, `DAY_PART`, `IS_WEEKEND`, cyclical encodings (`TIME_SIN`, `TIME_COS`, `MONTH_SIN`, `MONTH_COS`, `DOW_SIN`, `DOW_COS`).
* **Causal Prior-Day Network State Features (`HIST`, 58 features):**
  * Computed strictly prior to departure date ($t < \text{FL\_DATE}$) to prevent same-day or future leakage.
  * Expanding and rolling windows: `1-day`, `7-day`, `30-day`, and `cumulative all-history`.
  * Entities tracked: Global network, Airline, Origin airport, Destination airport, Route, Origin $\times$ Departure Hour, Airline $\times$ Route.
  * Statistics computed: Historical disruption rate (`RATE_*`), historical mean delay (`MEAN_*`), historical flight volume (`N_*`).
  * Bayesian Shrinkage: Empirical Bayes smoothing ($m = 50$) pulls small-sample route rates toward the global network baseline.

---

## 5. Descriptive Statistics & Distributional Parameters

Computed on the full population of `2,920,859` completed commercial flights:

| Parameter | Departure Delay (`DEP_DELAY`) | Flight Distance (`DISTANCE`) | Scheduled Block (`CRS_ELAPSED_TIME`) | Departure Hour (`SCHEDULED_HOUR`) |
| :--- | :--- | :--- | :--- | :--- |
| **Unit** | Minutes | Statute Miles | Minutes | Hour of Day (0–23) |
| **Mean ($\mu$)** | **`+10.0957 min`** | `811.0855 miles` | `142.4576 min` | `12.9917 hr` |
| **Median ($Q_2$)** | **`-2.0000 min`** | `655.0000 miles` | `125.0000 min` | `13.0000 hr` |
| **Mode** | **`-5.0000 min`** | `337.0000 miles` | `90.0000 min` | `6.0000 hr` (6:00 AM) |
| **Standard Deviation ($\sigma$)** | **`49.1777 min`** | `589.5143 miles` | `71.7430 min` | `4.8310 hr` |
| **Variance ($\sigma^2$)** | `2,418.4419 min²` | `347,527.12 miles²` | `5,147.06 min²` | `23.3386 hr²` |
| **Interquartile Range (IQR)** | **`12.0000 min`** ($[-6.0, +6.0]$) | `668.0000 miles` | `83.0000 min` | `8.0000 hr` |
| **Median Absolute Deviation (MedAD)** | **`4.0000 min`** | `320.0000 miles` | `40.0000 min` | `4.0000 hr` |
| **Mean Absolute Deviation (MAD)** | `21.3519 min` | `445.3271 miles` | `54.2952 min` | `4.1398 hr` |
| **Skewness ($\gamma_1$)** | **`+11.4906`** (Severe Right Skew) | `+1.4968` | `+1.4312` | `+0.0956` |
| **Excess Kurtosis ($\gamma_2$)** | **`+243.9439`** (Leptokurtic Tail) | `+2.8332` | `+2.5474` | `-1.0304` |
| **Minimum / Maximum** | `[-90.0 min, +2,966.0 min]` | `[29.0, 5,812.0 miles]` | `[1.0, 705.0 min]` | `[0, 23]` |
| **Percentiles ($P_{90}, P_{95}, P_{99}$)**| `P90 = 36m, P95 = 72m, P99 = 191m` | `P90 = 1608m, P95 = 2139m` | `P90 = 240m, P95 = 299m` | `P90 = 20, P95 = 21` |
| **95% Confidence Interval for $\mu$** | `[10.0393, 10.1521] min` | `[810.4101, 811.7609]` | `[142.3754, 142.5399]` | `[12.9862, 12.9973]` |

---

## 6. Inferential Statistics & Hypothesis Testing Summary

All 14 hypothesis tests were evaluated with **Benjamini-Hochberg False Discovery Rate (FDR)** control ($\alpha = 0.05$):

| Test ID | Hypothesis Test | Test Statistic | Raw $p$-value | BH Adjusted $p$-value | Effect Size | Statistical Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HT-01** | One-Sample $t$-test ($\mu = 0$ min) | $t = 350.8533$ | $< 10^{-300}$ | $< 10^{-300}$ | Cohen's $d = 0.2053$ | **Reject $H_0$** |
| **HT-02** | One-Sample $t$-test ($\mu = 15$ min) | $t = -170.4359$ | $< 10^{-300}$ | $< 10^{-300}$ | Cohen's $d = -0.0997$ | **Reject $H_0$** |
| **HT-03** | One-Sample Wilcoxon Signed-Rank ($m=0$) | $W = 548,166,017.5$ | $8.80 \times 10^{-8}$ | $8.80 \times 10^{-8}$ | Rank-Biserial $r = -0.2871$ | **Reject $H_0$** |
| **HT-04** | Two-Sample $t$-test (Evening vs Morning) | $t = 147.9410$ | $< 10^{-300}$ | $< 10^{-300}$ | Cohen's $d = 0.2306$ | **Reject $H_0$** |
| **HT-05** | Welch's $t$-test (Weekend vs Weekday) | Welch $t = 13.2233$ | $6.48 \times 10^{-40}$ | $6.97 \times 10^{-40}$ | Cohen's $d = 0.0176$ | **Reject $H_0$** |
| **HT-06** | Mann-Whitney $U$ (Evening vs Morning) | $U = 3.822 \times 10^{9}$ | $< 10^{-300}$ | $< 10^{-300}$ | Rank-Biserial $r = 0.1943$ | **Reject $H_0$** |
| **HT-07** | Paired $t$-test ($N=1,704$ matched days) | Paired $t = 42.2460$ | $2.17 \times 10^{-267}$ | $2.53 \times 10^{-267}$ | Paired $d_z = 1.0234$ | **Reject $H_0$** |
| **HT-08** | Paired Wilcoxon ($N=1,704$ matched days) | $W = 56,315.0$ | $1.46 \times 10^{-238}$ | $1.86 \times 10^{-238}$ | Rank-Biserial $r = 0.7735$ | **Reject $H_0$** |
| **HT-09** | Chi-Square Test (Airline vs Delay, $df=17$) | $\chi^2 = 32,732.28$ | $< 10^{-300}$ | $< 10^{-300}$ | Cramér's $V = 0.1059$ | **Reject $H_0$** |
| **HT-10** | Chi-Square Test (Day Part vs Delay, $df=3$) | $\chi^2 = 72,233.58$ | $< 10^{-300}$ | $< 10^{-300}$ | Cramér's $V = 0.1573$ | **Reject $H_0$** |
| **HT-11** | Fisher's Exact Test (Eve/Morn $\times$ Delay) | Odds Ratio $= 3.6749$ | $< 10^{-300}$ | $< 10^{-300}$ | Odds Ratio $= 3.6749$ | **Reject $H_0$** |
| **HT-12** | Shapiro-Wilk Normality Test ($n=5,000$) | $W = 0.3595$ | $1.35 \times 10^{-85}$ | $1.35 \times 10^{-85}$ | $1 - W = 0.6405$ | **Reject $H_0$** |
| **HT-13** | D'Agostino-Pearson Omnibus $K^2$ | $K^2 = 4,940,303.01$ | $< 10^{-300}$ | $< 10^{-300}$ | Skewness $\gamma_1 = 11.49$ | **Reject $H_0$** |
| **HT-14** | Kolmogorov-Smirnov Test ($n=50,000$) | $D = 0.3052$ | $< 10^{-300}$ | $< 10^{-300}$ | KS Distance $D = 0.3052$ | **Reject $H_0$** |

### ANOVA Architecture

* **One-Way ANOVA (Operating Airline, $k=18$):** $F = 651.10$, Welch's $F = 730.82$, Daily Cluster-Robust $F = 106.64$ ($p < 10^{-15}$), $\eta^2 = 0.0038$.
* **One-Way ANOVA (Scheduled Day Part, $k=4$):** $F = 7,965.87$, Welch's $F = 7,833.27$, Daily Cluster-Robust $F = 416.91$ ($p < 10^{-15}$), $\eta^2 = 0.0081$.
* **One-Way ANOVA (Calendar Month, $k=12$):** $F = 896.39$, Welch's $F = 876.45$, Daily Cluster-Robust $F = 18.27$ ($p < 10^{-15}$), $\eta^2 = 0.0034$.
* **Two-Way ANOVA (`DAY_PART × IS_WEEKEND`):**
  * Day Part Main Effect: $F = 7,966.46$, $p < 10^{-300}$, Partial $\eta^2 = 0.00812$.
  * Weekend Main Effect: $F = 184.78$, $p = 4.3 \times 10^{-42}$, Partial $\eta^2 = 0.00006$.
  * Interaction Effect: $F = 11.74$, $p = 2.4 \times 10^{-7}$, Partial $\eta^2 = 0.00001$.

---

## 7. Selected Classification Models Performance

Selected based on **Validation Delayed-Class F1-score** with validation-optimized decision threshold ($\tau^*$):

| Rank | Model Architecture | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Balanced Accuracy | Validation F1 | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **XGBoost Classifier** | **68.28%** | **36.56%** | **59.27%** | **0.4522** | **0.7068** | **0.4106** | **65.05%** | **0.4510** | **Champion Classifier** |
| 2 | HistGradientBoosting Classifier | 67.98% | 36.36% | 59.89% | 0.4525 | 0.7055 | 0.4090 | 65.08% | 0.4484 | Candidate (Top 4) |
| 3 | LightGBM Classifier | 68.79% | 36.88% | 58.06% | 0.4511 | 0.7049 | 0.4095 | 64.94% | 0.4479 | Candidate (Top 4) |
| 4 | Logistic Regression | 66.62% | 35.40% | 61.92% | 0.4505 | 0.7028 | 0.4061 | 64.94% | 0.4472 | Candidate (Top 4) |

### Champion Classifier Confusion Matrix (632,306 Untouched Test Flights)
* **True Negatives (TN):** `348,925 flights` (Correctly predicted on-time).
* **False Positives (FP):** `143,688 flights` (False delay alarms).
* **False Negatives (FN):** `56,892 flights` (Missed disruptions).
* **True Positives (TP):** `82,801 flights` (Successfully alerted delays).
* **Specificity (TNR):** `70.83%`.
* **Matthews Correlation Coefficient (MCC):** `0.2605`.
* **Log-Loss:** `0.4802`.
* **Optimal Decision Threshold ($\tau^*$):** `0.2372`.

---

## 8. Selected Regression Models Performance

Selected based on **Validation Mean Absolute Error (MAE)** minimizing robust L1 loss:

| Rank | Model Architecture | MAE (min) | RMSE (min) | $R^2$ Score | Median AE (min) | Validation MAE | Validation RMSE | Explained Variance | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **LightGBM Regressor (L1)** | **18.5561** | **57.4162** | **-0.0295** | **5.1581** | **17.8043** | **53.2622** | **0.0245** | **Champion Regressor** |
| 2 | HistGradientBoosting Regressor (L1) | 18.5962 | 57.5214 | -0.0333 | 5.1490 | 17.8662 | 53.2781 | 0.0223 | Candidate (Top 4) |
| 3 | XGBoost Regressor (L1) | 18.6308 | 57.6149 | -0.0367 | 5.1655 | 17.8945 | 53.3533 | 0.0202 | Candidate (Top 4) |
| 4 | Gradient Boosting Regressor (L1) | 18.7290 | 57.6410 | -0.0376 | 5.3890 | 17.9840 | 53.5618 | 0.0183 | Candidate (Top 4) |

### Key Regression Observations
* **Benchmark vs. Training Mean Baseline:** Training mean baseline test MAE is `22.6540 minutes`. The champion LightGBM Regressor achieves an **`18.09%` error reduction**.
* **Median Absolute Error:** Achieving **`5.1581 minutes`** demonstrates that for over 50% of commercial flights, the model's prediction is within ~5 minutes of actual pushback.
* **$R^2$ Score Explanation:** $R^2$ evaluates squared deviations around the mean ($+10.10$ min). Because L1 loss estimates the conditional median ($-2.0$ min), it incurs a negative squared bias penalty against the extreme positive tail, which is mathematically expected under skewness $\gamma_1 = 11.49$.

