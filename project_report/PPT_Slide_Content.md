# Flight Delay Prediction and Statistical Intelligence System
## PowerPoint Presentation Slide-by-Slide Outline (18 Slides)

> **Instructions for Presenter:** This document provides the exact text, key metrics, recommended visual figures, and speaker explanation for an academic project review or final-year capstone presentation.

---

### Slide 1: Title Slide
* **Slide Title:** Flight Delay Prediction and Statistical Intelligence System
* **Subtitle:** Pre-Departure Disruption Alerting and Duration Estimation Across 3,000,000 US Commercial Flights
* **Key Bullets:**
  * End-to-End Aviation Data Science, Statistical Testing & Machine Learning System
  * Strict Zero-Leakage & Causal Pre-Departure Feature Engineering
  * Production Multi-Page Streamlit Intelligence Platform
* **Presenter Explanation:**  
  "Good morning. Today we present our project, the Flight Delay Prediction and Statistical Intelligence System. Our project models commercial aviation disruptions using genuine pre-departure information across 3 million flights, providing actionable intelligence for airlines and passengers."

---

### Slide 2: Problem Statement & Motivation
* **Slide Title:** Problem Statement: The Aviation Disruption Challenge
* **Key Bullets:**
  * **Operational Impact:** Flight delays cascade across airframes, flight crews, gate availability, and connecting passengers.
  * **Critical Flaws in Prior Work:** Many prior models suffer from data leakage (using post-departure data) and synthetic distortion (SMOTE).
  * **Core Dispatch Need at $T - 2$ Hours:**
    1. *Binary Alert:* Will the flight be delayed by 15+ minutes (FAA standard)?
    2. *Continuous Estimation:* Exactly how many minutes late will departure occur?
* **Key Numerical Context:**
  * 17.68% of commercial flights experience delays $\ge 15$ min.
  * 64.86% of flights depart early or on time.
* **Recommended Visual:** Split-screen graphic showing airport congestion on the left and the dual task workflow (Classification + Regression) on the right.
* **Presenter Explanation:**  
  "Airlines must make proactive decisions hours before pushback. However, existing academic studies often commit data leakage by using departure or arrival timestamps as features. Our system solves this by using only information genuinely available before departure."

---

### Slide 3: Project Aim & Objectives
* **Slide Title:** Project Aim & Systematic Objectives
* **Key Bullets:**
  * **Aim:** Build a statistically rigorous, leakage-free machine learning intelligence system.
  * **Objective 1:** Clean and govern 3,000,000 BTS flight records (2019–2023), preserving real operational extremes.
  * **Objective 2:** Engineer 74 pre-departure features, including 58 causal prior-day historical metrics.
  * **Objective 3:** Implement 14 formal hypothesis tests (BH-FDR controlled) and full ANOVA suite.
  * **Objective 4:** Train and evaluate top 4 classification algorithms on natural class distribution.
  * **Objective 5:** Train and evaluate top 4 regression algorithms optimizing robust L1 loss.
  * **Objective 6:** Deploy an interactive 6-page Streamlit dashboard.
* **Presenter Explanation:**  
  "We structured our project around six clear objectives: strict data governance, zero-leakage causal feature engineering, deep statistical hypothesis testing, targeted model selection, and an interactive operational dashboard."

---

### Slide 4: Dataset Description & Provenance
* **Slide Title:** Dataset Description: USDOT / BTS TranStats
* **Key Bullets:**
  * **Data Source:** US Bureau of Transportation Statistics (BTS) Airline On-Time Performance database.
  * **Dataset Size:** 3,000,000 raw rows $\times$ 32 columns (`614.1 MB`).
  * **Time Period:** January 1, 2019 – August 31, 2023 (56 consecutive calendar months).
  * **Geographic & Network Scale:**
    * 18 Operating Commercial Airlines
    * 378 Commercial Origin Airports
    * 7,785 Unique Route Corridors
* **Key Numerical Results:**
  * Raw Rows: `3,000,000` | Cleaned Non-Cancelled Flights: `2,920,859`
* **Recommended Visual:** Table of dataset attributes and route map from `artifacts/ppt_charts/eda_05_route_performance.png`.
* **Presenter Explanation:**  
  "Our dataset covers 56 continuous months of commercial US flight data from the Bureau of Transportation Statistics. After excluding 79,140 cancelled flights where departure never occurred, we have a clean population of 2.92 million flights across 18 carriers."

---

### Slide 5: Data Cleaning & Preprocessing Decisions
* **Slide Title:** Data Governance & Zero-Leakage Policy
* **Key Bullets:**
  * **Exact Duplicates:** 72 duplicate records (`0.002%`) identified and removed.
  * **Cancellation Handling:** 79,140 cancelled flights excluded; 8,261 diverted flights retained.
  * **Preservation of Extreme Delays:** Delays up to `+2,966 minutes` (~49.4 hours) kept intact to preserve black-swan operational realities.
  * **Missing Delay Reasons (`NaN` Policy):** Delay cause columns (`DELAY_DUE_CARRIER`, etc.) are structurally missing on 81.9% of rows; retained as `NaN` and excluded from model inputs.
  * **No Synthetic Class Balancing:** Natural operational class ratio (`4.66 : 1`) preserved. No SMOTE or synthetic sample distortion.
* **Key Numerical Results:**
  * Target Class Balance: `82.32%` On-Time ($< 15$ min) vs. `17.68%` Delayed ($\ge 15$ min).
* **Recommended Visual:** `artifacts/ppt_charts/clf_01_class_distribution.png` (Natural class distribution).
* **Presenter Explanation:**  
  "We strictly avoid synthetic data oversampling. In the real world, about 18% of flights are delayed. Oversampling with SMOTE alters this base rate and causes high false-alarm rates. We also preserve genuine extreme delays up to 2,966 minutes."

---

### Slide 6: Causal Feature Engineering Architecture
* **Slide Title:** Feature Engineering: Causal Network State (`74 Features`)
* **Key Bullets:**
  * **Core Timetable Features (`BASE`, 8 features):** Scheduled hour, day of week, month, distance, block duration.
  * **Cyclical Temporal Encodings (`ENG`, 8 features):** Sine/cosine transformations for departure hour, month, and day of week.
  * **Causal Prior-Day Network Features (`HIST`, 58 features):**
    * Rolling 1-day, 7-day, 30-day, and cumulative all-history statistics.
    * Computed strictly prior to departure date ($t < \text{FL\_DATE}$).
    * Entities: Global network, Airline, Origin, Destination, Route, Origin $\times$ Hour.
    * **Empirical Bayes Shrinkage:** Smooths low-sample route delay rates toward the global network mean ($m = 50$).
* **Recommended Visual:** Diagram of rolling prior-day historical aggregation windows.
* **Presenter Explanation:**  
  "To give our models predictive power without data leakage, we engineered 58 causal historical features. These calculate the delay rates of airlines, airports, and routes strictly over prior days, using Empirical Bayes smoothing to prevent small-sample noise."

---

### Slide 7: Exploratory Data Analysis & Flight Delay Dynamics
* **Slide Title:** Exploratory Data Analysis: Key Delay Patterns
* **Key Bullets:**
  * **Early Departure Baseline:** 64.86% of flights depart on time or early ($\le 0$ min); modal departure is $-5.0$ minutes.
  * **Diurnal Delay Accumulation:**
    * Morning (05:00–11:59): Delay rate = `10.82%`, mean delay = `+4.18 min`.
    * Evening (17:00–20:59): Delay rate = `26.24%`, mean delay = `+17.82 min`.
  * **Carrier Heterogeneity:** Average delay ranges from `+4.11 min` (Hawaiian) to `+16.48 min` (JetBlue).
* **Key Numerical Results:**
  * Median Delay: `-2.0 min` | Mean Delay: `+10.10 min` | Std Dev: `49.18 min`.
* **Recommended Visual:** `artifacts/ppt_charts/eda_08_departure_hour.png` and `artifacts/ppt_charts/eda_03_airline_performance.png`.
* **Presenter Explanation:**  
  "EDA reveals that delays are not uniformly distributed. Flights departing in the morning experience only a 10.8% delay rate, while evening flights experience a 26.2% delay rate due to rotational delays compounding throughout the day."

---

### Slide 8: Descriptive Statistics & Distributional Parameters
* **Slide Title:** Statistical Intelligence: The Non-Gaussian Nature of Delays
* **Key Bullets:**
  * **Extreme Asymmetry:** Mean is `+10.10 min` while median is `-2.0 min` (a 12-minute disparity).
  * **Skewness ($\gamma_1 = +11.49$):** Massive positive skewness confirms heavy right tail.
  * **Excess Kurtosis ($\gamma_2 = +243.94$):** Extreme leptokurtic distribution with fat tails.
  * **Normality Rejection:** Shapiro-Wilk ($W = 0.3595$, $p < 10^{-85}$) and D'Agostino-Pearson ($K^2 = 4.94 \times 10^6$) conclusively reject normality.
* **Key Numerical Results:**
  * $P_{50} = -2.0$ min, $P_{90} = 36.0$ min, $P_{95} = 72.0$ min, $P_{99} = 191.0$ min, $\text{Max} = 2,966.0$ min.
* **Recommended Visual:** `artifacts/ppt_charts/eda_01_delay_distribution.png` and `artifacts/ppt_charts/eda_12_qq_normality_plots.png`.
* **Presenter Explanation:**  
  "Standard Gaussian assumptions fail completely on flight delays. With a skewness of 11.49 and excess kurtosis of 243.94, delays are dominated by a massive right tail. This statistical finding directly drives our choice of regression loss function."

---

### Slide 9: Inferential Statistics: 14 Hypothesis Tests
* **Slide Title:** Formal Hypothesis Testing (Benjamini-Hochberg FDR Controlled)
* **Key Bullets:**
  * 14 formal two-tailed hypothesis tests evaluated across 2.92M records.
  * All $p$-values adjusted using **Benjamini-Hochberg FDR** ($\alpha = 0.05$).
  * **Key Test Highlights:**
    * **HT-04 ($t$-test):** Evening vs Morning delay difference is highly significant ($t = 147.94$, $p < 10^{-300}$, $d = 0.231$).
    * **HT-07 (Paired $t$-test on 1,704 Matched Days):** Controls for day-to-day weather; evening delay exceeds morning by $+13.62$ min ($t = 42.25$, $d_z = 1.023$, large effect).
    * **HT-09 & HT-10 ($\chi^2$ Tests):** Disruption probability is highly dependent on carrier ($\chi^2 = 32,732$) and day part ($\chi^2 = 72,234$).
    * **HT-11 (Fisher's Exact Test):** Evening flights have **$3.67\times$ higher odds** of being delayed $\ge 15$ min compared to morning flights.
* **Recommended Visual:** `artifacts/ppt_charts/eda_13_hypothesis_and_anova.png`.
* **Presenter Explanation:**  
  "We performed 14 hypothesis tests with Benjamini-Hochberg correction. The paired test across 1,704 matched calendar days confirms that evening flights suffer an extra 13.6 minutes of delay, with evening flights having 3.67 times higher odds of disruption."

---

### Slide 10: ANOVA Suite & Multicollinearity (VIF)
* **Slide Title:** ANOVA Suite & Collinearity Diagnostics
* **Key Bullets:**
  * **Cluster-Robust Block-Mean ANOVA:** Flight clustering within calendar dates was corrected via daily cluster-robust ANOVA; all factors remain significant ($p < 10^{-15}$).
    * Day Part Factor ($k=4$): $F = 7,965.87$, Cluster-Robust $F = 416.91$, $\eta^2 = 0.0081$.
    * Carrier Factor ($k=18$): $F = 651.10$, Cluster-Robust $F = 106.64$, $\eta^2 = 0.0038$.
  * **Two-Way ANOVA (`DAY_PART × IS_WEEKEND`):** Day-part main effect dominates ($F = 7,966.46$), while weekend effect is secondary ($F = 184.78$).
  * **Variance Inflation Factor (VIF):**
    * Severe collinearity between `CRS_ELAPSED_TIME` ($\text{VIF} = 30.03$) and `DISTANCE` ($\text{VIF} = 29.99$).
    * Removing elapsed block duration collapses distance VIF to $1.029$.
* **Recommended Visual:** `artifacts/ppt_charts/eda_14_corr_methods_and_vif.png`.
* **Presenter Explanation:**  
  "Because flights on the same day share common airport weather, standard ANOVA $F$-values can be inflated. We implemented daily cluster-robust block-mean ANOVA, proving carrier and day-part effects remain rock-solid."

---

### Slide 11: System Architecture & Chronological Validation
* **Slide Title:** System Architecture & Chronological Partitioning
* **Key Bullets:**
  * **Strict Temporal Splitting (No Future Leakage):**
    * **Fit Partition (2019–2021):** `1,784,112` flights ($61.1\%$) – 3-fold `TimeSeriesSplit` tuning.
    * **Validation Partition (2022):** `504,441` flights ($17.3\%$) – Model selection & threshold tuning ($\tau^*$).
    * **Out-of-Time Test Partition (Sep 2022 – Aug 2023):** `632,306` flights ($21.6\%$) – Single final evaluation.
  * **Modular Python Core:** Data layer (`ui_data.py`), feature pipeline (`history_features.py`), model evaluators (`model_evaluation.py`), and multi-page Streamlit app.
* **Recommended Visual:** System Architecture Graphviz flow diagram from `pages/6_About_Project.py`.
* **Presenter Explanation:**  
  "Our system architecture enforces chronological integrity. Models are trained on data up to 2021, validated on early 2022, and evaluated once on an untouched test set of 632,306 flights from late 2022 to mid-2023."

---

### Slide 12: Classification Methodology & Algorithm Selection
* **Slide Title:** Classification Methodology: Predicting Delays $\ge 15$ Minutes
* **Key Bullets:**
  * **Objective:** Predict binary disruption flag ($\text{DELAYED} = 1$).
  * **Selection Protocol:** Evaluated on the validation partition using **Delayed-Class F1-Score** as primary metric, supported by Recall and PR-AUC.
  * **Top 4 Selected Algorithms:**
    1. **XGBoost Classifier (Champion):** Validation F1 = `0.4510`, Recall = `59.16%`, $\tau^* = 0.2372$.
    2. **HistGradientBoosting Classifier:** Validation F1 = `0.4484`, Recall = `62.30%`, $\tau^* = 0.2330$.
    3. **LightGBM Classifier:** Validation F1 = `0.4479`, Recall = `58.26%`, $\tau^* = 0.2361$.
    4. **Logistic Regression:** Validation F1 = `0.4472`, Recall = `60.65%`, $\tau^* = 0.5868$.
* **Presenter Explanation:**  
  "We selected the top four classifiers based strictly on validation delayed-class F1 performance. XGBoost proved strongest, providing the best trade-off between disruption recall and precision."

---

### Slide 13: Classification Results & Champion Performance
* **Slide Title:** Classification Evaluation on Untouched Out-of-Time Test Set
* **Key Numerical Results (632,306 Test Flights):**
  * **XGBoost Classifier (Champion):** F1 = **`0.4522`** | ROC-AUC = **`0.7068`** | Recall = **`59.27%`** | Precision = **`36.56%`** | Accuracy = **`68.28%`**.
  * **HistGradientBoosting:** F1 = `0.4525` | ROC-AUC = `0.7055` | Recall = `59.89%` | Precision = `36.36%`.
  * **LightGBM:** F1 = `0.4511` | ROC-AUC = `0.7049` | Recall = `58.06%` | Precision = `36.88%`.
  * **Logistic Regression:** F1 = `0.4505` | ROC-AUC = `0.7028` | Recall = `61.92%` | Precision = `35.40%`.
* **Champion Confusion Matrix:**
  * True Negatives: `348,925` | False Positives: `143,688`
  * False Negatives: `56,892` | True Positives: `82,801` (Disruptions Alerted)
* **Recommended Visual:** `artifacts/ppt_charts/clf_06_roc_curves.png` and `artifacts/ppt_charts/clf_08_confusion_matrices.png`.
* **Presenter Explanation:**  
  "On 632,306 test flights, our champion XGBoost classifier achieved an F1 of 0.4522 and ROC-AUC of 0.7068, correctly flagging 82,801 delayed flights before departure with a recall of 59.3%."

---

### Slide 14: Regression Methodology & Robust L1 Formulation
* **Slide Title:** Regression Methodology: Departure Delay Duration Estimation
* **Key Bullets:**
  * **Objective:** Predict continuous departure delay in minutes (`DEP_DELAY`).
  * **The Skewness Problem:** Delays range from $-90$ to $+2,966$ minutes with skewness $11.49$.
  * **L1 Loss Formulation:** Minimizing absolute error ($\text{MAE}$) targets the **conditional median** ($-2.0$ min), avoiding distortions caused by squared loss on extreme outliers.
  * **Selection Protocol:** Models selected based on lowest **Validation MAE**.
  * **Top 4 Selected Algorithms:**
    1. **LightGBM Regressor (L1, Champion):** Validation MAE = `17.8043 min`.
    2. **HistGradientBoosting Regressor (L1):** Validation MAE = `17.8662 min`.
    3. **XGBoost Regressor (L1):** Validation MAE = `17.8945 min`.
    4. **Gradient Boosting Regressor (L1):** Validation MAE = `17.9840 min`.
* **Presenter Explanation:**  
  "For continuous delay duration, we formulated our models with L1 loss to predict median delay. This protects the models from being destabilized by multi-day blizzard delays."

---

### Slide 15: Regression Results & Residual Diagnostics
* **Slide Title:** Regression Evaluation: Error Reduction & Residuals
* **Key Numerical Results (632,306 Test Flights):**
  * **LightGBM Regressor (Champion):** MAE = **`18.5561 min`** | Median AE = **`5.1581 min`** | RMSE = `57.4162 min`.
  * **HistGradientBoosting:** MAE = `18.5962 min` | Median AE = `5.1490 min` | RMSE = `57.5214 min`.
  * **XGBoost Regressor:** MAE = `18.6308 min` | Median AE = `5.1655 min` | RMSE = `57.6149 min`.
  * **Gradient Boosting:** MAE = `18.7290 min` | Median AE = `5.3890 min` | RMSE = `57.6410 min`.
* **Key Performance Findings:**
  * **-18.09% MAE Reduction** compared to training mean baseline (`22.65 min`).
  * **Median Absolute Error of `5.16 minutes`:** Half of all predictions are within ~5 minutes of actual departure.
* **Recommended Visual:** `artifacts/ppt_charts/reg_04_actual_vs_predicted.png` and `artifacts/ppt_charts/reg_05_residual_distribution.png`.
* **Presenter Explanation:**  
  "Our champion LightGBM regressor achieved an 18.1% error reduction over the mean baseline. Even more impressively, its median absolute error is just 5.16 minutes, providing highly accurate predictions for typical commercial flights."

---

### Slide 16: Model Comparison & Experiment Ledger
* **Slide Title:** Side-by-Side Model Comparison Ledger
* **Key Bullets:**
  * **Classification Comparison Table:**

    | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
    | :--- | :---: | :---: | :---: | :---: | :---: |
    | **XGBoost Classifier** | **68.28%** | **36.56%** | **59.27%** | **0.4522** | **0.7068** |
    | HistGradientBoosting | 67.98% | 36.36% | 59.89% | 0.4525 | 0.7055 |
    | LightGBM Classifier | 68.79% | 36.88% | 58.06% | 0.4511 | 0.7049 |
    | Logistic Regression | 66.62% | 35.40% | 61.92% | 0.4505 | 0.7028 |

  * **Regression Comparison Table:**

    | Model | MAE (min) | RMSE (min) | $R^2$ Score | Median AE (min) |
    | :--- | :---: | :---: | :---: | :---: |
    | **LightGBM Regressor** | **18.5561** | **57.4162** | **-0.0295** | **5.1581** |
    | HistGradientBoosting | 18.5962 | 57.5214 | -0.0333 | 5.1490 |
    | XGBoost Regressor | 18.6308 | 57.6149 | -0.0367 | 5.1655 |
    | Gradient Boosting | 18.7290 | 57.6410 | -0.0376 | 5.3890 |

* **Recommended Visual:** Bar charts from `artifacts/ppt_charts/clf_05_f1_comparison.png` and `artifacts/ppt_charts/reg_01_mae_comparison.png`.
* **Presenter Explanation:**  
  "Here is our complete side-by-side comparison. For classification, XGBoost leads in ROC-AUC and balanced F1. For regression, LightGBM leads with an MAE of 18.56 minutes and Median AE of 5.16 minutes."

---

### Slide 17: Interactive Streamlit Platform Demonstration
* **Slide Title:** Interactive Streamlit Dashboard Demonstration
* **Key Bullets:**
  * **6 Modules Deployed:**
    1. *Dashboard Overview:* Real-time traffic, delay distributions, carrier rankings.
    2. *Statistical Analysis & EDA:* 8 interactive tabs for deep inferential statistics.
    3. *Classification Analysis:* Confusion matrix, probability distribution, simulator.
    4. *Regression Analysis:* Actual vs. predicted scatter, residual Q-Q, delay duration estimator.
    5. *Model Comparison:* Benchmarks across all 8 models, experiment tracking ledger.
    6. *About Project:* System architecture, causal principles, technical documentation.
  * **Live Simulator:** Accepts airline, origin, destination, time, and distance to output calibrated delay probability and estimated minutes.
* **Recommended Visual:** Screenshots of the Streamlit overview page and simulator.
* **Presenter Explanation:**  
  "All analytical and predictive components are hosted in a 6-page interactive Streamlit dashboard. Operations personnel can select any flight route and immediately receive calibrated disruption alerts and minute estimates."

---

### Slide 18: Conclusion, Limitations & Future Scope
* **Slide Title:** Conclusion & Future Operational Enhancements
* **Key Bullets:**
  * **Key Achievements:**
    * Cleaned and analyzed 2.92 million flights without data leakage or synthetic distortion.
    * 14 hypothesis tests with FDR control proved carrier and diurnal significance.
    * Top 4 classifiers and top 4 regressors selected through chronological validation.
  * **Current Limitations:** Unobserved intra-day micro-weather and sudden airport ground stops.
  * **Future Operational Roadmap:**
    * Real-time NOAA METAR surface weather telemetry at $T - 2$ hours.
    * Physical aircraft tail-number (`TAIL_NUM`) rotation graphs to track incoming inbound legs.
    * Calibrated 90% conformal prediction intervals for gate buffer risk management.
* **Presenter Explanation:**  
  "In conclusion, our project demonstrates that leakage-free, causal historical machine learning provides robust pre-departure disruption forecasting. In future work, integrating real-time radar and aircraft tail-rotation tracking will push predictive accuracy even higher. Thank you, and we welcome your questions."

