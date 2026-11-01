# Flight Delay Prediction and Statistical Intelligence System
## PowerPoint Visual Reference & Asset Mapping Guide

> **Purpose:** This reference document maps every slide in `PPT_Slide_Content.md` to specific visual assets, including existing high-resolution publication charts in `artifacts/ppt_charts/`, Graphviz diagrams, interactive tables, and live Streamlit UI views.

---

## 1. Visual Asset Directory & Inventory

The project maintains a dedicated set of publication-grade charts generated directly from the 2.92M flight population and saved in `artifacts/ppt_charts/`. These charts are styled with high-DPI resolution, clear labels, and professional light-theme aesthetics.

### Inventory of Verified Visual Files:

| Category | File Name | Description | Recommended PPT Slides |
| :--- | :--- | :--- | :--- |
| **Exploratory Data Analysis** | `eda_01_delay_distribution.png` | Histogram & KDE of departure delays showing severe right skewness | Slide 8 |
| | `eda_02_delay_boxplot.png` | Boxplot with Tukey fences demonstrating extreme delay outliers | Slide 5, Slide 8 |
| | `eda_03_airline_performance.png` | Mean departure delay and volume ranking across 18 US carriers | Slide 4, Slide 7 |
| | `eda_04_airport_performance.png` | Top origin airport delay rates and volume | Slide 4, Slide 7 |
| | `eda_05_route_performance.png` | Busiest and highest-delay route corridors | Slide 4 |
| | `eda_06_monthly_delay_trends.png` | Monthly delay trend across the 56-month observation window | Slide 4, Slide 7 |
| | `eda_07_yearly_delay_trends.png` | Annual flight volume and mean delay comparison (2019–2023) | Slide 4 |
| | `eda_08_departure_hour.png` | Diurnal delay accumulation by scheduled hour of departure | Slide 7 |
| | `eda_09_day_of_week.png` | Day-of-week flight performance (weekday vs. weekend) | Slide 7, Slide 10 |
| | `eda_10_correlation_heatmap.png` | Pearson and Spearman correlation matrix across key attributes | Slide 10 |
| | `eda_11_violin_and_ecdf.png` | Violin plots by day part and empirical cumulative distribution | Slide 8 |
| | `eda_12_qq_normality_plots.png` | Normal Q-Q plots demonstrating severe non-Gaussian tails | Slide 8 |
| | `eda_13_hypothesis_and_anova.png` | Hypothesis test effect sizes and ANOVA factor comparisons | Slide 9, Slide 10 |
| | `eda_14_corr_methods_and_vif.png` | Collinearity diagnostics and VIF breakdown | Slide 10 |
| | `eda_15_outlier_and_bootstrap_ci.png`| Bootstrap 95% confidence intervals and extreme tail audit | Slide 5, Slide 8 |
| **Classification Models** | `clf_01_class_distribution.png` | Genuine class distribution (82.3% on-time vs. 17.7% delayed) | Slide 5 |
| | `clf_02_accuracy_comparison.png` | Accuracy comparison bar chart across classifiers | Slide 13, Slide 16 |
| | `clf_03_precision_comparison.png`| Precision comparison bar chart across classifiers | Slide 13, Slide 16 |
| | `clf_04_recall_comparison.png` | Recall comparison bar chart across classifiers | Slide 13, Slide 16 |
| | `clf_05_f1_comparison.png` | Delayed-class F1 score comparison across classifiers | Slide 12, Slide 16 |
| | `clf_06_roc_curves.png` | ROC curves comparing top classifiers against chance line | Slide 13 |
| | `clf_07_pr_curves.png` | Precision-Recall curves comparing classifiers vs. baseline | Slide 13 |
| | `clf_08_confusion_matrices.png` | Confusion matrices for top classifiers on 632k test flights | Slide 13 |
| | `clf_09_feature_importance.png` | Feature gain and split importance for classification | Slide 6 |
| | `clf_10_threshold_analysis.png` | Validation decision threshold sweep (F1, Precision, Recall vs. $\tau$) | Slide 12 |
| | `clf_14_shap_summary.png` | Exact TreeSHAP feature attribution in log-odds units | Slide 6, Slide 12 |
| **Regression Models** | `reg_01_mae_comparison.png` | Mean Absolute Error (MAE) bar chart across regressors | Slide 15, Slide 16 |
| | `reg_02_rmse_comparison.png` | Root Mean Squared Error (RMSE) bar chart across regressors | Slide 15, Slide 16 |
| | `reg_03_r2_comparison.png` | $R^2$ score comparison across regressors | Slide 15, Slide 16 |
| | `reg_04_actual_vs_predicted.png`| Scatter plot of actual vs. predicted delay with 45° reference line | Slide 15 |
| | `reg_05_residual_distribution.png`| Residual distribution histogram ($y - \hat{y}$, clipped $\pm 60$ min) | Slide 15 |
| | `reg_06_residual_vs_predicted.png`| Residuals vs. predicted values checking homoscedasticity | Slide 15 |
| | `reg_07_error_distribution.png` | Cumulative absolute error distribution across models | Slide 15 |
| | `reg_08_performance_by_delay_range.png` | MAE stratified across delay bands ($\le 60$m, $\le 120$m, all) | Slide 15 |
| | `reg_09_feature_importance.png` | Feature gain importance for delay duration regression | Slide 6, Slide 14 |
| | `reg_10_baseline_vs_model.png` | Error reduction comparison vs. training mean baseline | Slide 15 |
| | `reg_12_residual_qq_and_shap.png` | Residual normal Q-Q plot and TreeSHAP impact in minutes | Slide 15 |

---

## 2. Slide-by-Slide Visual Recommendations

### Slide 1: Title Slide
* **Visual:** High-resolution aviation graphic or logo alongside the system subtitle.

### Slide 2: Problem Statement & Motivation
* **Visual:** Conceptual split graphic: Left side represents airport pushback delays; Right side shows a structured decision tree with the two target questions:
  1. $\text{Delayed } \ge 15\text{ min?} \rightarrow [\text{Yes / No}]$
  2. $\text{Delay Duration?} \rightarrow [\text{Minutes}]$

### Slide 3: Project Aim & Objectives
* **Visual:** Clean 6-box workflow process diagram representing the 6 sequential project objectives (Data Ingestion $\rightarrow$ Feature Engineering $\rightarrow$ Statistical Testing $\rightarrow$ Classification $\rightarrow$ Regression $\rightarrow$ Streamlit Dashboard).

### Slide 4: Dataset Description & Provenance
* **Visual 1:** `artifacts/ppt_charts/eda_05_route_performance.png` (Geographic flight network routes).
* **Visual 2:** Structured dataset dimension summary table:
  * 3,000,000 Raw Records | 2,920,859 Cleaned Records | 18 Airlines | 378 Airports | 7,785 Routes.

### Slide 5: Data Cleaning & Preprocessing Decisions
* **Visual 1:** `artifacts/ppt_charts/clf_01_class_distribution.png` (Showing the genuine 82.32% on-time vs. 17.68% delayed ratio).
* **Visual 2:** Flowchart of cleaning rules (Duplicates $\rightarrow$ Cancellations Excluded $\rightarrow$ Diverted Retained $\rightarrow$ Extremes Preserved $\rightarrow$ Zero SMOTE).

### Slide 6: Causal Feature Engineering Architecture
* **Visual 1:** Diagram showing the rolling time window calculation: Prior Day (1D), Prior Week (7D), Prior Month (30D) strictly before flight date ($t < \text{FL\_DATE}$).
* **Visual 2:** `artifacts/ppt_charts/clf_09_feature_importance.png` (Highlighting that prior-day historical rates provide the highest feature importance).

### Slide 7: Exploratory Data Analysis & Flight Delay Dynamics
* **Visual 1:** `artifacts/ppt_charts/eda_08_departure_hour.png` (Bar chart illustrating delay rate climbing monotonically from 10.8% at 6:00 AM to 26.2% at 7:00 PM).
* **Visual 2:** `artifacts/ppt_charts/eda_03_airline_performance.png` (Horizontal bar chart showing Hawaiian at 4.1 min vs. JetBlue at 16.5 min).

### Slide 8: Descriptive Statistics & Distributional Parameters
* **Visual 1:** `artifacts/ppt_charts/eda_01_delay_distribution.png` (Histogram & KDE of departure delays showing extreme right skewness).
* **Visual 2:** `artifacts/ppt_charts/eda_12_qq_normality_plots.png` (Normal Q-Q plot showing dramatic deviation from the normal reference line in the positive tail).

### Slide 9: Inferential Statistics: 14 Hypothesis Tests
* **Visual:** `artifacts/ppt_charts/eda_13_hypothesis_and_anova.png` (Bar plot of effect sizes across all 14 tests, highlighting Cohen's $d_z = 1.023$ for paired diurnal delay and $\text{OR} = 3.67$ for evening disruption odds).

### Slide 10: ANOVA Suite & Multicollinearity (VIF)
* **Visual 1:** Summary table of One-Way and Two-Way ANOVA with Daily Cluster-Robust $F$-statistics.
* **Visual 2:** `artifacts/ppt_charts/eda_14_corr_methods_and_vif.png` (Bar chart illustrating VIF of 30.0 for `DISTANCE` / `CRS_ELAPSED_TIME` collapsing to 1.029).

### Slide 11: System Architecture & Chronological Validation
* **Visual:** The Graphviz system architecture diagram embedded in `pages/6_About_Project.py`:
  $$\text{flight\_data.csv} \longrightarrow \text{Clean Population} \longrightarrow \text{Causal Engine} \longrightarrow \begin{cases} \text{Fit Partition (1.78M)} \\ \text{Validation Partition (504k)} \\ \text{Test Partition (632k)} \end{cases}$$

### Slide 12: Classification Methodology & Algorithm Selection
* **Visual:** `artifacts/ppt_charts/clf_10_threshold_analysis.png` (Curves showing how tuning threshold $\tau^*$ from 0.50 down to 0.2372 maximizes delayed-class F1-score).

### Slide 13: Classification Results & Champion Performance
* **Visual 1:** `artifacts/ppt_charts/clf_06_roc_curves.png` (ROC curves highlighting champion XGBoost with $\text{AUC} = 0.7068$).
* **Visual 2:** `artifacts/ppt_charts/clf_08_confusion_matrices.png` (Champion confusion matrix displaying 82,801 detected disruptions on the test set).

### Slide 14: Regression Methodology & Robust L1 Formulation
* **Visual:** `artifacts/ppt_charts/reg_09_feature_importance.png` (Feature gain importance chart for regression duration modeling).

### Slide 15: Regression Results & Residual Diagnostics
* **Visual 1:** `artifacts/ppt_charts/reg_04_actual_vs_predicted.png` (Actual vs. predicted delay scatter with 45° reference line).
* **Visual 2:** `artifacts/ppt_charts/reg_05_residual_distribution.png` (Histogram of residual errors centered at zero).
* **Visual 3:** `artifacts/ppt_charts/reg_08_performance_by_delay_range.png` (Error breakdown across delay bands).

### Slide 16: Model Comparison & Experiment Ledger
* **Visual 1:** `artifacts/ppt_charts/clf_05_f1_comparison.png` (Bar chart comparing the 4 classification models).
* **Visual 2:** `artifacts/ppt_charts/reg_01_mae_comparison.png` (Bar chart comparing the 4 regression models).

### Slide 17: Interactive Streamlit Platform Demonstration
* **Visual:** High-resolution screenshots of:
  * **Module 1:** Dashboard Overview KPI cards and monthly trends.
  * **Module 3:** Interactive Classification Disruption Simulator.
  * **Module 4:** Interactive Delay Duration Estimator.

### Slide 18: Conclusion, Limitations & Future Scope
* **Visual:** Future Architecture Roadmap flowchart integrating real-time NOAA METAR weather feeds, physical aircraft tail-number graphs, and conformal prediction intervals.

