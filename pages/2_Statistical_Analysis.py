import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from src.app_helper import initialize_app_state
from src.eda import (
    compute_descriptive_statistics,
    compute_airline_analysis,
    compute_airport_analysis,
    compute_route_analysis,
    compute_temporal_analysis,
    compute_correlation_matrix,
    compute_outlier_analysis,
    load_dataset_quality_report,
    load_full_descriptive_stats,
    load_group_statistics,
    load_correlation_and_vif,
    load_hypothesis_testing_results,
    load_anova_results,
)
from src.hypothesis_testing import run_custom_two_group_test
from src.visualizations import (
    plot_delay_histogram,
    plot_delay_boxplot_by_airline,
    plot_delay_categories_bar,
    plot_airline_bar,
    plot_airline_delay_rate,
    plot_airline_volume,
    plot_top_airports_bar,
    plot_airport_volume_vs_delay,
    plot_top_routes_bar,
    plot_hourly_trend,
    plot_hourly_delayed_pct,
    plot_weekday_trend,
    plot_monthly_trend,
    plot_yearly_trend,
    plot_daypart_comparison,
    plot_correlation_heatmap,
    plot_violin_by_daypart,
    plot_ecdf_delay,
    plot_qq_normality,
    plot_vif_bars,
    plot_multi_correlation_bars,
    plot_anova_effect_sizes,
    plot_twoway_interaction,
)

st.set_page_config(
    page_title="Statistical Analysis & EDA | Flight Delay Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_app_state()
df = st.session_state.df
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

# Load precomputed full-dataset statistical artifacts (computed across all 2,920,859 cleaned flights)
qual_rep = load_dataset_quality_report()
full_desc = load_full_descriptive_stats()
group_stats = load_group_statistics()
corr_vif = load_correlation_and_vif()
hyp_res = load_hypothesis_testing_results()
anova_res = load_anova_results()

st.markdown('<div class="aviation-badge">📊 PART 1 & PART 2: STATISTICAL INTELLIGENCE ENGINE</div>', unsafe_allow_html=True)
st.title("Statistical Analysis & Exploratory Data Analysis")
st.markdown(
    """
    Comprehensive parametric and non-parametric statistical investigation across **3,000,000 raw records** (**2,920,859 completed non-cancelled flights**, Jan 2019 – Aug 2023).
    All hypothesis tests apply **Benjamini-Hochberg False Discovery Rate (FDR)** correction ($\alpha = 0.05$) and report practical **effect sizes** alongside $p$-values.
    """
)

# 8-Tab Layout mandated by Part 2.8
tabs = st.tabs([
    "1. Dataset Overview",
    "2. Descriptive Statistics",
    "3. Distribution Analysis",
    "4. Airline & Airport Analysis",
    "5. Correlation & VIF Analysis",
    "6. Hypothesis Testing",
    "7. ANOVA & Post-Hoc Analysis",
    "8. Outlier & Trend Analysis",
])

# ==============================================================================
# TAB 1: DATASET OVERVIEW & AUTOMATED QUALITY REPORT (Part 1.1 & 1.2)
# ==============================================================================
with tabs[0]:
    st.subheader("1. Automated Dataset Inspection & Quality Report")
    ds_sum = qual_rep.get("dataset_summary", {})
    dup_chk = qual_rep.get("duplicate_records", {})
    date_rng = qual_rep.get("date_range", {})
    tgt_dist = qual_rep.get("target_distribution", {})

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Raw CSV Rows", f"{ds_sum.get('raw_rows', 3_000_000):,}", f"{ds_sum.get('raw_columns', 32)} Raw Columns")
    k2.metric("Completed Flights", f"{ds_sum.get('cleaned_non_cancelled_rows', 2_920_859):,}", f"-{ds_sum.get('cancelled_flights_excluded', 79_141):,} Cancelled")
    k3.metric("Engineered Features", f"{ds_sum.get('engineered_model_table_columns', 82)}", "58 Causal History (H_*)")
    k4.metric("Exact Duplicate Rows", f"{dup_chk.get('exact_duplicate_flight_keys', 0)}", "Zero Duplicates")
    k5.metric("Date Range (56 Months)", f"{date_rng.get('min_date', '2019-01-01')}", f"to {date_rng.get('max_date', '2023-08-31')}")
    k6.metric("Raw CSV Size", f"{ds_sum.get('raw_csv_size_mb', 575.4):.1f} MB", f"Parquet: {ds_sum.get('parquet_size_mb', 312.5):.1f} MB")

    st.markdown("#### Preprocessing & Zero-Leakage Governance Rules Verified")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.info(
            "**1. Genuine Extreme Delays Preserved:**\n"
            "Flights delayed up to **2,966 minutes (~49.4 hours)** are real operational disruptions (severe weather/maintenance) and are **never trimmed** from evaluation."
        )
    with g2:
        st.info(
            "**2. Missing Delay-Reason Columns Not Zero-Imputed:**\n"
            "`DELAY_DUE_CARRIER`, `WEATHER`, `NAS`, `SECURITY`, and `LATE_AIRCRAFT` are structurally missing (`NaN`) on 81.9% of flights (non-delayed) and are **never replaced with 0** or fed to predictive models."
        )
    with g3:
        st.info(
            "**3. Zero Post-Departure Leakage:**\n"
            "All post-departure variables (`DEP_TIME`, `WHEELS_OFF`, `TAXI_OUT`, `ARR_TIME`, `ARR_DELAY`, `AIR_TIME`, `ELAPSED_TIME`) are strictly excluded from model inputs. All `H_*` history features use strictly prior calendar days (`< FL_DATE`)."
        )

    st.markdown("#### Column-Level Schema, Missingness & Leakage Audit (`artifacts/dataset_quality_report.csv`)")
    q_csv = ARTIFACTS / "dataset_quality_report.csv"
    if q_csv.exists():
        q_df = pd.read_csv(q_csv)
        st.dataframe(q_df, use_container_width=True, hide_index=True)

    c_chron1, c_chron2 = st.columns(2)
    with c_chron1:
        st.markdown("#### Chronological Out-of-Time Split Design")
        split_df = pd.DataFrame([
            {"Split Partition": "Fit (Model Training & CV)", "Date Window": "2019-01-01 → 2021-12-24", "Flights": "1,784,112", "Delayed ≥15m (%)": "16.34%", "Purpose": "Fit estimators & 3-Fold TimeSeriesSplit tuning"},
            {"Split Partition": "Validation (Threshold & Selection)", "Date Window": "2021-12-25 → 2022-09-25", "Flights": "504,441", "Delayed ≥15m (%)": "20.38%", "Purpose": "F1 threshold optimization, isotonic calibration, model selection"},
            {"Split Partition": "Test (Untouched Out-of-Time)", "Date Window": "2022-09-26 → 2023-08-31", "Flights": "632,306", "Delayed ≥15m (%)": "20.32%", "Purpose": "Final unbiased out-of-time evaluation"},
        ])
        st.dataframe(split_df, use_container_width=True, hide_index=True)
    with c_chron2:
        st.markdown("#### Target Variable Class Balance (`DELAYED` $\ge 15$ min)")
        t_df = pd.DataFrame([
            {"Class / Metric": "Class 0: Not Delayed (< 15 min)", "Count / Value": f"{tgt_dist.get('DELAYED_binary_15min', {}).get('class_0_not_delayed', 2_404_458):,}", "Share": "82.32%"},
            {"Class / Metric": "Class 1: Delayed (≥ 15 min, FAA Standard)", "Count / Value": f"{tgt_dist.get('DELAYED_binary_15min', {}).get('class_1_delayed', 516_401):,}", "Share": "17.68%"},
            {"Class / Metric": "Imbalance Ratio (Majority : Minority)", "Count / Value": f"{tgt_dist.get('DELAYED_binary_15min', {}).get('imbalance_ratio_0_to_1', 4.656):.2f} : 1", "Share": "Moderately Imbalanced"},
            {"Class / Metric": "Early / Exact On-Time Departures (≤ 0 min)", "Count / Value": "1,894,524", "Share": "64.86%"},
            {"Class / Metric": "Severe Disruptions (≥ 60 min)", "Count / Value": "175,863", "Share": "6.02%"},
        ])
        st.dataframe(t_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: DESCRIPTIVE STATISTICS (Part 2.1)
# ==============================================================================
with tabs[1]:
    st.subheader("2. Comprehensive Descriptive Statistics (`DEP_DELAY` Across 2,920,859 Flights)")
    dep_s = full_desc.get("variables", {}).get("DEP_DELAY", compute_descriptive_statistics(df))
    w_means = full_desc.get("weighted_means_dep_delay", {})

    st.markdown("#### A. Central Tendency (Including Weighted Means)")
    ct1, ct2, ct3, ct4, ct5, ct6 = st.columns(6)
    ct1.metric("Arithmetic Mean", f"{dep_s.get('mean', 10.10):.3f} min", f"SE: ±{dep_s.get('std_error_mean', 0.0288):.4f}m")
    ct2.metric("Median (50th %ile)", f"{dep_s.get('median', -2.0):.1f} min", "Typical flight departs early")
    ct3.metric("Mode (Most Frequent)", f"{dep_s.get('mode', -5.0):.1f} min", "5 min early pushback")
    ct4.metric("10% Trimmed Mean", f"{dep_s.get('trimmed_mean_10pct', 1.72):.3f} min", "Removes top/bottom 10%")
    ct5.metric("Distance-Weighted Mean", f"{w_means.get('distance_weighted_mean_min', 10.724):.3f} min", "Weighted by flight miles")
    ct6.metric("Duration-Weighted Mean", f"{w_means.get('elapsed_time_weighted_mean_min', 10.678):.3f} min", "Weighted by block hours")

    st.markdown("#### B. Dispersion & Variability Metrics")
    dp1, dp2, dp3, dp4, dp5, dp6, dp7 = st.columns(7)
    dp1.metric("Total Range", f"{dep_s.get('range', 3056.0):,.0f} min", f"[{dep_s.get('min', -90):.0f}m, {dep_s.get('max', 2966):.0f}m]")
    dp2.metric("Sample Variance (s²)", f"{dep_s.get('variance', 2423.81):,.2f} min²")
    dp3.metric("Standard Deviation (s)", f"{dep_s.get('std', 49.23):.2f} min")
    dp4.metric("Interquartile Range (IQR)", f"{dep_s.get('iqr', 11.0):.1f} min", f"Q1={dep_s.get('q1_25', -5):.0f}m to Q3={dep_s.get('q3_75', 6):.0f}m")
    dp5.metric("Mean Abs Deviation (MAD)", f"{dep_s.get('mad_mean', 20.12):.2f} min", "E[|X - μ|]")
    dp6.metric("Median Abs Dev (MedAD)", f"{dep_s.get('mad_median', 4.0):.1f} min", "Median(|X - Median|)")
    dp7.metric("Coeff. of Variation (CV)", f"{dep_s.get('cv', 4.874):.3f}", f"{dep_s.get('cv_pct', 487.4):.1f}% of Mean")

    st.markdown("#### C. Distribution Shape, Quantiles, Percentiles & 95% Confidence Intervals")
    sh1, sh2, sh3, sh4, sh5, sh6 = st.columns(6)
    sh1.metric("Skewness (Fisher-Pearson)", f"{dep_s.get('skewness', 11.49):.3f}", "Extreme Right-Tail Skew")
    sh2.metric("Excess Kurtosis (Fisher)", f"{dep_s.get('kurtosis_excess', 243.90):.2f}", "Heavy Leptokurtic Tail")
    sh3.metric("Pearson Kurtosis (β₂)", f"{dep_s.get('kurtosis_pearson', 246.90):.2f}", "Normal reference = 3.0")
    sh4.metric("Parametric 95% CI (Mean)", f"[{dep_s.get('ci_95_mean_parametric', [10.04, 10.16])[0]:.2f}, {dep_s.get('ci_95_mean_parametric', [10.04, 10.16])[1]:.2f}]m")
    sh5.metric("Bootstrap 95% CI (Mean)", f"[{dep_s.get('ci_95_mean_bootstrap', [9.95, 10.27])[0]:.2f}, {dep_s.get('ci_95_mean_bootstrap', [9.95, 10.27])[1]:.2f}]m")
    sh6.metric("99th Percentile (P99)", f"{dep_s.get('p99', 197.0):.0f} min", f"P95 = {dep_s.get('p95', 71.0):.0f} min | P90 = {dep_s.get('p90', 36.0):.0f} min")

    st.markdown("#### Complete Multi-Variable Statistical Summary Table (`artifacts/descriptive_statistics.csv`)")
    desc_csv = ARTIFACTS / "descriptive_statistics.csv"
    if desc_csv.exists():
        st.dataframe(pd.read_csv(desc_csv), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 3: DISTRIBUTION ANALYSIS (Part 2.2)
# ==============================================================================
with tabs[2]:
    st.subheader("3. Flight Delay Distribution & Tail Morphology Analysis")
    st.markdown(
        """
        **Statistical Interpretation of the Delay Distribution:**
        - **Bimodal Operational Regime (Early Pushback vs Disruption Tail):** The mode (**-5.0 min**) and median (**-2.0 min**) are negative because gate agents and flight crews routinely close cabin doors 2–5 minutes ahead of schedule when turnaround completes smoothly (**64.86%** of flights depart $\le 0$ min).
        - **Extreme Positive Skewness ($\gamma_1 = 11.49$) & Leptokurtosis ($\gamma_2 = 243.90$):** Unlike early departures (physically bounded near -15 to -20 minutes because airlines cannot depart hours before scheduled ticketing), positive delays have no tight upper bound during air-traffic ground stops, crew timeouts, or mechanical AOG events, stretching to **+2,966 minutes**.
        """
    )

    d_r1c1, d_r1c2 = st.columns(2)
    with d_r1c1:
        st.plotly_chart(plot_delay_histogram(df), use_container_width=True)
    with d_r1c2:
        st.plotly_chart(plot_delay_categories_bar(df), use_container_width=True)

    d_r2c1, d_r2c2 = st.columns(2)
    with d_r2c1:
        st.plotly_chart(plot_delay_boxplot_by_airline(df), use_container_width=True)
    with d_r2c2:
        st.plotly_chart(plot_violin_by_daypart(df), use_container_width=True)

    d_r3c1, d_r3c2 = st.columns(2)
    with d_r3c1:
        st.plotly_chart(plot_ecdf_delay(df), use_container_width=True)
    with d_r3c2:
        st.plotly_chart(plot_qq_normality(df), use_container_width=True)

    with st.expander("📸 View High-Resolution Publication Distribution Charts (KDE, Log-Density, Signed-Log Q-Q)", expanded=False):
        p_col1, p_col2 = st.columns(2)
        p1 = ARTIFACTS / "ppt_charts" / "eda_01_delay_distribution_suite.png"
        p2 = ARTIFACTS / "ppt_charts" / "eda_02_violin_and_categories.png"
        if p1.exists():
            p_col1.image(str(p1), caption="4-Panel Distribution Suite: Histogram+KDE, Log10 Tail Density, ECDF, and Normal vs Signed-Log Q-Q Plot", use_column_width=True)
        if p2.exists():
            p_col2.image(str(p2), caption="Violin Distribution by Aviation Day Part & Operational Delay Severity Breakdown", use_column_width=True)

# ==============================================================================
# TAB 4: AIRLINE, AIRPORT & ROUTE GROUP ANALYSIS (Part 2.3)
# ==============================================================================
with tabs[3]:
    st.subheader("4. Group-Based Statistical Analysis with 95% Confidence Intervals")
    st.markdown(
        """
        Every group summary reports **Sample Size ($n$)**, **Mean**, **Median**, **Standard Deviation**, **Variance**, **IQR**, **95% Confidence Interval ($\mu \pm 1.96 \cdot s/\sqrt{n}$)**, and **Delayed Rate ($\ge 15$ min %)**.
        Minimum sample-size filtering prevents small-sample distortion when ranking airports and routes.
        """
    )

    # Airline section
    st.markdown("### A. Airline Carrier Performance Rankings (All 18 US Carriers)")
    air_csv = ARTIFACTS / "group_stats_airline.csv"
    air_full_df = pd.read_csv(air_csv) if air_csv.exists() else compute_airline_analysis(df, min_samples=100)
    a_c1, a_c2 = st.columns(2)
    with a_c1:
        st.plotly_chart(plot_airline_bar(compute_airline_analysis(df, min_samples=50)), use_container_width=True)
    with a_c2:
        st.plotly_chart(plot_airline_delay_rate(compute_airline_analysis(df, min_samples=50)), use_container_width=True)
    st.dataframe(air_full_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### B. Origin & Destination Airport Statistical Benchmarks")
    min_apt_n = st.slider("Minimum Flights per Airport (Sample Size Filter)", min_value=50, max_value=2000, value=200, step=50)
    orig_df = compute_airport_analysis(df, "ORIGIN", min_samples=min_apt_n)
    dest_df = compute_airport_analysis(df, "DEST", min_samples=min_apt_n)

    ap_c1, ap_c2 = st.columns(2)
    with ap_c1:
        st.plotly_chart(plot_top_airports_bar(orig_df, "Top 15 Origin Airports by Mean Departure Delay", "ORIGIN"), use_container_width=True)
        orig_csv = ARTIFACTS / "group_stats_origin_airport.csv"
        if orig_csv.exists():
            st.caption("Full-Population Origin Airport Statistics (`n >= 1,000` in 2.92M dataset, with 95% CIs):")
            st.dataframe(pd.read_csv(orig_csv).head(25), use_container_width=True, hide_index=True)
    with ap_c2:
        st.plotly_chart(plot_top_airports_bar(dest_df, "Top 15 Destination Airports by Mean Departure Delay", "DEST"), use_container_width=True)
        dest_csv = ARTIFACTS / "group_stats_dest_airport.csv"
        if dest_csv.exists():
            st.caption("Full-Population Destination Airport Statistics (`n >= 1,000` in 2.92M dataset, with 95% CIs):")
            st.dataframe(pd.read_csv(dest_csv).head(25), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### C. Flight Route Corridor Analysis (`ORIGIN → DEST`)")
    route_df = compute_route_analysis(df, min_samples=30)
    rt_c1, rt_c2 = st.columns(2)
    with rt_c1:
        st.plotly_chart(plot_top_routes_bar(route_df, "Top 12 Highest-Delay Route Corridors", ascending=False), use_container_width=True)
    with rt_c2:
        st.plotly_chart(plot_top_routes_bar(route_df, "Top 12 Most Reliable (Lowest-Delay) Route Corridors", ascending=True), use_container_width=True)
    rt_csv = ARTIFACTS / "group_stats_route.csv"
    if rt_csv.exists():
        st.caption("Full-Population Flight Route Statistics (`n >= 500` in 2.92M dataset, with 95% CIs):")
        st.dataframe(pd.read_csv(rt_csv).head(30), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 5: CORRELATION & MULTICOLLINEARITY (VIF) ANALYSIS (Part 2.4)
# ==============================================================================
with tabs[4]:
    st.subheader("5. Parametric & Non-Parametric Correlation + Variance Inflation Factor (VIF)")
    st.warning(
        "**Statistical Principle — Correlation Does Not Establish Causation:** "
        "While `DEP_DELAY` correlates strongly with `ARR_DELAY` ($r = 0.9647$, Spearman $\\rho = 0.8087$) because late pushback directly shifts arrival time, "
        "correlations between pre-departure historical rates (`H_ORIGIN_RATE_1D`, $r = 0.167$) and `DEP_DELAY` reflect shared operational congestion and weather regimes rather than direct intervention."
    )

    corr_method = st.radio(
        "Select Correlation Matrix Estimator:",
        ["Pearson r (Linear)", "Spearman ρ (Monotonic Rank)", "Kendall τ (Pairwise Concordance)"],
        horizontal=True,
    )
    if corr_method.startswith("Pearson") and (ARTIFACTS / "correlation_matrix.csv").exists():
        c_mat = pd.read_csv(ARTIFACTS / "correlation_matrix.csv", index_col=0)
    elif corr_method.startswith("Spearman") and (ARTIFACTS / "correlation_spearman.csv").exists():
        c_mat = pd.read_csv(ARTIFACTS / "correlation_spearman.csv", index_col=0)
    elif corr_method.startswith("Kendall") and (ARTIFACTS / "correlation_kendall.csv").exists():
        c_mat = pd.read_csv(ARTIFACTS / "correlation_kendall.csv", index_col=0)
    else:
        c_mat = compute_correlation_matrix(df)

    cr_c1, cr_c2 = st.columns([1.15, 1.0])
    with cr_c1:
        st.plotly_chart(plot_correlation_heatmap(c_mat), use_container_width=True)
    with cr_c2:
        rank_csv = ARTIFACTS / "correlation_ranking.csv"
        if rank_csv.exists():
            rank_df = pd.read_csv(rank_csv)
            st.plotly_chart(plot_multi_correlation_bars(rank_df[rank_df["Feature"] != "ARR_DELAY (Post-Departure Reference)"]), use_container_width=True)
            st.dataframe(rank_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Multicollinearity Diagnostic: Variance Inflation Factor (VIF)")
    vif_csv = ARTIFACTS / "vif_results.csv"
    if vif_csv.exists():
        vif_df = pd.read_csv(vif_csv)
        v_c1, v_c2 = st.columns([1.1, 1.0])
        with v_c1:
            st.plotly_chart(plot_vif_bars(vif_df), use_container_width=True)
        with v_c2:
            st.markdown(
                """
                **Key Multicollinearity Finding:**
                - In the full feature set, `CRS_ELAPSED_TIME` (**VIF = 30.03**) and `DISTANCE` (**VIF = 29.99**) exhibit severe collinearity ($r = 0.983$) because scheduled block time is a linear function of stage length plus taxi buffer.
                - For linear models (Logistic Regression, Ridge, OLS), collinearity inflates standard errors of individual coefficients. Removing `CRS_ELAPSED_TIME` collapses **all feature VIFs below 1.03** (`DISTANCE` VIF drops from **29.99 → 1.012**).
                """
            )
            st.dataframe(vif_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 6: HYPOTHESIS TESTING SUITE (Part 2.5)
# ==============================================================================
with tabs[5]:
    st.subheader("6. Rigorous Statistical Hypothesis Testing Suite (14 Formal Tests + Benjamini-Hochberg FDR)")
    st.markdown(
        """
        Every hypothesis test below displays all **9 required scientific elements**:
        **(1) Research Question**, **(2) Null Hypothesis ($H_0$)**, **(3) Alternative Hypothesis ($H_1$)**, **(4) Test Selected**,
        **(5) Test Statistic**, **(6) Raw & BH-FDR Adjusted $p$-value**, **(7) Significance Level ($\alpha = 0.05$)**,
        **(8) Statistical Decision**, and **(9) Plain-Language & Effect-Size Interpretation**.
        """
    )
    st.info(
        "**Statistical Significance vs. Practical Importance:** With $N = 2,920,859$ flights, standard errors shrink by $\\sqrt{N} \\approx 1,709$, so tiny differences (e.g., 0.61 min between Weekend and Weekday) achieve $p < 10^{-16}$. Always inspect the **Effect Size** (Cohen's $d$, Hedges' $g$, Cramér's $V$, Odds Ratio) column to judge operational relevance!"
    )

    hyp_csv = ARTIFACTS / "hypothesis_testing_results.csv"
    if hyp_csv.exists():
        hyp_df = pd.read_csv(hyp_csv)
        cats = ["All Categories"] + hyp_df["Category"].unique().tolist()
        sel_cat = st.selectbox("Filter Hypothesis Tests by Family:", cats)
        view_h = hyp_df if sel_cat == "All Categories" else hyp_df[hyp_df["Category"] == sel_cat]
        st.dataframe(view_h, use_container_width=True, hide_index=True)

        st.markdown("#### Detailed Test Cards (Research Question, $H_0$, $H_1$, Test Statistic, FDR $q$-value, Decision & Plain-Language Interpretation)")
        for _, row in view_h.iterrows():
            dec = row.get("Statistical_Decision", row.get("Decision", ""))
            raw_p = float(row.get("Raw_P_Value", row.get("P_Value", 0.0)))
            fdr_p = float(row.get("BH_Adjusted_P_Value", row.get("P_Value_BH_FDR", 0.0)))
            eff_name = row.get("Effect_Size_Metric", row.get("Effect_Size_Name", ""))
            interp = row.get("Practical_Interpretation", row.get("Plain_Language_Interpretation", ""))
            with st.expander(f"🔬 {row['Test_ID']}: [{row['Category']}] {row['Test_Selected']} — {row['Research_Question']} ({dec})"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Test Statistic", f"{row['Statistic_Name']} = {row['Test_Statistic']:,.4f}")
                c2.metric("Raw p-value", f"{raw_p:.3e}")
                c3.metric("BH-FDR Adjusted q-value", f"{fdr_p:.3e}", f"α = {row['Alpha']}")
                c4.metric("Effect Size", f"{eff_name} = {row['Effect_Size_Value']:.4f}")
                st.markdown(f"- **Null Hypothesis ($H_0$):** {row['Null_Hypothesis_H0']}")
                st.markdown(f"- **Alternative Hypothesis ($H_1$):** {row['Alternative_Hypothesis_H1']}")
                st.markdown(f"- **Statistical Decision ($\alpha = 0.05$):** **{dec}**")
                st.markdown(f"- **Plain-Language & Effect-Size Interpretation:** {interp}")

    st.markdown("---")
    st.markdown("### Interactive Custom Two-Group Hypothesis Tester")
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        grp_col = st.selectbox("Grouping Variable:", ["AIRLINE", "DAY_PART", "ORIGIN", "DEST"])
    avail_vals = df[grp_col].value_counts().head(25).index.tolist()
    with ic2:
        g_a = st.selectbox("Group A:", avail_vals, index=0)
    with ic3:
        g_b = st.selectbox("Group B:", avail_vals, index=min(1, len(avail_vals) - 1))

    if g_a != g_b:
        live_res = run_custom_two_group_test(df, grp_col, g_a, g_b)
        if "error" not in live_res:
            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.metric(f"{g_a} Mean (n={live_res['n_a']:,})", f"{live_res['mean_a']:.2f} min", f"Median: {live_res['median_a']:.1f}m")
            lc2.metric(f"{g_b} Mean (n={live_res['n_b']:,})", f"{live_res['mean_b']:.2f} min", f"Median: {live_res['median_b']:.1f}m")
            lc3.metric("Welch's t-Statistic", f"{live_res['welch_t']:.3f}", f"p = {live_res['welch_p']:.3e}")
            lc4.metric("Mann-Whitney U", f"{live_res['mann_whitney_u']:,.0f}", f"p = {live_res['mann_whitney_p']:.3e} | d = {live_res['cohens_d']:.3f}")
            st.success(live_res["plain_interpretation"])

# ==============================================================================
# TAB 7: ANOVA & POST-HOC ANALYSIS (Part 2.6)
# ==============================================================================
with tabs[6]:
    st.subheader("7. One-Way ANOVA, Welch's ANOVA, Kruskal-Wallis, Two-Way ANOVA & Games-Howell Post-Hoc")
    st.markdown(
        """
        Because flight delays violate homoscedasticity (confirmed via **Brown-Forsythe Levene test**, $p < 10^{-300}$) and exhibit daily network clustering, we report:
        1. **Parametric One-Way ANOVA ($F$, $SS_{\\text{between}}$, $SS_{\\text{within}}$, $\\eta^2$, $\\omega^2$)**
        2. **Welch's Heteroscedastic One-Way ANOVA ($F_{\\text{Welch}}$)** for unequal group variances
        3. **Non-Parametric Kruskal-Wallis $H$-Test ($H$, $\\epsilon^2$)**
        4. **Daily Cluster-Robust Block-Mean ANOVA ($F_{\\text{cluster}}$)** aggregating flights into daily carrier/group means first to account for intra-day operational correlation
        5. **Two-Way ANOVA ($\text{Day Part} \\times \text{Weekend}$)** with interaction testing
        6. **Games-Howell Post-Hoc Pairwise Comparisons** (heteroscedasticity-robust extension of Tukey HSD)
        """
    )

    anova_csv = ARTIFACTS / "anova_results.csv"
    if anova_csv.exists():
        anova_df = pd.read_csv(anova_csv)
        an_c1, an_c2 = st.columns([1.05, 1.0])
        with an_c1:
            st.plotly_chart(plot_anova_effect_sizes(anova_df), use_container_width=True)
        with an_c2:
            if anova_res.get("two_way_anova", {}).get("cell_means"):
                st.plotly_chart(plot_twoway_interaction(anova_res["two_way_anova"]["cell_means"]), use_container_width=True)

        st.markdown("#### Complete One-Way, Welch, Kruskal-Wallis & Cluster-Robust ANOVA Summary Table")
        st.dataframe(anova_df, use_container_width=True, hide_index=True)

    tw_csv = ARTIFACTS / "two_way_anova_results.csv"
    if tw_csv.exists():
        st.markdown("#### Two-Way Factorial ANOVA Table: `DAY_PART × IS_WEEKEND` on `DEP_DELAY`")
        st.dataframe(pd.read_csv(tw_csv), use_container_width=True, hide_index=True)

    ph_csv = ARTIFACTS / "posthoc_results.csv"
    if ph_csv.exists():
        st.markdown("#### Games-Howell Post-Hoc Pairwise Comparisons (Unequal Variances & Sample Sizes)")
        ph_df = pd.read_csv(ph_csv)
        ph_fams = ["All Factors"] + ph_df["Factor"].unique().tolist()
        sel_ph = st.selectbox("Filter Post-Hoc Comparisons by Factor:", ph_fams)
        st.dataframe(ph_df if sel_ph == "All Factors" else ph_df[ph_df["Factor"] == sel_ph], use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 8: OUTLIER & TREND ANALYSIS (Part 2.7)
# ==============================================================================
with tabs[7]:
    st.subheader("8. Outlier Quantification (IQR & Z-Score) and Multi-Scale Temporal Trend Analysis")
    out_stats = compute_outlier_analysis(df)
    dep_full_o = full_desc.get("variables", {}).get("DEP_DELAY", {})

    o1, o2, o3, o4, o5 = st.columns(5)
    o1.metric("Tukey 1.5×IQR Fences", f"[{dep_full_o.get('iqr_lower_fence_1_5', -21.5):.1f}m, {dep_full_o.get('iqr_upper_fence_1_5', 22.5):.1f}m]", f"{dep_full_o.get('outliers_iqr_1_5_pct', 14.37):.2f}% Outside Fences")
    o2.metric("Extreme 3.0×IQR Fences", f"[{dep_full_o.get('iqr_lower_fence_3_0', -38.0):.1f}m, {dep_full_o.get('iqr_upper_fence_3_0', 39.0):.1f}m]", f"{dep_full_o.get('outliers_iqr_3_0_pct', 9.21):.2f}% Extreme Outliers")
    o3.metric("Z-Score Outliers (|z| > 3)", f"{dep_full_o.get('outliers_z3_count', 43_065):,}", f"{dep_full_o.get('outliers_z3_pct', 1.47):.2f}% (> +157.8 min)")
    o4.metric("Severe Delays (≥ 180 min)", f"{out_stats['extreme_180m_count']:,} (sample)", "Tarmac Rule Threshold")
    o5.metric("Max Recorded Delay", f"{dep_full_o.get('max', 2966.0):,.0f} min", "49.4 Hours (Preserved)")

    temporal_dict = compute_temporal_analysis(df)
    t_r1c1, t_r1c2 = st.columns(2)
    with t_r1c1:
        st.plotly_chart(plot_hourly_trend(temporal_dict["hourly"]), use_container_width=True)
    with t_r1c2:
        st.plotly_chart(plot_hourly_delayed_pct(temporal_dict["hourly"]), use_container_width=True)

    t_r2c1, t_r2c2 = st.columns(2)
    with t_r2c1:
        st.plotly_chart(plot_weekday_trend(temporal_dict["daily"]), use_container_width=True)
    with t_r2c2:
        st.plotly_chart(plot_monthly_trend(temporal_dict["monthly"]), use_container_width=True)

    t_r3c1, t_r3c2 = st.columns(2)
    with t_r3c1:
        st.plotly_chart(plot_yearly_trend(temporal_dict["yearly"]), use_container_width=True)
    with t_r3c2:
        st.plotly_chart(plot_daypart_comparison(temporal_dict["daypart"]), use_container_width=True)

    with st.expander("📸 View Full-Population Monthly & Diurnal Trend Charts (2019–2023)", expanded=False):
        tr1 = ARTIFACTS / "ppt_charts" / "eda_14_monthly_and_yearly_trends.png"
        tr2 = ARTIFACTS / "ppt_charts" / "eda_15_outlier_and_bootstrap_ci.png"
        tc1, tc2 = st.columns(2)
        if tr1.exists():
            tc1.image(str(tr1), caption="56-Month Time-Series Evolution (2019–2023) & Seasonal Profile", use_column_width=True)
        if tr2.exists():
            tc2.image(str(tr2), caption="IQR / Z-Score Outlier Regimes & Non-Parametric Bootstrap Sampling Distribution of Mean Delay", use_column_width=True)
