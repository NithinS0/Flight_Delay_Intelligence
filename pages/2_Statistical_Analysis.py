import streamlit as st
import pandas as pd
import numpy as np

from src.app_helper import initialize_app_state
from src.eda import (
    compute_dataset_overview,
    compute_descriptive_statistics,
    compute_delay_categories,
    compute_airline_analysis,
    compute_airport_analysis,
    compute_route_analysis,
    compute_temporal_analysis,
    compute_correlation_matrix,
    generate_dynamic_insights,
)
from src.visualizations import (
    plot_delay_donut,
    plot_delay_histogram,
    plot_delay_boxplot,
    plot_kde_distribution,
    plot_delay_categories_bar,
    plot_airline_bar,
    plot_airline_delayed_pct,
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
)

st.set_page_config(
    page_title="Statistical Analysis & EDA | Flight Delay Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_app_state()
df = st.session_state.df
summary_meta = st.session_state.get("summary", {})

# Page Header
st.markdown('<div class="aviation-badge">📊 CORE EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)
st.title("Statistical Analysis & Exploratory Data Analysis")
st.markdown(
    """
    Comprehensive empirical exploration of commercial airline flight operations, departure delay distributions, 
    carrier performance variances, airport bottlenecks, route corridors, temporal diurnal dynamics, and feature correlations.
    All metrics and visualizations are dynamically calculated from the live dataset.
    """
)

st.markdown("---")

# Precompute core analytics
overview_meta = compute_dataset_overview(df, summary_meta=summary_meta)
desc_stats = compute_descriptive_statistics(df)
delay_cat_df = compute_delay_categories(df)
airline_df = compute_airline_analysis(df)
temporal_dict = compute_temporal_analysis(df)
corr_matrix = compute_correlation_matrix(df)

# Tabs Organization for clear structured navigation
tab_overview, tab_dist, tab_airline, tab_airport, tab_route, tab_time, tab_corr_insights = st.tabs([
    "📋 Overview & Descriptive Stats",
    "📈 Delay Distribution",
    "✈️ Airline Performance",
    "🏢 Airport Analysis",
    "🛫 Route Analysis",
    "⏰ Time-Based Analysis",
    "🔍 Correlation & Dynamic Insights"
])

# ==========================================
# TAB 1: SECTION A & SECTION B
# ==========================================
with tab_overview:
    st.subheader("Section A – Dataset Overview")
    st.markdown("Global structural parameters and data integrity metrics.")
    
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        st.metric("Total Records", f"{overview_meta['total_rows']:,}")
        st.metric("Operating Airlines", f"{overview_meta['airlines_count']}")
    with col_a2:
        st.metric("Total Features", f"{overview_meta['total_cols']}")
        st.metric("Origin Airports", f"{overview_meta['origins_count']}")
    with col_a3:
        st.metric("Date Range Horizon", overview_meta["date_range"])
        st.metric("Destination Airports", f"{overview_meta['dest_count']}")
    with col_a4:
        st.metric("Unique Route Corridors", f"{overview_meta['routes_count']:,}")
        st.metric("Initial Duplicates", f"{overview_meta['duplicates_count']}")

    st.markdown("---")
    
    st.subheader("Section B – Comprehensive Descriptive Statistics")
    st.markdown(
        "Complete statistical summary of departure delay duration in minutes, quantifying central tendency, "
        "dispersion, five-number summary, IQR, skewness, and kurtosis."
    )
    
    stats_table_data = [
        {"Statistic": "Sample Count (N)", "Value": f"{desc_stats['count']:,}", "Description": "Total non-cancelled flights with valid departure delay"},
        {"Statistic": "Mean Departure Delay", "Value": f"{desc_stats['mean']:.4f} min", "Description": "Arithmetic average duration of departure delay"},
        {"Statistic": "Median Departure Delay", "Value": f"{desc_stats['median']:.4f} min", "Description": "50th percentile (typical flight pushes back 2m early)"},
        {"Statistic": "Mode Departure Delay", "Value": f"{desc_stats['mode']:.4f} min", "Description": "Most frequent observed departure delay value"},
        {"Statistic": "Standard Deviation (σ)", "Value": f"{desc_stats['std']:.4f} min", "Description": "Standard dispersion around the mean"},
        {"Statistic": "Sample Variance (s²)", "Value": f"{desc_stats['variance']:.3f}", "Description": "Squared deviation metric reflecting extreme variance"},
        {"Statistic": "Minimum Observed Delay", "Value": f"{desc_stats['min']:.4f} min", "Description": "Maximum early departure buffer observed"},
        {"Statistic": "Maximum Observed Delay", "Value": f"{desc_stats['max']:.4f} min", "Description": "Genuine extreme operational black-swan delay (preserved)"},
        {"Statistic": "First Quartile (Q1 - 25%)", "Value": f"{desc_stats['q1']:.4f} min", "Description": "25% of flights depart with delay below this threshold"},
        {"Statistic": "Second Quartile (Q2 - 50%)", "Value": f"{desc_stats['q2']:.4f} min", "Description": "Median midpoint of the delay distribution"},
        {"Statistic": "Third Quartile (Q3 - 75%)", "Value": f"{desc_stats['q3']:.4f} min", "Description": "75% of flights depart with delay below this threshold"},
        {"Statistic": "Interquartile Range (IQR)", "Value": f"{desc_stats['iqr']:.4f} min", "Description": "Spread of the middle 50% of flights (Q3 - Q1)"},
        {"Statistic": "Skewness (Fisher-Pearson)", "Value": f"{desc_stats['skewness']:.4f}", "Description": "Severe positive right skew (heavy tail of long delays)"},
        {"Statistic": "Excess Kurtosis", "Value": f"{desc_stats['kurtosis']:.4f}", "Description": "Extreme leptokurtic distribution with heavy fat tails"},
        {"Statistic": "Tukey Lower Fence (Q1 - 1.5·IQR)", "Value": f"{desc_stats['lower_fence']:.2f} min", "Description": "Statistical threshold for early outliers"},
        {"Statistic": "Tukey Upper Fence (Q3 + 1.5·IQR)", "Value": f"{desc_stats['upper_fence']:.2f} min", "Description": "Statistical threshold for delay outliers"},
        {"Statistic": "IQR Outlier Flight Count", "Value": f"{desc_stats['outlier_count']:,} ({desc_stats['outlier_percentage']:.2f}%)", "Description": "Genuine operational delays preserved without truncation"},
        {"Statistic": "Delayed Flights (≥ 15 min)", "Value": f"{desc_stats['delayed_flights']:,} ({desc_stats['delayed_percentage']:.2f}%)", "Description": "Official FAA operational disruption classification target"},
        {"Statistic": "On-Time Flights (< 15 min)", "Value": f"{desc_stats['non_delayed_flights']:,} ({desc_stats['non_delayed_percentage']:.2f}%)", "Description": "Flights operating within scheduled buffer window"},
    ]
    st.dataframe(pd.DataFrame(stats_table_data), use_container_width=True, hide_index=True)


# ==========================================
# TAB 2: SECTION C – DELAY DISTRIBUTION
# ==========================================
with tab_dist:
    st.subheader("Section C – Departure Delay Distributions")
    st.markdown(
        "Empirical visualization of departure delay characteristics, demonstrating heavy positive skewness, "
        "the FAA 15-minute operational disruption boundary, and structured duration categories."
    )
    
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.plotly_chart(plot_delay_histogram(df), use_container_width=True)
    with d_col2:
        st.plotly_chart(plot_kde_distribution(df), use_container_width=True)

    d_col3, d_col4 = st.columns(2)
    with d_col3:
        st.plotly_chart(plot_delay_boxplot(df), use_container_width=True)
    with d_col4:
        st.plotly_chart(plot_delay_donut(desc_stats['delayed_flights'], desc_stats['non_delayed_flights']), use_container_width=True)

    st.markdown("#### Operational Delay Categories Breakdown")
    st.markdown(
        "Categorization provides granular operational insight into dispatch severity. "
        "Notice that flights departing between -5m and +14m represent the standard operating regime."
    )
    st.plotly_chart(plot_delay_categories_bar(delay_cat_df), use_container_width=True)
    st.dataframe(delay_cat_df, use_container_width=True, hide_index=True)


# ==========================================
# TAB 3: SECTION D – AIRLINE PERFORMANCE
# ==========================================
with tab_airline:
    st.subheader("Section D – Airline Carrier Performance")
    st.markdown("Empirical comparison of operating airlines across departure punctuality, delay rates, and flight volumes.")
    
    st.plotly_chart(plot_airline_bar(airline_df), use_container_width=True)
    
    air_col1, air_col2 = st.columns(2)
    with air_col1:
        st.plotly_chart(plot_airline_delayed_pct(airline_df), use_container_width=True)
    with air_col2:
        st.plotly_chart(plot_airline_volume(airline_df), use_container_width=True)

    st.markdown("#### Interactive Airline Drilldown Selector")
    selected_airline = st.selectbox("Select an Airline to Inspect Granular Operational Statistics:", airline_df["Airline"].tolist())
    
    if selected_airline:
        airline_record = airline_df[airline_df["Airline"] == selected_airline].iloc[0]
        airline_flights = df[df["AIRLINE"] == selected_airline]
        
        c_al1, c_al2, c_al3, c_al4 = st.columns(4)
        with c_al1:
            st.metric("Total Carrier Flights", f"{airline_record['Flight Count']:,}")
        with c_al2:
            st.metric("Average Departure Delay", f"{airline_record['Mean Delay (min)']:.2f} min")
        with c_al3:
            st.metric("Median Delay", f"{airline_record['Median Delay (min)']:.2f} min")
        with c_al4:
            st.metric("Disruption Rate (≥15m)", f"{airline_record['Delayed %']:.2f}%")
            
        st.markdown(f"**Carrier Data Summary Table for {selected_airline}:**")
        st.dataframe(airline_df[airline_df["Airline"] == selected_airline], use_container_width=True, hide_index=True)


# ==========================================
# TAB 4: SECTION E – AIRPORT ANALYSIS
# ==========================================
with tab_airport:
    st.subheader("Section E – Origin and Destination Airport Analysis")
    st.markdown(
        "Analysis of airport delay patterns. A minimum flight threshold is enforced to prevent small regional airfields "
        "with single-flight disruptions from distorting network rankings."
    )
    
    min_flights_threshold = st.slider("Minimum Flight Sample Threshold per Airport:", min_value=10, max_value=200, value=30, step=5)
    st.caption(f"Currently displaying airports with at least **{min_flights_threshold} scheduled flights** in the dataset.")
    
    airport_dict = compute_airport_analysis(df, min_flights=min_flights_threshold)
    origins_df = airport_dict["origins"]
    dests_df = airport_dict["destinations"]
    
    ap_col1, ap_col2 = st.columns(2)
    with ap_col1:
        if not origins_df.empty:
            st.plotly_chart(plot_top_airports_bar(origins_df, f"Top Origin Gateways by Average Delay (≥{min_flights_threshold} flights)", "Origin Airport"), use_container_width=True)
            st.dataframe(origins_df.head(15), use_container_width=True, hide_index=True)
        else:
            st.info("No origin airports match the selected flight threshold.")
            
    with ap_col2:
        if not dests_df.empty:
            st.plotly_chart(plot_top_airports_bar(dests_df, f"Top Destination Gateways by Average Delay (≥{min_flights_threshold} flights)", "Destination Airport"), use_container_width=True)
            st.dataframe(dests_df.head(15), use_container_width=True, hide_index=True)
        else:
            st.info("No destination airports match the selected flight threshold.")

    if not origins_df.empty:
        st.markdown("#### Airport Volume vs. Delay Duration")
        st.plotly_chart(plot_airport_volume_vs_delay(origins_df, "Origin Airport"), use_container_width=True)


# ==========================================
# TAB 5: SECTION F – ROUTE ANALYSIS
# ==========================================
with tab_route:
    st.subheader("Section F – Origin-Destination Route Corridor Analysis")
    st.markdown(
        "Corridor delay dynamics (`ROUTE = ORIGIN → DEST`). To provide statistically meaningful comparisons, "
        "a sample volume threshold is applied."
    )
    
    route_threshold = st.slider("Minimum Flights per Route Corridor Threshold:", min_value=10, max_value=100, value=20, step=5)
    route_df = compute_route_analysis(df, min_flights=route_threshold)
    
    if not route_df.empty:
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.plotly_chart(plot_top_routes_bar(route_df, f"Top Delayed Route Corridors (≥{route_threshold} flights)", ascending=False), use_container_width=True)
        with r_col2:
            st.plotly_chart(plot_top_routes_bar(route_df, f"Most Punctual Route Corridors (≥{route_threshold} flights)", ascending=True), use_container_width=True)

        st.markdown(f"**Busiest and Most Delayed Corridors Meeting Threshold (N = {len(route_df):,} corridors):**")
        st.dataframe(route_df.head(25), use_container_width=True, hide_index=True)
    else:
        st.info("No routes meet the selected threshold. Try lowering the threshold slider.")


# ==========================================
# TAB 6: SECTION G – TIME-BASED ANALYSIS
# ==========================================
with tab_time:
    st.subheader("Section G – Temporal Diurnal, Weekly, Monthly & Yearly Dynamics")
    st.markdown("Explores how departure delays evolve across hours of the day, weekdays, seasons, and years.")
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.plotly_chart(plot_hourly_trend(temporal_dict["hourly"]), use_container_width=True)
    with t_col2:
        st.plotly_chart(plot_hourly_delayed_pct(temporal_dict["hourly"]), use_container_width=True)

    t_col3, t_col4 = st.columns(2)
    with t_col3:
        st.plotly_chart(plot_weekday_trend(temporal_dict["daily"]), use_container_width=True)
    with t_col4:
        st.plotly_chart(plot_monthly_trend(temporal_dict["monthly"]), use_container_width=True)

    t_col5, t_col6 = st.columns(2)
    with t_col5:
        st.plotly_chart(plot_yearly_trend(temporal_dict["yearly"]), use_container_width=True)
    with t_col6:
        if not temporal_dict["daypart"].empty:
            st.plotly_chart(plot_daypart_comparison(temporal_dict["daypart"]), use_container_width=True)
        else:
            st.info("Day part feature not yet engineered in current frame.")


# ==========================================
# TAB 7: SECTION H & SECTION I
# ==========================================
with tab_corr_insights:
    st.subheader("Section H – Correlation Analysis")
    st.markdown(
        "Pearson correlation matrix between continuous departure delay and pre-flight scheduled parameters. "
        "Features are strictly leakage-safe."
    )
    
    st.plotly_chart(plot_correlation_heatmap(corr_matrix), use_container_width=True)
    
    st.markdown(
        """
        > [!NOTE]
        > **Interpretation of Observed Correlations:**
        > - **Scheduled Hour vs. Departure Delay (~ +0.09):** The strongest temporal signal is scheduled departure hour. Because delays cascade across aircraft rotations throughout the operating day, later departures incur higher delay risk.
        > - **Distance vs. Scheduled Elapsed Time (~ +0.98):** Extremely strong collinearity exists between scheduled flight distance and scheduled elapsed block time. Both variables are normalized via `StandardScaler` during pipeline training.
        > - **Day of Week & Month vs. Delay:** Modest correlation values reflect non-linear seasonal and holiday peaks rather than simple linear trends.
        > 
        > **Scientific Causation Warning:** Correlation measures linear association and does NOT imply direct physical causation. Unobserved weather systems, convective storms, and runway construction introduce non-linear variance.
        """
    )
    
    st.markdown("---")
    
    st.subheader("Section I – Dynamic Analytical Insights")
    st.markdown("Automated insights derived from the computed statistics of the live dataset:")
    
    insights_list = generate_dynamic_insights(df, airline_df, airport_dict, route_df, temporal_dict)
    
    for item in insights_list:
        st.markdown(
            f"""
            <div class="light-card" style="margin-bottom:12px; padding:16px 20px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong style="color:#1e3a8a; font-size:1rem;">{item['title']}</strong>
                    <span style="font-family:monospace; font-weight:700; color:#2563eb; background:#eff6ff; padding:2px 8px; border-radius:6px; border:1px solid #bfdbfe;">
                        {item['stat']}
                    </span>
                </div>
                <p style="margin:6px 0 0 0; color:#475569; font-size:0.88rem;">{item['desc']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
