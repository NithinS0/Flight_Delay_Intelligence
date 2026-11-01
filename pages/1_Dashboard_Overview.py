import streamlit as st

from src.app_helper import initialize_app_state
from src import ui_data as data
from src.ui_theme import page_header, section, kpi_grid, callout, note
from src.model_evaluation import get_best_models_summary
from src.visualizations import (
    plot_delay_donut,
    plot_delay_histogram,
    plot_airline_bar,
    plot_monthly_flights_series,
    plot_monthly_delay_series,
    plot_delay_categories_bar,
)

st.set_page_config(
    page_title="Dashboard Overview | Flight Delay Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_app_state()
df = st.session_state.df
n = len(df)

ov = data.overview_stats(df, n)
series = data.monthly_series(df, n)
airline_df = data.airline_table(df, n, 100)
best = get_best_models_summary()
best_cls, best_reg = best.get("classification", {}), best.get("regression", {})

page_header(
    "Flight Delay Intelligence Dashboard",
    "Statistical Analysis, Machine Learning and Flight Delay Prediction",
    f"This system studies {ov['total']:,} completed US commercial flights ({ov['date_min']} to {ov['date_max']}) across "
    f"{ov['airlines']} airlines. It combines comprehensive statistical intelligence with 4 Classification and 4 Regression "
    "models evaluated on genuine operational flight data without synthetic balancing.",
    eyebrow="4 Classification Models | 4 Regression Models",
)

section("Key indicators", "Computed from the full cleaned flight table.")
kpi_grid([
    {"label": "Total flights analysed", "value": f"{ov['total']:,}", "note": f"{ov['date_min']} to {ov['date_max']}"},
    {"label": "Delayed flights", "value": f"{ov['delayed']:,}", "note": "Departure delay of 15 minutes or more", "tone": "orange"},
    {"label": "Non-delayed flights", "value": f"{ov['on_time']:,}", "note": "Departure delay under 15 minutes", "tone": "teal"},
    {"label": "Delay percentage", "value": f"{ov['delayed_pct']:.2f}%", "note": "Share of flights delayed 15+ min", "tone": "orange"},
    {"label": "Average departure delay", "value": f"{ov['mean']:.2f} min", "note": f"Median {ov['median']:+.1f} min (right-skewed)", "tone": "blue"},
    {"label": "Airlines", "value": f"{ov['airlines']}", "note": f"{ov['routes']:,} routes, {ov['airports']} origin airports", "tone": "blue"},
])

section("Traffic and delay trends", "Monthly view across the full study period.")
st.plotly_chart(plot_monthly_flights_series(series), use_container_width=True)
_hi, _lo = series.loc[series["flights"].idxmax()], series.loc[series["flights"].idxmin()]
note(
    f"Busiest month: {_hi['period']:%b %Y} ({int(_hi['flights']):,} flights). Quietest month: {_lo['period']:%b %Y} "
    f"({int(_lo['flights']):,} flights)."
)
st.plotly_chart(plot_monthly_delay_series(series), use_container_width=True)
_dh = series.loc[series["mean_delay"].idxmax()]
note(
    f"Solid line: mean departure delay (left axis). Dotted line: share of flights delayed 15 minutes or more (right axis). "
    f"Highest monthly mean delay: {_dh['period']:%b %Y} ({_dh['mean_delay']:.1f} min)."
)

section("Delay profile", "How delays are distributed and how airlines compare.")
c1, c2 = st.columns(2, gap="large")
with c1:
    st.plotly_chart(plot_delay_donut(ov["delayed"], ov["on_time"]), use_container_width=True)
with c2:
    st.plotly_chart(data.figure("histogram", df, n), use_container_width=True)
note(
    f"{ov['le0_pct']:.1f}% of flights depart on time or early (delay of 0 minutes or less), while a long right tail of severe "
    f"delays pulls the mean ({ov['mean']:.1f} min) well above the median ({ov['median']:+.1f} min)."
)

st.plotly_chart(plot_airline_bar(airline_df), use_container_width=True)
note("Mean departure delay per airline. The airline with the highest mean delay is highlighted in orange.")

st.plotly_chart(data.figure("categories", df, n), use_container_width=True)
note("Delay categories: early (under -5 min), on time or minor (-5 to 15), moderate (15 to 60), high (60 to 120) and severe (120+ min).")

if best_cls and best_reg:
    section("Champion models", "Final out-of-time evaluation on 632,306 untouched test flights (4 Classification | 4 Regression).")
    kpi_grid([
        {"label": "Best classifier", "value": best_cls["name"], "note": f"F1 {best_cls['test_f1_decimal']:.4f} | ROC-AUC {best_cls['test_roc_auc_decimal']:.4f}", "tone": "teal"},
        {"label": "Classifier recall / precision", "value": f"{best_cls['test_recall']:.1f}% / {best_cls['test_precision']:.1f}%", "note": f"Accuracy {best_cls['test_accuracy']:.1f}% | PR-AUC {best_cls['test_pr_auc_decimal']:.4f}", "tone": "teal"},
        {"label": "Best regressor (MAE)", "value": best_reg["name"], "note": f"Test MAE {best_reg['test_mae']:.2f} min | Median AE {best_reg['test_median_ae']:.2f} min", "tone": "blue"},
        {"label": "MAE vs mean baseline", "value": f"-{best_reg['mae_improvement_pct']:.1f}%", "note": f"Baseline MAE {best_reg['baseline_mean_mae']:.2f} min", "tone": "blue"},
    ])
    callout(
        "Use the sidebar to open the statistical analysis, the classification and regression evaluation pages, "
        "the model comparison tables and the project documentation.",
        tone="info",
    )
