import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats

import textwrap
from typing import Optional
# Light Theme Palette Tokens (shared design system: navy / muted blue / teal / muted orange)
PAPER_BG = "#ffffff"
PLOT_BG = "#ffffff"
TEXT_COLOR = "#1f2933"
SUBTEXT_COLOR = "#64748b"
GRID_COLOR = "#edf1f6"
BORDER_COLOR = "#cfd8e3"
FONT_FAMILY = "Inter, 'Segoe UI', system-ui, -apple-system, Roboto, Arial, sans-serif"

# Curated Accent Tokens (names kept for backwards compatibility)
PRIMARY_BLUE = "#3b6aa8"      # general data
SECONDARY_BLUE = "#9db9dc"    # light / supporting blue
ACCENT_CYAN = "#0f8f87"
ACCENT_INDIGO = "#14305c"     # navy
ACCENT_EMERALD = "#0f8f87"    # teal: positive / selected
ACCENT_AMBER = "#d98a3d"      # muted orange: comparisons
ACCENT_ROSE = "#c0392b"       # red: warnings / negative findings only
NEUTRAL_GREY = "#a8b4c4"      # supporting information

def apply_light_layout(
    fig: go.Figure, 
    title: str, 
    x_title: str = "", 
    y_title: str = "",
    show_legend: bool = False,
    height: Optional[int] = None,
) -> go.Figure:
    """Applies the shared light analytics design system to Plotly charts.

    Long titles are wrapped, the legend sits below the plot area and axes use
    automargin, so titles, legends and tick labels never collide.
    """
    lines = textwrap.wrap(title, width=64) or [title]
    title_html = "<br>".join(lines)
    top = 28 + 22 * len(lines)
    fig.update_layout(
        template="plotly_white",
        height=height if height is not None else (fig.layout.height or 400),
        title=dict(
            text=f"<b>{title_html}</b>",
            font=dict(family=FONT_FAMILY, size=14, color=TEXT_COLOR),
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR, size=12),
        margin=dict(l=24, r=24, t=top + 8, b=24 + (36 if show_legend else 0), pad=4),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0,
            font=dict(size=11, color=SUBTEXT_COLOR),
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=12, color=SUBTEXT_COLOR), standoff=10),
            showgrid=False,
            linecolor=BORDER_COLOR,
            tickfont=dict(size=11, color=SUBTEXT_COLOR),
            zeroline=False,
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=12, color=SUBTEXT_COLOR), standoff=10),
            showgrid=True,
            gridcolor=GRID_COLOR,
            gridwidth=1,
            linecolor=BORDER_COLOR,
            tickfont=dict(size=11, color=SUBTEXT_COLOR),
            zeroline=False,
            automargin=True,
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_size=12,
            font_family=FONT_FAMILY,
            bordercolor=BORDER_COLOR,
        ),
        uniformtext=dict(minsize=9, mode="show"),
    )
    _pad_bar_ranges(fig)
    return fig


def _pad_bar_ranges(fig: go.Figure) -> None:
    """Gives outside bar labels headroom so value labels are never clipped or overlapped."""
    for tr in fig.data:
        if isinstance(tr, go.Bar) and tr.text is not None and tr.textposition == "outside":
            tr.cliponaxis = False
            try:
                if tr.orientation == "h":
                    vals = np.asarray(tr.x, dtype=float)
                    if vals.size:
                        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
                        span = max(hi - min(lo, 0.0), 1e-9)
                        fig.update_xaxes(range=[min(lo, 0.0) - (0.04 * span if lo < 0 else 0), hi + 0.22 * span])
                else:
                    vals = np.asarray(tr.y, dtype=float)
                    if vals.size:
                        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
                        span = max(hi - min(lo, 0.0), 1e-9)
                        fig.update_yaxes(range=[min(lo, 0.0) - (0.04 * span if lo < 0 else 0), hi + 0.16 * span])
            except (TypeError, ValueError):
                pass

def plot_delay_donut(delayed_count: int, ontime_count: int) -> go.Figure:
    """Renders a clean donut chart comparing delayed vs on-time flights."""
    labels = ["On-time (under 15 min)", "Delayed (15 min or more)"]
    values = [ontime_count, delayed_count]
    colors = [PRIMARY_BLUE, ACCENT_AMBER]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            sort=False,
            direction="clockwise",
            marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
            textinfo="percent",
            textposition="inside",
            textfont=dict(color="#ffffff", size=13),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        )
    ])
    apply_light_layout(fig, "Delayed vs On-Time Flights (15-minute threshold)", show_legend=True, height=400)
    fig.update_layout(
        annotations=[dict(text=f"<b>{delayed_count + ontime_count:,}</b><br><span style='font-size:11px;color:{SUBTEXT_COLOR}'>flights</span>", x=0.5, y=0.5, font_size=17, showarrow=False, font_color=TEXT_COLOR)]
    )
    return fig

def plot_delay_histogram(df: pd.DataFrame) -> go.Figure:
    """Plots delay distribution histogram highlighting the 15-minute delayed threshold."""
    delays = df["DEP_DELAY"].dropna()
    clipped = np.clip(delays, -40, 120)
    
    # Pre-bin for speed and responsive rendering
    counts, bin_edges = np.histogram(clipped, bins=32, range=(-40, 120))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bar_colors = [ACCENT_AMBER if x >= 15 else PRIMARY_BLUE for x in bin_centers]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        width=4.5,
        marker=dict(color=bar_colors, line=dict(color="#ffffff", width=1)),
        hovertemplate="Departure Delay: ~%{x:.0f} min<br>Flight Volume: %{y:,}<extra></extra>",
    ))
    
    fig.add_vline(
        x=15, line_width=2, line_dash="dash", line_color=ACCENT_INDIGO, 
        annotation_text="15 min delay threshold", 
        annotation_position="top right",
        annotation_font=dict(color=ACCENT_INDIGO, size=11)
    )
    
    return apply_light_layout(fig, "Departure Delay Distribution (-40m to 120m+)", "Departure Delay (minutes)", "Flight Count")

def plot_delay_boxplot(df: pd.DataFrame, sample_size: int = 10000) -> go.Figure:
    """Plots five-number summary boxplot preserving extreme outliers."""
    delays = df["DEP_DELAY"].dropna()
    if len(delays) > sample_size:
        # Sample for smooth responsive browser rendering while preserving min/max extremes
        sample = delays.sample(sample_size, random_state=42).tolist()
        sample.extend([float(delays.min()), float(delays.max())])
        data_to_plot = sample
    else:
        data_to_plot = delays
        
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=data_to_plot,
        name="All Flights",
        boxpoints="outliers",
        marker=dict(color=NEUTRAL_GREY, size=3, opacity=0.5),
        line=dict(color=PRIMARY_BLUE, width=1.5),
        fillcolor="rgba(59, 106, 168, 0.14)",
        hovertemplate="Delay: %{x} min<extra></extra>",
    ))
    return apply_light_layout(fig, "Five-number summary and extreme delay outliers", "Departure Delay (minutes)", "", height=360)

plot_delay_boxplot_by_airline = plot_delay_boxplot

def plot_kde_distribution(df: pd.DataFrame, sample_size: int = 15000) -> go.Figure:
    """Computes and renders smooth KDE density curve for departure delay."""
    delays = df["DEP_DELAY"].dropna()
    clipped = np.clip(delays, -40, 120)
    if len(clipped) > sample_size:
        sample = clipped.sample(sample_size, random_state=42).values
    else:
        sample = clipped.values
        
    kde = stats.gaussian_kde(sample, bw_method=0.15)
    x_range = np.linspace(-40, 120, 300)
    density = kde(x_range)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_range,
        y=density,
        mode="lines",
        line=dict(color=PRIMARY_BLUE, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(59, 106, 168, 0.15)",
        hovertemplate="Delay: %{x:.1f} min<br>Probability Density: %{y:.5f}<extra></extra>",
    ))
    fig.add_vline(x=15, line_width=2, line_dash="dash", line_color=ACCENT_AMBER, annotation_text="15 min threshold", annotation_font_color=ACCENT_AMBER)
    return apply_light_layout(fig, "Kernel Density Estimate (KDE) Distribution Curve", "Departure Delay (minutes)", "Probability Density")

def plot_delay_categories_bar(cat_df: pd.DataFrame) -> go.Figure:
    """Plots delay category distribution."""
    if "Category" not in cat_df.columns and "DEP_DELAY" in cat_df.columns:
        from src.statistical_analysis import compute_delay_categories
        cat_df = compute_delay_categories(cat_df)
    colors = [PRIMARY_BLUE, SECONDARY_BLUE, "#e3b07a", ACCENT_AMBER, ACCENT_ROSE]
    labels = ["<br>".join(textwrap.wrap(str(c), 14)) for c in cat_df["Category"]]
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=cat_df["Flight Count"],
            text=cat_df["Percentage"].apply(lambda p: f"{p:.1f}%"),
            textposition="outside",
            marker=dict(color=colors, line=dict(color="#ffffff", width=1)),
            customdata=np.stack([cat_df["Range Definition"], cat_df["Percentage"]], axis=-1),
            hovertemplate="<b>%{x}</b><br>Definition: %{customdata[0]}<br>Flights: %{y:,}<br>Share: %{customdata[1]}%<extra></extra>",
        )
    ])
    return apply_light_layout(fig, "Delay category distribution", "", "Flight count")

def plot_airline_bar(airline_df: pd.DataFrame) -> go.Figure:
    """Plots horizontal bar chart of average departure delay by airline."""
    df_sorted = airline_df.sort_values("Mean Delay (min)", ascending=True)
    top = df_sorted["Mean Delay (min)"].max()
    colors = [ACCENT_AMBER if x == top else PRIMARY_BLUE for x in df_sorted["Mean Delay (min)"]]
    
    fig = go.Figure(go.Bar(
        x=df_sorted["Mean Delay (min)"],
        y=df_sorted["Airline"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="#ffffff", width=1)),
        text=df_sorted["Mean Delay (min)"].apply(lambda x: f"{x:.1f} min"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Mean Delay: %{x:.2f} min<extra></extra>",
    ))
    return apply_light_layout(fig, "Average departure delay by airline", "Mean departure delay (minutes)", "", height=max(380, 26 * len(df_sorted) + 120))

def plot_airline_delayed_pct(airline_df: pd.DataFrame) -> go.Figure:
    """Plots horizontal bar chart of delayed percentage by airline."""
    df_sorted = airline_df.sort_values("Delayed %", ascending=True)
    top = df_sorted["Delayed %"].max()
    fig = go.Figure(go.Bar(
        x=df_sorted["Delayed %"],
        y=df_sorted["Airline"],
        orientation="h",
        marker=dict(color=[ACCENT_AMBER if v == top else PRIMARY_BLUE for v in df_sorted["Delayed %"]], line=dict(color="#ffffff", width=1)),
        text=df_sorted["Delayed %"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Delayed rate (15+ min): %{x:.1f}%<extra></extra>",
    ))
    return apply_light_layout(fig, "Share of flights delayed 15 minutes or more, by airline", "Delayed flights (%)", "", height=max(380, 26 * len(df_sorted) + 120))

plot_airline_delay_rate = plot_airline_delayed_pct

def plot_airline_volume(airline_df: pd.DataFrame) -> go.Figure:
    """Plots flight count volume per carrier."""
    df_sorted = airline_df.sort_values("Flight Count", ascending=True)
    fig = go.Figure(go.Bar(
        x=df_sorted["Flight Count"],
        y=df_sorted["Airline"],
        orientation="h",
        marker=dict(color=ACCENT_INDIGO, line=dict(color="#ffffff", width=1)),
        text=df_sorted["Flight Count"].apply(lambda x: f"{x:,}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Total Flights: %{x:,}<extra></extra>",
    ))
    return apply_light_layout(fig, "Airline Flight Volume (Market Share)", "Total Flights Analysed", "")

def plot_top_airports_bar(airport_df: pd.DataFrame, title: str, airport_col: str, top_n: int = 15) -> go.Figure:
    """Plots top airports ranked by average delay."""
    df_top = airport_df.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=df_top["Mean Delay (min)"],
        y=df_top[airport_col],
        orientation="h",
        marker=dict(color=PRIMARY_BLUE, line=dict(color="#ffffff", width=1)),
        text=df_top["Mean Delay (min)"].apply(lambda x: f"{x:.1f}m"),
        textposition="outside",
        hovertemplate="<b>Airport: %{y}</b><br>Mean Delay: %{x:.2f} min<extra></extra>",
    ))
    return apply_light_layout(fig, title, "Mean departure delay (minutes)", "", height=max(380, 26 * len(df_top) + 120))


def plot_airport_volume_vs_delay(airport_df: pd.DataFrame, airport_col: str) -> go.Figure:
    """Plots scatter of flight volume versus mean departure delay."""
    top_labels = set(airport_df.nlargest(12, "Flight Count")[airport_col].astype(str))
    labels = [str(a) if str(a) in top_labels else "" for a in airport_df[airport_col]]
    fig = go.Figure(go.Scatter(
        x=airport_df["Flight Count"],
        y=airport_df["Mean Delay (min)"],
        mode="markers+text",
        text=labels,
        customdata=np.stack([airport_df[airport_col].astype(str), airport_df["Delayed %"]], axis=-1),
        textposition="top center",
        textfont=dict(size=10, color=SUBTEXT_COLOR),
        marker=dict(
            size=9,
            color=airport_df["Delayed %"],
            colorscale=[[0, "#dbe6f4"], [1, "#14305c"]],
            showscale=True,
            colorbar=dict(title=dict(text="Delayed %", side="right"), tickfont=dict(size=10), thickness=12),
            line=dict(color="#ffffff", width=0.8),
            opacity=0.9,
        ),
        hovertemplate="<b>%{customdata[0]}</b><br>Flights: %{x:,}<br>Mean delay: %{y:.1f} min<br>Delayed rate: %{customdata[1]:.1f}%<extra></extra>",
    ))
    fig.update_xaxes(type="log")
    return apply_light_layout(fig, f"{airport_col}: flight volume vs average delay (labels: 12 busiest)", "Total flights (log scale)", "Mean delay (minutes)", height=460)

def plot_top_routes_bar(route_df: pd.DataFrame, title: str, ascending: bool = False, top_n: int = 12) -> go.Figure:
    """Plots top or lowest delayed routes."""
    df_sorted = route_df.sort_values("Mean Delay (min)", ascending=ascending).head(top_n)
    if not ascending:
        df_sorted = df_sorted.iloc[::-1]
    color = ACCENT_AMBER if not ascending else ACCENT_EMERALD
    
    fig = go.Figure(go.Bar(
        x=df_sorted["Mean Delay (min)"],
        y=df_sorted["Route Corridor"],
        orientation="h",
        marker=dict(color=color, line=dict(color="#ffffff", width=1)),
        text=df_sorted["Mean Delay (min)"].apply(lambda x: f"{x:.1f}m"),
        textposition="outside",
        hovertemplate="<b>Route: %{y}</b><br>Mean Delay: %{x:.2f} min<extra></extra>",
    ))
    return apply_light_layout(fig, title, "Mean Delay (minutes)", "")

def plot_hourly_trend(hourly_df: pd.DataFrame) -> go.Figure:
    """Plots diurnal departure delay curve by scheduled departure hour."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_df["Hour Label"],
        y=hourly_df["Mean Delay (min)"],
        mode="lines+markers",
        name="Mean Delay (min)",
        line=dict(color=PRIMARY_BLUE, width=3),
        marker=dict(size=7, color=PRIMARY_BLUE),
        hovertemplate="Hour: %{x}<br>Mean Delay: %{y:.2f} min<extra></extra>",
    ))
    return apply_light_layout(fig, "Diurnal Trend: Average Delay by Scheduled Departure Hour", "Scheduled Departure Hour", "Mean Delay (minutes)", show_legend=False)

def plot_hourly_delayed_pct(hourly_df: pd.DataFrame) -> go.Figure:
    """Plots delayed percentage by scheduled departure hour."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly_df["Hour Label"],
        y=hourly_df["Delayed %"],
        marker=dict(color=SECONDARY_BLUE, line=dict(color="#ffffff", width=1)),
        hovertemplate="Hour: %{x}<br>Delayed Rate: %{y:.1f}%<extra></extra>",
    ))
    return apply_light_layout(fig, "Diurnal Disruption Rate: % Delayed by Scheduled Departure Hour", "Scheduled Departure Hour", "Delayed % (≥15m)")

def plot_weekday_trend(daily_df: pd.DataFrame) -> go.Figure:
    """Plots average delay by day of week."""
    fig = go.Figure(go.Bar(
        x=daily_df["Day of Week"],
        y=daily_df["Mean Delay (min)"],
        marker=dict(color=ACCENT_INDIGO, line=dict(color="#ffffff", width=1)),
        text=daily_df["Mean Delay (min)"].apply(lambda x: f"{x:.1f}m"),
        textposition="outside",
        hovertemplate="Day: %{x}<br>Mean Delay: %{y:.2f} min<br>Flights: %{customdata:,}<extra></extra>",
        customdata=daily_df["Flight Count"],
    ))
    return apply_light_layout(fig, "Weekly Dynamic: Average Departure Delay by Weekday", "Day of Week", "Mean Delay (minutes)")

def plot_monthly_trend(monthly_df: pd.DataFrame) -> go.Figure:
    """Plots monthly average delay trend."""
    fig = go.Figure(go.Scatter(
        x=monthly_df["Month"],
        y=monthly_df["Mean Delay (min)"],
        mode="lines+markers",
        line=dict(color=PRIMARY_BLUE, width=3),
        marker=dict(size=8, color=PRIMARY_BLUE),
        hovertemplate="Month: %{x}<br>Mean Delay: %{y:.2f} min<extra></extra>",
    ))
    return apply_light_layout(fig, "Seasonality: average departure delay by calendar month", "Month", "Mean delay (minutes)")

def plot_yearly_trend(yearly_df: pd.DataFrame) -> go.Figure:
    """Plots yearly average delay comparison."""
    fig = go.Figure(go.Bar(
        x=yearly_df["Year"].astype(str),
        y=yearly_df["Mean Delay (min)"],
        marker=dict(color=PRIMARY_BLUE, line=dict(color="#ffffff", width=1)),
        text=yearly_df["Mean Delay (min)"].apply(lambda x: f"{x:.1f}m"),
        textposition="outside",
        hovertemplate="Year: %{x}<br>Mean Delay: %{y:.2f} min<br>Flights: %{customdata:,}<extra></extra>",
        customdata=yearly_df["Flight Count"],
    ))
    return apply_light_layout(fig, "Yearly Multi-Year Departure Delay Evolution", "Year", "Mean Delay (minutes)")

def plot_daypart_comparison(daypart_df: pd.DataFrame) -> go.Figure:
    """Plots day part comparison (Morning, Afternoon, Evening, Night)."""
    peak = daypart_df["Mean Delay (min)"].max()
    fig = go.Figure(go.Bar(
        x=daypart_df["Day Part"],
        y=daypart_df["Mean Delay (min)"],
        marker=dict(color=[ACCENT_AMBER if v == peak else PRIMARY_BLUE for v in daypart_df["Mean Delay (min)"]], line=dict(color="#ffffff", width=1)),
        text=daypart_df["Mean Delay (min)"].apply(lambda x: f"{x:.1f}m"),
        textposition="outside",
        hovertemplate="Day Part: %{x}<br>Mean Delay: %{y:.2f} min<br>Delayed: %{customdata:.1f}%<extra></extra>",
        customdata=daypart_df["Delayed %"],
    ))
    return apply_light_layout(fig, "Aviation Day Part Comparison", "Day Part", "Mean Delay (minutes)")

def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Renders interactive annotated Pearson correlation heatmap."""
    z = corr_df.values
    x = corr_df.columns.tolist()
    y = corr_df.index.tolist()
    
    text_annotations = np.round(z, 2).astype(str)
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=[[0.0, "#14305c"], [0.25, "#7fa0cc"], [0.5, "#ffffff"], [0.75, "#eab57f"], [1.0, "#b8651b"]],
        zmin=-1,
        zmax=1,
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"size": 11},
        colorbar=dict(title=dict(text="r", side="right"), tickfont=dict(size=10), thickness=12),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.4f}<extra></extra>",
    ))
    fig.update_xaxes(tickangle=-40, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return apply_light_layout(fig, "Correlation matrix heatmap", "", "", height=max(460, 34 * len(y) + 160))

def plot_confusion_matrix(cm: list, class_names: list = None) -> go.Figure:
    """Plots clean annotated confusion matrix heatmap with counts and row percentages."""
    if class_names is None:
        class_names = ["On-time (under 15 min)", "Delayed (15+ min)"]
    z = np.array(cm)
    row_pct = z / np.maximum(z.sum(axis=1, keepdims=True), 1) * 100.0
    labels = [[f"<b>{z[i, j]:,}</b><br>{row_pct[i, j]:.1f}% of actual" for j in range(z.shape[1])] for i in range(z.shape[0])]
    dark = z.max() * 0.55
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=class_names,
        y=class_names,
        colorscale=[[0.0, "#f1f5fa"], [1.0, "#14305c"]],
        text=labels,
        texttemplate="%{text}",
        textfont={"size": 13},
        showscale=False,
        xgap=3, ygap=3,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>",
    ))
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            fig.add_annotation(x=class_names[j], y=class_names[i], text=labels[i][j], showarrow=False,
                               font=dict(size=13, color="#ffffff" if z[i, j] > dark else "#1f2933"))
    fig.update_traces(texttemplate="")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return apply_light_layout(fig, "Confusion matrix (test split)", "Predicted label", "Actual label", height=420)

def plot_model_metric_bars(
    models_df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_title: str,
    highlight: Optional[str] = None,
    higher_is_better: bool = True,
    top_n: Optional[int] = None,
    fmt: str = "{:.3f}",
) -> go.Figure:
    """Horizontal, sorted bar chart comparing one metric across models.

    The best model (or `highlight`) is teal; all other models stay blue so a single
    meaningful winner stands out instead of colouring every bar.
    """
    if metric_col not in models_df.columns:
        return go.Figure()
    d = models_df[["Model", metric_col]].copy()
    cleaned_col = d[metric_col].astype(str).str.replace("%", "", regex=False).str.replace("min", "", regex=False).str.replace("m", "", regex=False).str.replace("+", "", regex=False).str.strip()
    d[metric_col] = pd.to_numeric(cleaned_col, errors="coerce")
    d = d.dropna().sort_values(metric_col, ascending=not higher_is_better)
    if d.empty:
        return go.Figure()
    if top_n:
        d = d.head(top_n)
    best_name = highlight if highlight in set(d["Model"]) else d["Model"].iloc[0]
    d = d.iloc[::-1]
    colors = [ACCENT_EMERALD if m == best_name else PRIMARY_BLUE for m in d["Model"]]
    fig = go.Figure(go.Bar(
        x=d[metric_col],
        y=d["Model"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="#ffffff", width=1)),
        text=d[metric_col].apply(lambda v: fmt.format(v)),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>" + metric_col + ": %{x:.4f}<extra></extra>",
    ))
    return apply_light_layout(fig, title, y_title, "", height=max(340, 30 * len(d) + 130))

def plot_actual_vs_predicted_scatter(y_true: np.ndarray, y_pred: np.ndarray, sample_n: int = 3000) -> go.Figure:
    """Renders actual vs predicted scatter plot with 45-degree identity reference line."""
    if len(y_true) > sample_n:
        idx = np.random.RandomState(42).choice(len(y_true), sample_n, replace=False)
        yt, yp = y_true[idx], y_pred[idx]
    else:
        yt, yp = y_true, y_pred
        
    min_val = min(float(np.quantile(yt, 0.005)), float(np.min(yp)), -20.0)
    max_val = max(float(np.quantile(yt, 0.99)), float(np.quantile(yp, 0.99)), 120.0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yt,
        y=yp,
        mode="markers",
        name="Test flights (sample)",
        marker=dict(size=4, color=PRIMARY_BLUE, opacity=0.35),
        hovertemplate="Actual: %{x:.1f} min<br>Predicted: %{y:.1f} min<extra></extra>",
    ))
    # Identity line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color=ACCENT_AMBER, width=2, dash="dash"),
        name="Perfect prediction (y = x)",
    ))
    fig.update_xaxes(range=[min_val, max_val])
    fig.update_yaxes(range=[min_val, max_val])
    return apply_light_layout(fig, "Actual vs predicted departure delay", "Actual delay (minutes)", "Predicted delay (minutes)", show_legend=True, height=440)

def plot_residuals_distribution(residuals: np.ndarray) -> go.Figure:
    """Plots histogram of model residuals."""
    clipped = np.clip(residuals, -60, 60)
    counts, bin_edges = np.histogram(clipped, bins=35, range=(-60, 60))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    fig = go.Figure(go.Bar(
        x=bin_centers,
        y=counts,
        marker=dict(color=PRIMARY_BLUE, line=dict(color="#ffffff", width=1)),
        hovertemplate="Residual: %{x:.1f} min<br>Count: %{y:,}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=2, line_dash="solid", line_color=ACCENT_AMBER, annotation_text="Zero error", annotation_font_color=ACCENT_AMBER)
    return apply_light_layout(fig, "Residual distribution (actual minus predicted, clipped to ±60 min)", "Residual (minutes)", "Count", height=420)

def plot_residuals_vs_predicted(y_pred: np.ndarray, residuals: np.ndarray, sample_n: int = 3000) -> go.Figure:
    """Plots residuals vs predicted delay."""
    if len(y_pred) > sample_n:
        idx = np.random.RandomState(42).choice(len(y_pred), sample_n, replace=False)
        yp, res = y_pred[idx], residuals[idx]
    else:
        yp, res = y_pred, residuals
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yp,
        y=res,
        mode="markers",
        marker=dict(size=4, color=PRIMARY_BLUE, opacity=0.35),
        hovertemplate="Predicted: %{x:.1f} min<br>Residual: %{y:.1f} min<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=2, line_dash="dash", line_color=ACCENT_AMBER)
    fig.update_yaxes(range=[float(np.quantile(res, 0.005)) - 5, float(np.quantile(res, 0.99)) + 5])
    return apply_light_layout(fig, "Residuals vs predicted values", "Predicted delay (minutes)", "Residual (minutes)", height=420)


def plot_residual_qq(residuals: np.ndarray, sample_n: int = 2500) -> go.Figure:
    """Normal Q-Q plot of model residuals."""
    from scipy import stats as sp_stats

    r = np.asarray(residuals, dtype=float)
    rng = np.random.default_rng(42)
    sub = rng.choice(r, size=min(sample_n, len(r)), replace=False)
    (osm, osr), (slope, intercept, r_val) = sp_stats.probplot(sub, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Residual quantiles",
                             marker=dict(size=5, color=PRIMARY_BLUE, opacity=0.55),
                             hovertemplate="Normal quantile: %{x:.2f}<br>Residual: %{y:.1f} min<extra></extra>"))
    xl = np.array([float(np.min(osm)), float(np.max(osm))])
    fig.add_trace(go.Scatter(x=xl, y=slope * xl + intercept, mode="lines", name=f"Normal reference (R² = {r_val ** 2:.3f})",
                             line=dict(color=ACCENT_AMBER, width=2, dash="dash")))
    return apply_light_layout(fig, "Normal Q-Q plot of residuals", "Theoretical normal quantiles", "Residual quantiles (minutes)", show_legend=True, height=440)


def plot_monthly_flights_series(series: pd.DataFrame) -> go.Figure:
    """Flights per calendar month (time series)."""
    fig = go.Figure(go.Scatter(
        x=series["period"], y=series["flights"], mode="lines+markers",
        line=dict(color=PRIMARY_BLUE, width=2.5), marker=dict(size=5, color=PRIMARY_BLUE),
        fill="tozeroy", fillcolor="rgba(59, 106, 168, 0.10)",
        hovertemplate="%{x|%b %Y}<br>Flights: %{y:,}<extra></extra>",
    ))
    fig.update_xaxes(tickformat="%b %Y", dtick="M6")
    return apply_light_layout(fig, "Monthly flight volume", "Month", "Completed flights", height=360)


def plot_monthly_delay_series(series: pd.DataFrame) -> go.Figure:
    """Mean departure delay per calendar month with delayed-share overlay."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series["period"], y=series["mean_delay"], mode="lines+markers", name="Mean delay (min)",
        line=dict(color=ACCENT_INDIGO, width=2.5), marker=dict(size=5, color=ACCENT_INDIGO),
        hovertemplate="%{x|%b %Y}<br>Mean delay: %{y:.2f} min<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=series["period"], y=series["delayed_pct"], mode="lines", name="Delayed share (%)", yaxis="y2",
        line=dict(color=ACCENT_AMBER, width=2, dash="dot"),
        hovertemplate="%{x|%b %Y}<br>Delayed share: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title=dict(text="Delayed share (%)", font=dict(size=12, color=SUBTEXT_COLOR)),
                                   tickfont=dict(size=11, color=SUBTEXT_COLOR), automargin=True))
    fig.update_xaxes(tickformat="%b %Y", dtick="M6")
    return apply_light_layout(fig, "Monthly average departure delay and delayed share", "Month", "Mean delay (minutes)", show_legend=True, height=380)


def _curve_palette(selected: Optional[str], names) -> dict:
    return {n: (ACCENT_EMERALD if n == selected else NEUTRAL_GREY) for n in names}


def plot_roc_curves(curves: dict, selected: Optional[str] = None) -> go.Figure:
    """ROC curves; `curves` maps model name -> (fpr, tpr, auc). Selected model is teal, others grey."""
    fig = go.Figure()
    colors = _curve_palette(selected, curves)
    order = [n for n in curves if n != selected] + ([selected] if selected in curves else [])
    for n in order:
        fpr, tpr, auc = curves[n]
        is_sel = n == selected
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{n} (AUC {auc:.3f})" if is_sel else n,
            line=dict(color=colors[n], width=3 if is_sel else 1.4), opacity=1.0 if is_sel else 0.7,
            hovertemplate=f"{n}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color=ACCENT_AMBER, width=1.5, dash="dash")))
    fig.update_xaxes(range=[0, 1], showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(range=[0, 1.02])
    return apply_light_layout(fig, "ROC curve", "False positive rate", "True positive rate", show_legend=True, height=460)


def plot_pr_curves(curves: dict, base_rate: float, selected: Optional[str] = None) -> go.Figure:
    """Precision-recall curves; `curves` maps model name -> (precision, recall, ap)."""
    fig = go.Figure()
    colors = _curve_palette(selected, curves)
    order = [n for n in curves if n != selected] + ([selected] if selected in curves else [])
    for n in order:
        prec, rec, ap = curves[n]
        is_sel = n == selected
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines", name=f"{n} (AP {ap:.3f})" if is_sel else n,
            line=dict(color=colors[n], width=3 if is_sel else 1.4), opacity=1.0 if is_sel else 0.7,
            hovertemplate=f"{n}<br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[base_rate, base_rate], mode="lines", name=f"Base rate ({base_rate:.1%})",
                             line=dict(color=ACCENT_AMBER, width=1.5, dash="dash")))
    fig.update_xaxes(range=[0, 1], showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(range=[0, 1.02])
    return apply_light_layout(fig, "Precision-recall curve", "Recall", "Precision", show_legend=True, height=460)


def plot_threshold_analysis(thr: np.ndarray, precision: np.ndarray, recall: np.ndarray, f1: np.ndarray, chosen: Optional[float] = None) -> go.Figure:
    """Precision, recall and F1 as a function of the decision threshold."""
    fig = go.Figure()
    for name, vals, col in (("Precision", precision, PRIMARY_BLUE), ("Recall", recall, ACCENT_AMBER), ("F1 score", f1, ACCENT_EMERALD)):
        fig.add_trace(go.Scatter(x=thr, y=vals, mode="lines", name=name, line=dict(color=col, width=2.5),
                                 hovertemplate=name + " at %{x:.2f}: %{y:.3f}<extra></extra>"))
    if chosen is not None:
        fig.add_vline(x=chosen, line_width=1.5, line_dash="dash", line_color=ACCENT_INDIGO,
                      annotation_text=f"Selected threshold {chosen:.3f}", annotation_position="top", annotation_font_color=ACCENT_INDIGO)
    fig.update_yaxes(range=[0, 1.02])
    return apply_light_layout(fig, "Threshold analysis: precision, recall and F1", "Decision threshold", "Score", show_legend=True, height=420)


def plot_group_means(labels, means, ci_low=None, ci_high=None, title: str = "Group means", y_title: str = "Mean departure delay (minutes)", highlight: Optional[str] = None) -> go.Figure:
    """Group means with optional 95% confidence intervals (for ANOVA sections)."""
    labels = list(labels)
    means = np.asarray(means, dtype=float)
    err = None
    if ci_low is not None and ci_high is not None:
        err = dict(type="data", symmetric=False, array=np.asarray(ci_high, dtype=float) - means, arrayminus=means - np.asarray(ci_low, dtype=float),
                   color=NEUTRAL_GREY, thickness=1.5, width=4)
    fig = go.Figure(go.Bar(
        x=labels, y=means, error_y=err,
        marker=dict(color=[ACCENT_AMBER if (highlight is not None and l == highlight) else PRIMARY_BLUE for l in labels], line=dict(color="#ffffff", width=1)),
        hovertemplate="<b>%{x}</b><br>Mean: %{y:.2f} min<extra></extra>",
    ))
    fig.update_xaxes(tickangle=-35 if len(labels) > 6 else 0)
    return apply_light_layout(fig, title, "", y_title, height=420)


def plot_violin_by_daypart(df: pd.DataFrame) -> go.Figure:
    """Renders interactive violin plot of departure delay by Aviation Day Part."""
    sample = df[["DAY_PART", "DEP_DELAY"]].dropna()
    if len(sample) > 20000:
        sample = sample.sample(20000, random_state=42)
    sample = sample.copy()
    sample["DAY_PART"] = sample["DAY_PART"].astype(str)
    sample["DEP_DELAY_CLIPPED"] = sample["DEP_DELAY"].clip(-25, 120)
    present = list(sample["DAY_PART"].unique())
    preferred = ["Morning", "Afternoon", "Evening", "Night"]
    order = [p for p in preferred if p in present] + [p for p in present if p not in preferred]
    palette = [PRIMARY_BLUE, SECONDARY_BLUE, ACCENT_AMBER, ACCENT_INDIGO] * 3
    fig = go.Figure()
    for dp, col in zip(order, palette):
        sub = sample.loc[sample["DAY_PART"] == dp, "DEP_DELAY_CLIPPED"]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Violin(
            y=sub, name=dp, box_visible=True, meanline_visible=True,
            line_color=col, fillcolor=col, opacity=0.6,
            hovertemplate=f"<b>{dp}</b><br>Delay: %{{y:.1f}} min<extra></extra>"
        ))
    fig.add_hline(y=15, line_width=1.5, line_dash="dash", line_color=ACCENT_AMBER, annotation_text="15 min threshold", annotation_position="top left", annotation_font_color=ACCENT_AMBER)
    return apply_light_layout(fig, "Departure delay by day part (clipped to -25 to 120 min, sample of 20,000)", "Day part", "Departure delay (minutes)", height=420)


def plot_ecdf_delay(df: pd.DataFrame) -> go.Figure:
    """Renders Empirical Cumulative Distribution Function (ECDF) of departure delay."""
    s = np.sort(df["DEP_DELAY"].dropna().to_numpy())
    pct_le0 = float(np.searchsorted(s, 0, side="right") / len(s) * 100.0)
    pct_lt15 = float(np.searchsorted(s, 15, side="left") / len(s) * 100.0)
    if len(s) > 4000:
        idx = np.linspace(0, len(s) - 1, 2000).astype(int)
        x = np.clip(s[idx], -30, 180)
        y = (idx + 1) / len(s) * 100.0
    else:
        x = np.clip(s, -30, 180)
        y = np.arange(1, len(s) + 1) / len(s) * 100.0
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color=PRIMARY_BLUE, width=3),
        hovertemplate="Delay: %{x:.1f} min<br>Cumulative share of flights: %{y:.2f}%<extra></extra>",
        name="Empirical CDF"
    ))
    fig.add_vline(x=0, line_width=1.5, line_dash="dot", line_color=ACCENT_EMERALD, annotation_text=f"0 min: {pct_le0:.1f}%", annotation_position="bottom right", annotation_font_color=ACCENT_EMERALD)
    fig.add_vline(x=15, line_width=1.8, line_dash="dash", line_color=ACCENT_AMBER, annotation_text=f"15 min: {pct_lt15:.1f}% under", annotation_position="bottom right", annotation_font_color=ACCENT_AMBER)
    return apply_light_layout(fig, "Cumulative distribution of departure delay", "Departure delay (minutes, clipped to -30 to 180)", "Cumulative share of flights (%)", height=420)


def plot_qq_normality(df: pd.DataFrame) -> go.Figure:
    """Renders Normal Quantile-Quantile (Q-Q) plot showing extreme right-tail departure from Gaussianity."""
    from scipy import stats as sp_stats
    s = df["DEP_DELAY"].dropna().to_numpy()
    rng = np.random.default_rng(42)
    sub = rng.choice(s, size=min(2500, len(s)), replace=False)
    (osm, osr), (slope, intercept, r_val) = sp_stats.probplot(sub, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers",
        marker=dict(size=5, color=PRIMARY_BLUE, opacity=0.6),
        name="Empirical quantiles",
        hovertemplate="Normal quantile: %{x:.2f}<br>Observed delay: %{y:.1f} min<extra></extra>"
    ))
    x_line = np.array([float(np.min(osm)), float(np.max(osm))])
    fig.add_trace(go.Scatter(
        x=x_line, y=slope * x_line + intercept, mode="lines",
        line=dict(color=ACCENT_AMBER, width=2, dash="dash"),
        name=f"Normal reference (R² = {r_val**2:.3f})"
    ))
    return apply_light_layout(fig, "Normal Q-Q plot of departure delay (heavy right tail)", "Theoretical normal quantiles", "Observed departure delay (minutes)", show_legend=True, height=420)


def plot_vif_bars(vif_df: pd.DataFrame) -> go.Figure:
    """Plots Variance Inflation Factor (VIF) bars before and after removing collinear CRS_ELAPSED_TIME."""
    df_s = vif_df.sort_values("VIF_Full_Set", ascending=True)
    red_col = "VIF_After_Removing_CRS_ELAPSED_TIME" if "VIF_After_Removing_CRS_ELAPSED_TIME" in df_s.columns else "VIF_After_Dropping_Elapsed_Time"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_s["Feature"], x=df_s["VIF_Full_Set"], orientation="h",
        name="Full feature set", marker=dict(color=PRIMARY_BLUE),
        hovertemplate="<b>%{y}</b><br>Full VIF: %{x:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=df_s["Feature"], x=df_s[red_col].fillna(0), orientation="h",
        name="Reduced set (CRS_ELAPSED_TIME dropped)", marker=dict(color=ACCENT_EMERALD),
        hovertemplate="<b>%{y}</b><br>Reduced VIF: %{x:.2f}<extra></extra>"
    ))
    fig.add_vline(x=5.0, line_width=1.5, line_dash="dash", line_color=ACCENT_AMBER)
    fig.add_vline(x=10.0, line_width=1.5, line_dash="dot", line_color=ACCENT_ROSE)
    fig.update_layout(barmode="group")
    return apply_light_layout(fig, "Variance inflation factor (dashed: VIF 5, dotted: VIF 10)", "Variance inflation factor", "", show_legend=True, height=max(420, 40 * len(df_s) + 140))


def plot_multi_correlation_bars(corr_rank_df: pd.DataFrame) -> go.Figure:
    """Plots Pearson r, Spearman rho, and Kendall tau correlations with DEP_DELAY side by side."""
    df_s = corr_rank_df.iloc[::-1]
    p_col = "Pearson_r_DEP_DELAY" if "Pearson_r_DEP_DELAY" in df_s.columns else "Pearson_r"
    s_col = "Spearman_rho_DEP_DELAY" if "Spearman_rho_DEP_DELAY" in df_s.columns else "Spearman_rho"
    k_col = "Kendall_tau_DEP_DELAY" if "Kendall_tau_DEP_DELAY" in df_s.columns else "Kendall_tau"
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_s["Feature"], x=df_s[p_col], orientation="h", name="Pearson r (Linear)", marker=dict(color=PRIMARY_BLUE)))
    fig.add_trace(go.Bar(y=df_s["Feature"], x=df_s[s_col], orientation="h", name="Spearman ρ (Rank)", marker=dict(color=ACCENT_INDIGO)))
    fig.add_trace(go.Bar(y=df_s["Feature"], x=df_s[k_col], orientation="h", name="Kendall τ (Concordance)", marker=dict(color=ACCENT_EMERALD)))
    fig.update_layout(barmode="group")
    return apply_light_layout(fig, "Parametric & Non-Parametric Correlation with DEP_DELAY", "Correlation Coefficient", "", show_legend=True)


def plot_anova_effect_sizes(anova_df: pd.DataFrame) -> go.Figure:
    """Plots ANOVA F-statistic and Eta-Squared effect sizes across grouping factors."""
    df_s = anova_df.sort_values("Eta_Squared", ascending=True)
    f_col = "F_Statistic" if "F_Statistic" in df_s.columns else "ANOVA_F"
    fig = go.Figure(go.Bar(
        y=df_s["Factor"], x=df_s["Eta_Squared"] * 100.0, orientation="h",
        marker=dict(color=PRIMARY_BLUE, line=dict(color="#ffffff", width=1)),
        text=df_s.apply(lambda r: f"η²={r['Eta_Squared']*100:.2f}% (F={r[f_col]:,.0f})", axis=1),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Eta-Squared (η²): %{x:.3f}%<extra></extra>"
    ))
    return apply_light_layout(fig, "One-Way ANOVA Variance Explained (η² %) Across Operational Grouping Factors", "Variance Explained η² (%)", "")


def plot_twoway_interaction(cell_means: list) -> go.Figure:
    """Plots Two-Way ANOVA interaction profile (Day Part x Weekend Indicator)."""
    cdf = pd.DataFrame(cell_means)
    preferred = ["Morning (05-11)", "Afternoon (12-16)", "Evening (17-20)", "Night (21-04)", "Morning", "Afternoon", "Evening", "Night"]
    present = list(cdf["DAY_PART"].astype(str).unique())
    order = [p for p in preferred if p in present] + [p for p in present if p not in preferred]
    fig = go.Figure()
    for wk, label, col in [(0, "Weekday (Mon–Fri)", PRIMARY_BLUE), (1, "Weekend (Sat–Sun)", ACCENT_AMBER)]:
        sub = cdf[cdf["IS_WEEKEND"] == wk].set_index("DAY_PART").reindex(order).reset_index()
        fig.add_trace(go.Scatter(
            x=sub["DAY_PART"], y=sub["mean"], mode="lines+markers",
            name=label, line=dict(color=col, width=3), marker=dict(size=9, color=col),
            hovertemplate=f"<b>{label}</b><br>Day part: %{{x}}<br>Mean delay: %{{y:.2f}} min<extra></extra>"
        ))
    return apply_light_layout(fig, "Two-way ANOVA interaction: day part by weekday/weekend", "Day part", "Mean departure delay (minutes)", show_legend=True, height=420)


def plot_feature_importance_horizontal(fi_df: pd.DataFrame, val_col: str, title: str, x_title: str, color: str = PRIMARY_BLUE, top_n: int = 20) -> go.Figure:
    """Plots horizontal bar chart of feature importances or TreeSHAP values."""
    df_top = fi_df.head(top_n).iloc[::-1]
    feat_col = "Feature" if "Feature" in df_top.columns else "feature"
    fig = go.Figure(go.Bar(
        x=df_top[val_col], y=df_top[feat_col], orientation="h",
        marker=dict(color=color, line=dict(color="#ffffff", width=1)),
        text=df_top[val_col].apply(lambda v: f"{v:.3f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>" + x_title + ": %{x:.4f}<extra></extra>"
    ))
    return apply_light_layout(fig, title, x_title, "", height=max(420, top_n * 26 + 120))


def plot_error_by_delay_band(band_df: pd.DataFrame) -> go.Figure:
    """Plots MAE across flight delay severity bands for key regression architectures."""
    fig = go.Figure()
    palette = [PRIMARY_BLUE, ACCENT_EMERALD, ACCENT_INDIGO, ACCENT_AMBER, NEUTRAL_GREY]
    models = band_df["model"].unique().tolist()
    for m, col in zip(models, palette * 3):
        sub = band_df[band_df["model"] == m]
        fig.add_trace(go.Bar(
            x=sub["band"], y=sub["mae"], name=m, marker=dict(color=col),
            hovertemplate=f"<b>{m}</b><br>Band: %{{x}}<br>MAE: %{{y:.2f}} min<extra></extra>"
        ))
    fig.update_layout(barmode="group")
    return apply_light_layout(fig, "Regression Mean Absolute Error (MAE) Stratified by Actual Delay Band", "Actual Departure Delay Band", "Mean Absolute Error (minutes)", show_legend=True)



def plot_scatter_matrix(sample_df: pd.DataFrame) -> go.Figure:
    """Pairwise relationship matrix (scatter plots) on a random sample of flights."""
    cols = list(sample_df.columns)
    fig = go.Figure(go.Splom(
        dimensions=[dict(label=c.replace("_", " ").title(), values=sample_df[c]) for c in cols],
        diagonal_visible=False,
        showupperhalf=False,
        marker=dict(size=3, color=PRIMARY_BLUE, opacity=0.35, line=dict(width=0)),
    ))
    apply_light_layout(fig, "Pairwise relationships (random sample of 3,000 flights)", height=640)
    fig.update_layout(margin=dict(l=70, r=24, t=70, b=70))
    return fig


def plot_predicted_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.237,
    model_name: str = "Classifier",
) -> go.Figure:
    """Plots predicted delay probability distribution for actual On-Time vs Delayed flights."""
    fig = go.Figure()
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)

    prob_ontime = yp[yt == 0]
    prob_delayed = yp[yt == 1]

    fig.add_trace(go.Histogram(
        x=prob_ontime,
        name="Actual On-Time (0)",
        opacity=0.60,
        marker=dict(color=ACCENT_EMERALD, line=dict(color="#ffffff", width=0.5)),
        xbins=dict(start=0.0, end=1.0, size=0.025),
        hovertemplate="P(Delayed): %{x:.3f}<br>Flights: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=prob_delayed,
        name="Actual Delayed (1)",
        opacity=0.65,
        marker=dict(color=ACCENT_AMBER, line=dict(color="#ffffff", width=0.5)),
        xbins=dict(start=0.0, end=1.0, size=0.025),
        hovertemplate="P(Delayed): %{x:.3f}<br>Flights: %{y:,}<extra></extra>",
    ))

    fig.add_vline(
        x=threshold,
        line_width=2.5,
        line_dash="dash",
        line_color=ACCENT_ROSE,
        annotation_text=f"Decision Threshold τ* = {threshold:.3f}",
        annotation_position="top right",
        annotation_font_color=ACCENT_ROSE,
    )

    fig.update_layout(barmode="overlay")
    fig.update_xaxes(range=[0.0, 1.0])
    return apply_light_layout(
        fig,
        f"Predicted Delay Probability Distribution — {model_name}",
        "Predicted Delay Probability P(Delay ≥ 15 min)",
        "Flight Count",
        show_legend=True,
        height=420,
    )


def plot_roc_curve_interactive(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Classifier",
    roc_auc: float = 0.0,
) -> go.Figure:
    """Plots interactive ROC curve with AUC metric."""
    from sklearn.metrics import roc_curve, auc

    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)
    fpr, tpr, _ = roc_curve(yt, yp)
    score = auc(fpr, tpr) if roc_auc <= 0 else roc_auc

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"{model_name} (AUC = {score:.4f})",
        line=dict(color=PRIMARY_BLUE, width=2.5),
        hovertemplate="FPR: %{x:.3f}<br>TPR (Recall): %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Chance (AUC = 0.5000)",
        line=dict(color=NEUTRAL_GREY, width=1.5, dash="dash"),
    ))
    fig.update_xaxes(range=[0.0, 1.0], title="False Positive Rate (1 - Specificity)")
    fig.update_yaxes(range=[0.0, 1.05], title="True Positive Rate (Recall / Sensitivity)")
    return apply_light_layout(
        fig,
        f"Receiver Operating Characteristic (ROC) — {model_name}",
        "False Positive Rate",
        "True Positive Rate",
        show_legend=True,
        height=420,
    )


def plot_pr_curve_interactive(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Classifier",
    pr_auc: float = 0.0,
) -> go.Figure:
    """Plots interactive Precision-Recall curve with PR-AUC metric."""
    from sklearn.metrics import precision_recall_curve, auc

    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)
    precision, recall, _ = precision_recall_curve(yt, yp)
    score = auc(recall, precision) if pr_auc <= 0 else pr_auc
    baseline = float(np.mean(yt))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode="lines",
        name=f"{model_name} (PR-AUC = {score:.4f})",
        line=dict(color=ACCENT_EMERALD, width=2.5),
        hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[baseline, baseline],
        mode="lines",
        name=f"Natural Baseline Prevalence ({baseline*100:.1f}%)",
        line=dict(color=NEUTRAL_GREY, width=1.5, dash="dash"),
    ))
    fig.update_xaxes(range=[0.0, 1.0], title="Recall (Delayed Class Sensitivity)")
    fig.update_yaxes(range=[0.0, 1.05], title="Precision (Delayed Class PPV)")
    return apply_light_layout(
        fig,
        f"Precision-Recall (PR) Curve — {model_name}",
        "Recall",
        "Precision",
        show_legend=True,
        height=420,
    )

