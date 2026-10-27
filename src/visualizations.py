import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats

# Light Theme Palette Tokens
PAPER_BG = "#ffffff"
PLOT_BG = "#f8fafc"
TEXT_COLOR = "#0f172a"
SUBTEXT_COLOR = "#475569"
GRID_COLOR = "#e2e8f0"
BORDER_COLOR = "#cbd5e1"

# Curated Accent Tokens
PRIMARY_BLUE = "#2563eb"
SECONDARY_BLUE = "#0ea5e9"
ACCENT_CYAN = "#06b6d4"
ACCENT_INDIGO = "#4f46e5"
ACCENT_EMERALD = "#10b981"
ACCENT_AMBER = "#f59e0b"
ACCENT_ROSE = "#ef4444"

def apply_light_layout(
    fig: go.Figure, 
    title: str, 
    x_title: str = "", 
    y_title: str = "",
    show_legend: bool = False
) -> go.Figure:
    """Applies a crisp, modern, light-themed aviation design system to Plotly charts."""
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=15, color=TEXT_COLOR),
            x=0.01,
            y=0.96,
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", color=TEXT_COLOR),
        margin=dict(l=50, r=30, t=55, b=45),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=SUBTEXT_COLOR),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=12, color=SUBTEXT_COLOR)),
            showgrid=True,
            gridcolor=GRID_COLOR,
            gridwidth=1,
            linecolor=BORDER_COLOR,
            tickfont=dict(size=11, color=SUBTEXT_COLOR),
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=12, color=SUBTEXT_COLOR)),
            showgrid=True,
            gridcolor=GRID_COLOR,
            gridwidth=1,
            linecolor=BORDER_COLOR,
            tickfont=dict(size=11, color=SUBTEXT_COLOR),
            zeroline=False,
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_size=12,
            font_family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            bordercolor=BORDER_COLOR,
        ),
    )
    return fig

def plot_delay_donut(delayed_count: int, ontime_count: int) -> go.Figure:
    """Renders a clean donut chart comparing delayed vs on-time flights."""
    labels = ["On-Time / Minor Buffer (< 15m)", "Delayed (≥ 15m)"]
    values = [ontime_count, delayed_count]
    colors = [ACCENT_EMERALD, ACCENT_ROSE]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
            textinfo="percent+label",
            textposition="outside",
            pull=[0, 0.05],
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Proportion: %{percent}<extra></extra>",
        )
    ])
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        margin=dict(t=40, b=30, l=30, r=30),
        title=dict(text="<b>Delayed vs. On-Time Flights (15-min Threshold)</b>", font=dict(size=15, color=TEXT_COLOR)),
        showlegend=False,
        annotations=[dict(text=f"<b>Total</b><br>{delayed_count + ontime_count:,}", x=0.5, y=0.5, font_size=14, showarrow=False, font_color=TEXT_COLOR)]
    )
    return fig

def plot_delay_histogram(df: pd.DataFrame) -> go.Figure:
    """Plots delay distribution histogram highlighting the 15-minute delayed threshold."""
    delays = df["DEP_DELAY"].dropna()
    clipped = np.clip(delays, -40, 120)
    
    # Pre-bin for speed and responsive rendering
    counts, bin_edges = np.histogram(clipped, bins=32, range=(-40, 120))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bar_colors = [ACCENT_ROSE if x >= 15 else ACCENT_EMERALD if x < 0 else PRIMARY_BLUE for x in bin_centers]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        width=4.5,
        marker=dict(color=bar_colors, line=dict(color="#ffffff", width=1)),
        hovertemplate="Departure Delay: ~%{x:.0f} min<br>Flight Volume: %{y:,}<extra></extra>",
    ))
    
    fig.add_vline(
        x=15, line_width=2, line_dash="dash", line_color=ACCENT_AMBER, 
        annotation_text="15m FAA Delayed Threshold", 
        annotation_position="top right",
        annotation_font=dict(color=ACCENT_AMBER, size=11, family="sans-serif")
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
        marker=dict(color=ACCENT_ROSE, size=3, opacity=0.5),
        line=dict(color=PRIMARY_BLUE, width=1.5),
        fillcolor="rgba(37, 99, 235, 0.12)",
        hovertemplate="Delay: %{x} min<extra></extra>",
    ))
    return apply_light_layout(fig, "Five-Number Summary & Extreme Delay Outliers Boxplot", "Departure Delay (minutes)", "")

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
        fillcolor="rgba(37, 99, 235, 0.15)",
        hovertemplate="Delay: %{x:.1f} min<br>Probability Density: %{y:.5f}<extra></extra>",
    ))
    fig.add_vline(x=15, line_width=2, line_dash="dash", line_color=ACCENT_AMBER, annotation_text="15m Delayed Threshold", annotation_font_color=ACCENT_AMBER)
    return apply_light_layout(fig, "Kernel Density Estimate (KDE) Distribution Curve", "Departure Delay (minutes)", "Probability Density")

def plot_delay_categories_bar(cat_df: pd.DataFrame) -> go.Figure:
    """Plots delay category distribution."""
    colors = [ACCENT_EMERALD, PRIMARY_BLUE, SECONDARY_BLUE, ACCENT_AMBER, ACCENT_ROSE]
    fig = go.Figure(data=[
        go.Bar(
            x=cat_df["Category"],
            y=cat_df["Flight Count"],
            text=cat_df["Percentage"].apply(lambda p: f"{p:.1f}%"),
            textposition="outside",
            marker=dict(color=colors, line=dict(color="#ffffff", width=1)),
            hovertemplate="<b>%{x}</b><br>Definition: " + cat_df["Range Definition"] + "<br>Flights: %{y:,}<br>Share: " + cat_df["Percentage"].astype(str) + "%<extra></extra>",
        )
    ])
    return apply_light_layout(fig, "Delay Category Distribution Breakdown", "Operational Category", "Flight Count")

def plot_airline_bar(airline_df: pd.DataFrame) -> go.Figure:
    """Plots horizontal bar chart of average departure delay by airline."""
    df_sorted = airline_df.sort_values("Mean Delay (min)", ascending=True)
    colors = [ACCENT_ROSE if x > 15 else ACCENT_AMBER if x > 10 else PRIMARY_BLUE for x in df_sorted["Mean Delay (min)"]]
    
    fig = go.Figure(go.Bar(
        x=df_sorted["Mean Delay (min)"],
        y=df_sorted["Airline"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="#ffffff", width=1)),
        text=df_sorted["Mean Delay (min)"].apply(lambda x: f"{x:.1f} min"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Mean Delay: %{x:.2f} min<extra></extra>",
    ))
    return apply_light_layout(fig, "Airline Performance: Average Departure Delay", "Mean Departure Delay (minutes)", "")

def plot_airline_delayed_pct(airline_df: pd.DataFrame) -> go.Figure:
    """Plots horizontal bar chart of delayed percentage by airline."""
    df_sorted = airline_df.sort_values("Delayed %", ascending=True)
    fig = go.Figure(go.Bar(
        x=df_sorted["Delayed %"],
        y=df_sorted["Airline"],
        orientation="h",
        marker=dict(color=SECONDARY_BLUE, line=dict(color="#ffffff", width=1)),
        text=df_sorted["Delayed %"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Delayed Rate (≥15m): %{x:.1f}%<extra></extra>",
    ))
    return apply_light_layout(fig, "Airline Delay Frequency: % of Flights Delayed ≥ 15 Minutes", "Delayed Flights (%)", "")

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
    return apply_light_layout(fig, title, "Mean Departure Delay (minutes)", "")

def plot_airport_volume_vs_delay(airport_df: pd.DataFrame, airport_col: str) -> go.Figure:
    """Plots scatter of flight volume versus mean departure delay."""
    fig = go.Figure(go.Scatter(
        x=airport_df["Flight Count"],
        y=airport_df["Mean Delay (min)"],
        mode="markers+text",
        text=airport_df[airport_col],
        textposition="top center",
        marker=dict(
            size=10,
            color=airport_df["Delayed %"],
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Delayed %", tickfont=dict(size=10)),
            line=dict(color=BORDER_COLOR, width=1),
        ),
        hovertemplate="<b>Airport: %{text}</b><br>Flights: %{x:,}<br>Mean Delay: %{y:.1f} min<br>Delayed Rate: %{marker.color:.1f}%<extra></extra>",
    ))
    return apply_light_layout(fig, f"{airport_col}: Flight Volume vs Average Delay Duration", "Total Flights", "Mean Delay (minutes)")

def plot_top_routes_bar(route_df: pd.DataFrame, title: str, ascending: bool = False, top_n: int = 12) -> go.Figure:
    """Plots top or lowest delayed routes."""
    df_sorted = route_df.sort_values("Mean Delay (min)", ascending=ascending).head(top_n)
    if not ascending:
        df_sorted = df_sorted.iloc[::-1]
    color = ACCENT_ROSE if not ascending else ACCENT_EMERALD
    
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
        line=dict(color=ACCENT_AMBER, width=3),
        marker=dict(size=8, color=ACCENT_AMBER),
        hovertemplate="Month: %{x}<br>Mean Delay: %{y:.2f} min<extra></extra>",
    ))
    return apply_light_layout(fig, "Seasonality: Monthly Average Departure Delay Trend", "Month", "Mean Delay (minutes)")

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
    fig = go.Figure(go.Bar(
        x=daypart_df["Day Part"],
        y=daypart_df["Mean Delay (min)"],
        marker=dict(color=[SECONDARY_BLUE, PRIMARY_BLUE, ACCENT_ROSE, ACCENT_INDIGO], line=dict(color="#ffffff", width=1)),
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
    
    text_annotations = np.round(z, 3).astype(str)
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"size": 11, "family": "sans-serif"},
        colorbar=dict(title="Pearson r", tickfont=dict(size=10)),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation (r): %{z:.4f}<extra></extra>",
    ))
    return apply_light_layout(fig, "Pearson Correlation Matrix Heatmap", "", "")

def plot_confusion_matrix(cm: list, class_names: list = None) -> go.Figure:
    """Plots clean annotated confusion matrix heatmap."""
    if class_names is None:
        class_names = ["On-Time (< 15m)", "Delayed (≥ 15m)"]
    z = np.array(cm)
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=class_names,
        y=class_names,
        colorscale="Blues",
        text=z,
        texttemplate="<b>%{text:,}</b>",
        textfont={"size": 15, "color": "#0f172a"},
        colorbar=dict(title="Count"),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>",
    ))
    return apply_light_layout(fig, "Confusion Matrix Heatmap", "Predicted Label", "Actual Ground Truth")

def plot_model_metric_bars(models_df: pd.DataFrame, metric_col: str, title: str, y_title: str) -> go.Figure:
    """Generic bar chart comparing model metrics."""
    fig = go.Figure(go.Bar(
        x=models_df["Model"],
        y=models_df[metric_col],
        marker=dict(color=PRIMARY_BLUE, line=dict(color="#ffffff", width=1)),
        text=models_df[metric_col].apply(lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)),
        textposition="outside",
        hovertemplate="Model: %{x}<br>" + metric_col + ": %{y}<extra></extra>",
    ))
    return apply_light_layout(fig, title, "Algorithm", y_title)

def plot_actual_vs_predicted_scatter(y_true: np.ndarray, y_pred: np.ndarray, sample_n: int = 3000) -> go.Figure:
    """Renders actual vs predicted scatter plot with 45-degree identity reference line."""
    if len(y_true) > sample_n:
        idx = np.random.RandomState(42).choice(len(y_true), sample_n, replace=False)
        yt, yp = y_true[idx], y_pred[idx]
    else:
        yt, yp = y_true, y_pred
        
    min_val = min(float(np.min(yt)), float(np.min(yp)), -20.0)
    max_val = max(float(np.quantile(yt, 0.99)), float(np.quantile(yp, 0.99)), 120.0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yt,
        y=yp,
        mode="markers",
        marker=dict(size=4, color=PRIMARY_BLUE, opacity=0.35),
        hovertemplate="Actual: %{x:.1f} min<br>Predicted: %{y:.1f} min<extra></extra>",
    ))
    # Identity line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color=ACCENT_ROSE, width=2, dash="dash"),
        name="Perfect Identity (y=x)",
    ))
    return apply_light_layout(fig, "Actual vs. Predicted Departure Delay Scatter Plot", "Actual Delay (minutes)", "Predicted Delay (minutes)", show_legend=True)

def plot_residuals_distribution(residuals: np.ndarray) -> go.Figure:
    """Plots histogram of model residuals."""
    clipped = np.clip(residuals, -60, 60)
    counts, bin_edges = np.histogram(clipped, bins=35, range=(-60, 60))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    fig = go.Figure(go.Bar(
        x=bin_centers,
        y=counts,
        marker=dict(color=SECONDARY_BLUE, line=dict(color="#ffffff", width=1)),
        hovertemplate="Residual: %{x:.1f} min<br>Count: %{y:,}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=2, line_dash="solid", line_color=ACCENT_ROSE, annotation_text="Zero Error Line", annotation_font_color=ACCENT_ROSE)
    return apply_light_layout(fig, "Residual Distribution Histogram (Error = Actual - Predicted)", "Residual Error (minutes)", "Count")

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
        marker=dict(size=4, color=ACCENT_INDIGO, opacity=0.35),
        hovertemplate="Predicted: %{x:.1f} min<br>Residual Error: %{y:.1f} min<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=2, line_dash="dash", line_color=ACCENT_ROSE)
    return apply_light_layout(fig, "Residuals vs. Predicted Values Plot", "Predicted Delay (minutes)", "Residual Error (minutes)")
