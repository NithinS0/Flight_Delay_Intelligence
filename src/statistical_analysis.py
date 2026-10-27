import pandas as pd
import numpy as np
from typing import Dict, Any, List

def compute_dataset_overview(df: pd.DataFrame, raw_df: pd.DataFrame = None, summary_meta: dict = None) -> Dict[str, Any]:
    """Computes Section A Dataset Overview metrics dynamically from the actual dataset."""
    total_rows = len(df)
    total_cols = len(df.columns)
    
    min_date = df["FL_DATE"].min()
    max_date = df["FL_DATE"].max()
    date_str = f"{min_date.strftime('%d %b %Y')} – {max_date.strftime('%d %b %Y')}" if pd.notnull(min_date) else "Unknown"
    
    airlines_count = df["AIRLINE"].nunique()
    origins_count = df["ORIGIN"].nunique()
    dest_count = df["DEST"].nunique()
    
    # Calculate unique routes
    if "ROUTE" in df.columns:
        routes_count = df["ROUTE"].nunique()
    else:
        routes_count = df.groupby(["ORIGIN", "DEST"]).ngroups
        
    missing_sum = int(df.isna().sum().sum())
    duplicates_count = summary_meta.get("initial_duplicates", 0) if summary_meta else int(df.duplicated().sum())
    
    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "date_range": date_str,
        "airlines_count": airlines_count,
        "origins_count": origins_count,
        "dest_count": dest_count,
        "routes_count": routes_count,
        "missing_values": missing_sum,
        "duplicates_count": duplicates_count,
    }

def compute_descriptive_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes Section B Comprehensive descriptive statistics for departure delay.
    Includes Mean, Median, Mode, Std, Variance, Min, Max, Q1, Q2, Q3, IQR, Skewness, Kurtosis.
    """
    delays = df["DEP_DELAY"].dropna()
    total_count = len(delays)
    
    mean_val = float(delays.mean())
    median_val = float(delays.median())
    mode_series = delays.mode()
    mode_val = float(mode_series.iloc[0]) if not mode_series.empty else median_val
    std_val = float(delays.std())
    var_val = float(delays.var())
    min_val = float(delays.min())
    max_val = float(delays.max())
    
    q1 = float(delays.quantile(0.25))
    q2 = median_val
    q3 = float(delays.quantile(0.75))
    iqr = q3 - q1
    skewness = float(delays.skew())
    kurtosis_val = float(delays.kurtosis())
    
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers_count = int(((delays < lower_fence) | (delays > upper_fence)).sum())
    outlier_pct = float((outliers_count / total_count) * 100) if total_count > 0 else 0.0
    
    delayed_count = int((delays >= 15.0).sum())
    non_delayed_count = int((delays < 15.0).sum())
    
    delayed_pct = float((delayed_count / total_count) * 100) if total_count > 0 else 0.0
    non_delayed_pct = float((non_delayed_count / total_count) * 100) if total_count > 0 else 0.0
    
    return {
        "count": total_count,
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "mode": round(mode_val, 4),
        "std": round(std_val, 4),
        "variance": round(var_val, 3),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "q1": round(q1, 4),
        "q2": round(q2, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis_val, 4),
        "delayed_flights": delayed_count,
        "non_delayed_flights": non_delayed_count,
        "delayed_percentage": round(delayed_pct, 2),
        "non_delayed_percentage": round(non_delayed_pct, 2),
        "outlier_count": outliers_count,
        "outlier_percentage": round(outlier_pct, 2),
        "lower_fence": round(lower_fence, 2),
        "upper_fence": round(upper_fence, 2),
    }

def compute_delay_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes delay category distribution:
    - Early departure (< -5 min)
    - On-time or minor delay (-5 to < 15 min)
    - Moderate delay (15 to < 60 min)
    - High delay (60 to < 120 min)
    - Severe delay (>= 120 min)
    """
    delays = df["DEP_DELAY"].dropna()
    total = len(delays)
    
    early = int((delays < -5).sum())
    ontime = int(((delays >= -5) & (delays < 15)).sum())
    moderate = int(((delays >= 15) & (delays < 60)).sum())
    high = int(((delays >= 60) & (delays < 120)).sum())
    severe = int((delays >= 120).sum())
    
    categories = [
        {"Category": "Early Departure", "Range Definition": "DEP_DELAY < -5 min", "Flight Count": early, "Percentage": round(early/total*100, 2)},
        {"Category": "On-Time / Minor Delay", "Range Definition": "-5 min ≤ DEP_DELAY < 15 min", "Flight Count": ontime, "Percentage": round(ontime/total*100, 2)},
        {"Category": "Moderate Delay", "Range Definition": "15 min ≤ DEP_DELAY < 60 min", "Flight Count": moderate, "Percentage": round(moderate/total*100, 2)},
        {"Category": "High Delay", "Range Definition": "60 min ≤ DEP_DELAY < 120 min", "Flight Count": high, "Percentage": round(high/total*100, 2)},
        {"Category": "Severe Delay", "Range Definition": "DEP_DELAY ≥ 120 min", "Flight Count": severe, "Percentage": round(severe/total*100, 2)},
    ]
    return pd.DataFrame(categories)

def compute_airline_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Computes airline performance rankings and delay summaries."""
    grouped = df.groupby("AIRLINE")
    records = []
    for airline, group in grouped:
        delays = group["DEP_DELAY"]
        flight_count = len(group)
        delayed_count = int((delays >= 15).sum())
        delayed_pct = float((delayed_count / flight_count) * 100) if flight_count > 0 else 0.0
        
        records.append({
            "Airline": str(airline),
            "Flight Count": flight_count,
            "Mean Delay (min)": round(float(delays.mean()), 2),
            "Median Delay (min)": round(float(delays.median()), 2),
            "Std Deviation": round(float(delays.std()), 2),
            "Delayed Flights": delayed_count,
            "Delayed %": round(delayed_pct, 2),
        })
    res_df = pd.DataFrame(records).sort_values("Mean Delay (min)", ascending=False).reset_index(drop=True)
    return res_df

def compute_airport_analysis(df: pd.DataFrame, min_flights: int = 50) -> Dict[str, pd.DataFrame]:
    """Computes Origin and Destination airport rankings with a configurable minimum flight threshold."""
    def analyze_col(col_name: str, label: str) -> pd.DataFrame:
        grouped = df.groupby(col_name)
        res = []
        for airport, group in grouped:
            count = len(group)
            if count >= min_flights:
                delays = group["DEP_DELAY"]
                delayed_count = int((delays >= 15).sum())
                delayed_pct = float((delayed_count / count) * 100)
                res.append({
                    label: str(airport),
                    "Flight Count": count,
                    "Mean Delay (min)": round(float(delays.mean()), 2),
                    "Median Delay (min)": round(float(delays.median()), 2),
                    "Delayed Flights": delayed_count,
                    "Delayed %": round(delayed_pct, 2),
                })
        return pd.DataFrame(res).sort_values("Mean Delay (min)", ascending=False).reset_index(drop=True)
        
    return {
        "origins": analyze_col("ORIGIN", "Origin Airport"),
        "destinations": analyze_col("DEST", "Destination Airport"),
    }

def compute_route_analysis(df: pd.DataFrame, min_flights: int = 30) -> pd.DataFrame:
    """Computes route performance rankings for origin-destination pairs."""
    df_copy = df.copy()
    if "ROUTE" not in df_copy.columns:
        df_copy["ROUTE"] = df_copy["ORIGIN"].astype(str) + " → " + df_copy["DEST"].astype(str)
    else:
        df_copy["ROUTE"] = df_copy["ROUTE"].astype(str).str.replace("_", " → ")
        
    grouped = df_copy.groupby("ROUTE")
    res = []
    for route, group in grouped:
        count = len(group)
        if count >= min_flights:
            delays = group["DEP_DELAY"]
            delayed_count = int((delays >= 15).sum())
            res.append({
                "Route Corridor": str(route),
                "Flight Count": count,
                "Mean Delay (min)": round(float(delays.mean()), 2),
                "Median Delay (min)": round(float(delays.median()), 2),
                "Delayed Flights": delayed_count,
                "Delayed %": round(float((delayed_count / count) * 100), 2),
            })
            
    return pd.DataFrame(res).sort_values("Mean Delay (min)", ascending=False).reset_index(drop=True)

def compute_temporal_analysis(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Computes temporal delay trends by hour, day of week, month, year, and day part."""
    df_temp = df.copy()
    if "SCHEDULED_HOUR" not in df_temp.columns:
        crs = df_temp["CRS_DEP_TIME"].astype(int)
        df_temp["SCHEDULED_HOUR"] = crs // 100
        
    if "FL_DATE" in df_temp.columns and not pd.api.types.is_datetime64_any_dtype(df_temp["FL_DATE"]):
        df_temp["FL_DATE"] = pd.to_datetime(df_temp["FL_DATE"])
        
    if "DAY_OF_WEEK" not in df_temp.columns and "FL_DATE" in df_temp.columns:
        df_temp["DAY_OF_WEEK"] = df_temp["FL_DATE"].dt.dayofweek
    if "MONTH" not in df_temp.columns and "FL_DATE" in df_temp.columns:
        df_temp["MONTH"] = df_temp["FL_DATE"].dt.month
    if "YEAR" not in df_temp.columns and "FL_DATE" in df_temp.columns:
        df_temp["YEAR"] = df_temp["FL_DATE"].dt.year
        
    # By Hour
    hourly = []
    for hour, grp in df_temp.groupby("SCHEDULED_HOUR"):
        delays = grp["DEP_DELAY"]
        c = len(grp)
        hourly.append({
            "Hour": int(hour),
            "Hour Label": f"{int(hour):02d}:00",
            "Flight Count": c,
            "Mean Delay (min)": round(float(delays.mean()), 2),
            "Median Delay (min)": round(float(delays.median()), 2),
            "Delayed %": round(float(((delays >= 15).sum() / c) * 100), 2) if c > 0 else 0.0,
        })
    df_hourly = pd.DataFrame(hourly).sort_values("Hour").reset_index(drop=True)
    
    # By Day of Week
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = []
    for dow, grp in df_temp.groupby("DAY_OF_WEEK"):
        delays = grp["DEP_DELAY"]
        c = len(grp)
        daily.append({
            "Day Index": int(dow),
            "Day of Week": day_names[int(dow)] if int(dow) < 7 else str(dow),
            "Flight Count": c,
            "Mean Delay (min)": round(float(delays.mean()), 2),
            "Median Delay (min)": round(float(delays.median()), 2),
            "Delayed %": round(float(((delays >= 15).sum() / c) * 100), 2),
        })
    df_daily = pd.DataFrame(daily).sort_values("Day Index").reset_index(drop=True)
    
    # By Month
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = []
    for m, grp in df_temp.groupby("MONTH"):
        delays = grp["DEP_DELAY"]
        c = len(grp)
        monthly.append({
            "Month Index": int(m),
            "Month": month_names[int(m) - 1] if 1 <= int(m) <= 12 else str(m),
            "Flight Count": c,
            "Mean Delay (min)": round(float(delays.mean()), 2),
            "Median Delay (min)": round(float(delays.median()), 2),
            "Delayed %": round(float(((delays >= 15).sum() / c) * 100), 2),
        })
    df_monthly = pd.DataFrame(monthly).sort_values("Month Index").reset_index(drop=True)
    
    # By Year
    yearly = []
    for yr, grp in df_temp.groupby("YEAR"):
        delays = grp["DEP_DELAY"]
        c = len(grp)
        yearly.append({
            "Year": int(yr),
            "Flight Count": c,
            "Mean Delay (min)": round(float(delays.mean()), 2),
            "Delayed %": round(float(((delays >= 15).sum() / c) * 100), 2),
        })
    df_yearly = pd.DataFrame(yearly).sort_values("Year").reset_index(drop=True)
    
    # By Day Part
    if "DAY_PART" in df_temp.columns:
        day_parts = []
        for dp, grp in df_temp.groupby("DAY_PART"):
            delays = grp["DEP_DELAY"]
            c = len(grp)
            day_parts.append({
                "Day Part": str(dp),
                "Flight Count": c,
                "Mean Delay (min)": round(float(delays.mean()), 2),
                "Delayed %": round(float(((delays >= 15).sum() / c) * 100), 2),
            })
        order = {"Night": 0, "Morning": 1, "Afternoon": 2, "Evening": 3}
        df_daypart = pd.DataFrame(day_parts)
        df_daypart["sort_key"] = df_daypart["Day Part"].map(order).fillna(99)
        df_daypart = df_daypart.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
    else:
        df_daypart = pd.DataFrame()
        
    return {
        "hourly": df_hourly,
        "daily": df_daily,
        "monthly": df_monthly,
        "yearly": df_yearly,
        "daypart": df_daypart,
    }

def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Pearson correlation for numerical features."""
    df_corr = df.copy()
    if "FL_DATE" in df_corr.columns and not pd.api.types.is_datetime64_any_dtype(df_corr["FL_DATE"]):
        df_corr["FL_DATE"] = pd.to_datetime(df_corr["FL_DATE"])
    if "SCHEDULED_HOUR" not in df_corr.columns:
        crs = df_corr["CRS_DEP_TIME"].astype(int)
        df_corr["SCHEDULED_HOUR"] = crs // 100
        df_corr["SCHEDULED_MINUTE"] = crs % 100
    if "DAY_OF_WEEK" not in df_corr.columns:
        df_corr["DAY_OF_WEEK"] = df_corr["FL_DATE"].dt.dayofweek
    if "MONTH" not in df_corr.columns:
        df_corr["MONTH"] = df_corr["FL_DATE"].dt.month
        
    corr_cols = [
        "DEP_DELAY",
        "DISTANCE",
        "CRS_ELAPSED_TIME",
        "SCHEDULED_HOUR",
        "SCHEDULED_MINUTE",
        "DAY_OF_WEEK",
        "MONTH"
    ]
    valid_cols = [c for c in corr_cols if c in df_corr.columns]
    return df_corr[valid_cols].corr()

def generate_dynamic_insights(df: pd.DataFrame, airline_df: pd.DataFrame, airport_dict: Dict[str, pd.DataFrame], route_df: pd.DataFrame, temporal_dict: Dict[str, pd.DataFrame]) -> List[Dict[str, str]]:
    """Automatically generates Section I concise dynamic analytical insights from computed data."""
    insights = []
    
    # 1. Airline with highest delay
    if not airline_df.empty:
        highest_airline = airline_df.iloc[0]
        lowest_airline = airline_df.iloc[-1]
        insights.append({
            "title": "Highest Delay Airline",
            "stat": f"{highest_airline['Airline']}: {highest_airline['Mean Delay (min)']:.1f} min avg",
            "desc": f"Exhibits highest average departure delay with {highest_airline['Delayed %']:.1f}% of flights delayed ≥ 15 mins."
        })
        insights.append({
            "title": "Most Punctual Airline",
            "stat": f"{lowest_airline['Airline']}: {lowest_airline['Mean Delay (min)']:.1f} min avg",
            "desc": f"Demonstrates lowest average departure delay with only {lowest_airline['Delayed %']:.1f}% delayed flights."
        })
        
    # 2. Origin airport with highest delay
    origins = airport_dict.get("origins", pd.DataFrame())
    if not origins.empty:
        top_origin = origins.iloc[0]
        insights.append({
            "title": "Most Delayed Origin Airport",
            "stat": f"{top_origin['Origin Airport']}: {top_origin['Mean Delay (min)']:.1f} min avg",
            "desc": f"Highest departure delay among airports meeting sample volume ({top_origin['Flight Count']:,} flights, {top_origin['Delayed %']:.1f}% delayed)."
        })
        
    # 3. Most delayed departure hour
    hourly = temporal_dict.get("hourly", pd.DataFrame())
    if not hourly.empty:
        peak_hour = hourly.sort_values("Mean Delay (min)", ascending=False).iloc[0]
        min_hour = hourly.sort_values("Mean Delay (min)", ascending=True).iloc[0]
        insights.append({
            "title": "Peak Disruption Departure Hour",
            "stat": f"{peak_hour['Hour Label']} ({peak_hour['Mean Delay (min)']:.1f} min avg)",
            "desc": f"Late afternoon/evening diurnal cascading congestion peaks around {peak_hour['Hour Label']} ({peak_hour['Delayed %']:.1f}% delayed)."
        })
        insights.append({
            "title": "Most Punctual Departure Window",
            "stat": f"{min_hour['Hour Label']} ({min_hour['Mean Delay (min)']:.1f} min avg)",
            "desc": f"Early morning flights experience the lowest delays due to clean overnight gate turnarounds."
        })
        
    # 4. Month with highest average delay
    monthly = temporal_dict.get("monthly", pd.DataFrame())
    if not monthly.empty:
        peak_month = monthly.sort_values("Mean Delay (min)", ascending=False).iloc[0]
        insights.append({
            "title": "Highest Delay Month",
            "stat": f"{peak_month['Month']} ({peak_month['Mean Delay (min)']:.1f} min avg)",
            "desc": f"Summer peak travel combined with convective thunderstorm systems elevates nationwide departure delays."
        })
        
    # 5. Most frequently travelled route
    if not route_df.empty:
        busiest_route = route_df.sort_values("Flight Count", ascending=False).iloc[0]
        insights.append({
            "title": "Most Frequent Route Corridor",
            "stat": f"{busiest_route['Route Corridor']} ({busiest_route['Flight Count']:,} flights)",
            "desc": f"Average delay on this top corridor is {busiest_route['Mean Delay (min)']:.1f} minutes with {busiest_route['Delayed %']:.1f}% delayed."
        })
        
    return insights
