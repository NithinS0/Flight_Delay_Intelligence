"""Step 2: exploratory data analysis -> statistics JSON + presentation charts (all values computed from the data)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from common import *

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 220, "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
    "axes.titlesize": 14, "axes.labelsize": 11, "legend.frameon": False,
})
BLUE, CYAN, AMBER, GREEN, RED, SLATE, PURPLE = "#2563eb", "#06b6d4", "#f59e0b", "#10b981", "#ef4444", "#64748b", "#7c3aed"

df = pd.read_parquet(MODEL_TABLE, columns=["FL_DATE", "AIRLINE", "ORIGIN", "DEST", "ROUTE", "DEP_DELAY", "DELAYED", "DIVERTED",
                                           "DISTANCE", "CRS_ELAPSED_TIME", "SCHEDULED_HOUR", "DAY_OF_WEEK", "MONTH", "YEAR",
                                           "IS_WEEKEND", "HOLIDAY_DIST", "SPLIT", "DAY_PART", "TIME_SIN", "TIME_COS"])
for c in ["AIRLINE", "ORIGIN", "DEST", "ROUTE", "DAY_PART"]:
    df[c] = df[c].astype(str)
n = len(df)
y = df["DEP_DELAY"].astype(float)
S = {}

# ---------- descriptive statistics ----------
q1, q2, q3 = y.quantile([.25, .5, .75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
mode_val = float(y.round().mode().iloc[0])
S["delay_descriptive"] = {
    "count": n, "mean": y.mean(), "median": q2, "mode_rounded_minutes": mode_val, "std": y.std(), "variance": y.var(),
    "min": y.min(), "max": y.max(), "q1": q1, "q3": q3, "iqr": iqr, "skewness": stats.skew(y), "kurtosis_excess": stats.kurtosis(y),
    "tukey_lower_fence": lo, "tukey_upper_fence": hi, "n_outliers_upper": int((y > hi).sum()), "n_outliers_lower": int((y < lo).sum()),
    "pct_outliers": float(((y > hi) | (y < lo)).mean() * 100),
    "pct_gt_60": float((y > 60).mean() * 100), "pct_gt_120": float((y > 120).mean() * 100), "pct_gt_300": float((y > 300).mean() * 100),
    "p90": y.quantile(.9), "p95": y.quantile(.95), "p99": y.quantile(.99),
    "pct_early_departures": float((y < 0).mean() * 100),
}
S["class_distribution"] = {
    "overall": {"on_time": int((df.DELAYED == 0).sum()), "delayed": int((df.DELAYED == 1).sum()), "delayed_pct": float(df.DELAYED.mean() * 100)},
    **{sp: {"rows": int((df.SPLIT == sp).sum()), "delayed": int(df.loc[df.SPLIT == sp, "DELAYED"].sum()),
            "delayed_pct": float(df.loc[df.SPLIT == sp, "DELAYED"].mean() * 100),
            "mean_delay": float(y[df.SPLIT == sp].mean()), "median_delay": float(y[df.SPLIT == sp].median())} for sp in ["fit", "val", "test"]},
}
bands = pd.cut(y, [-np.inf, 0, 15, 60, 120, np.inf], labels=["Early (<0)", "On-time/minor (0-15)", "Moderate (15-60)", "High (60-120)", "Severe (>120)"], right=False)
S["delay_bands"] = {k: {"count": int(v), "pct": float(v / n * 100)} for k, v in bands.value_counts().sort_index().items()}
S["diverted_retained"] = int(df.DIVERTED.sum())

# ---------- groupings ----------
def grp(col, min_n=1):
    g = df.groupby(col, observed=True).agg(flights=("DEP_DELAY", "size"), mean_delay=("DEP_DELAY", "mean"),
                                          median_delay=("DEP_DELAY", "median"), delayed_pct=("DELAYED", lambda s: s.mean() * 100))
    return g[g.flights >= min_n]

air = grp("AIRLINE").sort_values("delayed_pct", ascending=False)
apt = grp("ORIGIN", 5000).sort_values("delayed_pct", ascending=False)
rte = grp("ROUTE", 1500).sort_values("delayed_pct", ascending=False)
mon = grp("MONTH"); yr = grp("YEAR"); hr = grp("SCHEDULED_HOUR"); dow = grp("DAY_OF_WEEK")
ym = df.assign(YM=df.FL_DATE.dt.to_period("M")).groupby("YM").agg(delayed_pct=("DELAYED", lambda s: s.mean() * 100), mean_delay=("DEP_DELAY", "mean"), flights=("DEP_DELAY", "size"))
S["airlines"] = air.round(3).reset_index().to_dict("records")
S["origin_airports_min5000"] = {"n_airports": len(apt), "top10_most_delayed": apt.head(10).round(3).reset_index().to_dict("records"),
                                "top10_most_punctual": apt.tail(10).iloc[::-1].round(3).reset_index().to_dict("records")}
S["routes_min1500"] = {"n_routes": len(rte), "top10_most_delayed": rte.head(10).round(3).reset_index().to_dict("records"),
                       "top10_most_punctual": rte.tail(10).iloc[::-1].round(3).reset_index().to_dict("records")}
S["monthly"] = mon.round(3).reset_index().to_dict("records")
S["yearly"] = yr.round(3).reset_index().to_dict("records")
S["hourly"] = hr.round(3).reset_index().to_dict("records")
S["day_of_week"] = dow.round(3).reset_index().to_dict("records")
S["day_part"] = grp("DAY_PART").round(3).reset_index().to_dict("records")
S["weekend_vs_weekday"] = grp("IS_WEEKEND").round(3).reset_index().to_dict("records")

# ---------- correlations ----------
num = df[["DEP_DELAY", "DELAYED", "DISTANCE", "CRS_ELAPSED_TIME", "SCHEDULED_HOUR", "DAY_OF_WEEK", "MONTH", "YEAR", "IS_WEEKEND", "HOLIDAY_DIST"]].astype(float)
corr = num.corr(method="pearson")
S["correlation_with_dep_delay"] = corr["DEP_DELAY"].drop("DEP_DELAY").round(4).to_dict()
S["correlation_with_delayed"] = corr["DELAYED"].drop("DELAYED").round(4).to_dict()
hist = pd.read_parquet(MODEL_TABLE, columns=["DELAYED", "DEP_DELAY", "H_AIRLINE_RATE_ALL", "H_ORIGIN_RATE_30D", "H_ROUTE_RATE_ALL",
                                              "H_ORIGINHOUR_RATE_ALL", "H_GLOBAL_RATE_1D", "H_GLOBAL_RATE_7D", "H_DEST_RATE_30D"])
S["history_feature_correlation_with_delayed"] = hist.corr()["DELAYED"].drop(["DELAYED"]).round(4).to_dict()
S["history_feature_correlation_with_dep_delay"] = hist.corr()["DEP_DELAY"].drop(["DEP_DELAY"]).round(4).to_dict()
save_json(S, METRICS_DIR / "eda_stats.json")
print("stats saved")

def save(fig, name):
    fig.tight_layout()
    fig.savefig(CHART_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)

# ---------- charts ----------
# 1 distribution
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
yc = y[(y > -40) & (y < 180)]
ax[0].hist(yc, bins=110, color=BLUE, alpha=.85)
ax[0].axvline(15, color=RED, ls="--", lw=1.5, label="Delay threshold (15 min)")
ax[0].axvline(y.median(), color=AMBER, ls="-", lw=1.5, label=f"Median = {y.median():.0f} min")
ax[0].axvline(y.mean(), color=GREEN, ls="-", lw=1.5, label=f"Mean = {y.mean():.1f} min")
ax[0].set_title("Departure Delay Distribution (-40 to 180 min)"); ax[0].set_xlabel("DEP_DELAY (minutes)"); ax[0].set_ylabel("Flights"); ax[0].legend()
ax[1].hist(y[y > 15], bins=np.logspace(np.log10(15), np.log10(y.max()), 60), color=PURPLE, alpha=.85)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_title("Right Tail of Delays (log-log, DEP_DELAY > 15 min)")
ax[1].set_xlabel("DEP_DELAY (minutes, log scale)"); ax[1].set_ylabel("Flights (log scale)")
save(fig, "eda_01_delay_distribution")

# 2 boxplot
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].boxplot(y, vert=False, showfliers=False, patch_artist=True, boxprops=dict(facecolor="#bfdbfe"), medianprops=dict(color=RED))
ax[0].set_title("Departure Delay Boxplot (outliers hidden)"); ax[0].set_xlabel("DEP_DELAY (minutes)"); ax[0].set_yticks([])
order = air.sort_values("mean_delay").index
data = [df.loc[df.AIRLINE == a, "DEP_DELAY"].clip(-30, 120).values for a in order]
bp = ax[1].boxplot(data, vert=False, showfliers=False, patch_artist=True, labels=[a[:22] for a in order], medianprops=dict(color=RED))
for b in bp["boxes"]: b.set_facecolor("#bfdbfe")
ax[1].set_title("Delay Spread by Airline (clipped to 120 min, outliers hidden)"); ax[1].set_xlabel("DEP_DELAY (minutes)")
save(fig, "eda_02_delay_boxplot")

# 3 airline performance
a = air.sort_values("delayed_pct")
fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
ax[0].barh([i[:24] for i in a.index], a.delayed_pct, color=BLUE); ax[0].set_title("Delayed Flights (>=15 min) by Airline"); ax[0].set_xlabel("Delayed %")
for i, v in enumerate(a.delayed_pct): ax[0].text(v + .2, i, f"{v:.1f}%", va="center", fontsize=8)
b = air.sort_values("mean_delay")
ax[1].barh([i[:24] for i in b.index], b.mean_delay, color=CYAN); ax[1].set_title("Mean Departure Delay by Airline"); ax[1].set_xlabel("Mean delay (min)")
for i, v in enumerate(b.mean_delay): ax[1].text(v + .1, i, f"{v:.1f}", va="center", fontsize=8)
save(fig, "eda_03_airline_performance")

# 4 airport performance
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
t = apt.head(15).iloc[::-1]; ax[0].barh(t.index, t.delayed_pct, color=RED); ax[0].set_title("15 Most Delay-Prone Origin Airports\n(>=5,000 flights)"); ax[0].set_xlabel("Delayed %")
t = apt.tail(15); ax[1].barh(t.index, t.delayed_pct, color=GREEN); ax[1].set_title("15 Most Punctual Origin Airports\n(>=5,000 flights)"); ax[1].set_xlabel("Delayed %")
save(fig, "eda_04_airport_performance")

# 5 route performance
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
t = rte.head(15).iloc[::-1]; ax[0].barh([i.replace("_", "→") for i in t.index], t.delayed_pct, color=RED); ax[0].set_title("15 Most Delay-Prone Routes\n(>=1,500 flights)"); ax[0].set_xlabel("Delayed %")
t = rte.tail(15); ax[1].barh([i.replace("_", "→") for i in t.index], t.delayed_pct, color=GREEN); ax[1].set_title("15 Most Punctual Routes\n(>=1,500 flights)"); ax[1].set_xlabel("Delayed %")
save(fig, "eda_05_route_performance")

# 6 monthly
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].bar(mon.index, mon.delayed_pct, color=BLUE); ax[0].set_xticks(range(1, 13)); ax[0].set_xlabel("Month"); ax[0].set_ylabel("Delayed %")
ax[0].set_title("Delay Rate by Calendar Month (2019-2023 pooled)")
ax[1].plot(ym.index.to_timestamp(), ym.delayed_pct, color=BLUE, lw=2, marker="o", ms=3)
ax[1].axvline(pd.Timestamp("2022-09-26"), color=RED, ls="--", label="Test start (26 Sep 2022)"); ax[1].legend()
ax[1].set_title("Monthly Delay Rate Time Series"); ax[1].set_ylabel("Delayed %"); ax[1].set_xlabel("Month")
save(fig, "eda_06_monthly_delay_trends")

# 7 yearly
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].bar(yr.index.astype(str), yr.delayed_pct, color=PURPLE); ax[0].set_title("Delay Rate by Year"); ax[0].set_ylabel("Delayed %")
for i, v in enumerate(yr.delayed_pct): ax[0].text(i, v + .2, f"{v:.1f}%", ha="center")
ax[1].bar(yr.index.astype(str), yr.mean_delay, color=CYAN); ax[1].set_title("Mean Departure Delay by Year"); ax[1].set_ylabel("Mean delay (min)")
for i, v in enumerate(yr.mean_delay): ax[1].text(i, v + .1, f"{v:.1f}", ha="center")
save(fig, "eda_07_yearly_delay_trends")

# 8 hourly
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(hr.index, hr.delayed_pct, color=BLUE, marker="o", lw=2, label="Delayed %")
ax.set_xlabel("Scheduled departure hour"); ax.set_ylabel("Delayed %", color=BLUE); ax.set_xticks(range(0, 24))
ax2 = ax.twinx(); ax2.bar(hr.index, hr.flights, color=SLATE, alpha=.25); ax2.set_ylabel("Flights", color=SLATE); ax2.grid(False)
ax.set_title("Delay Rate by Scheduled Departure Hour (bars = flight volume)")
save(fig, "eda_08_departure_hour")

# 9 day of week
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ax[0].bar(names, dow.delayed_pct, color=[BLUE] * 5 + [AMBER] * 2); ax[0].set_title("Delay Rate by Day of Week"); ax[0].set_ylabel("Delayed %")
for i, v in enumerate(dow.delayed_pct): ax[0].text(i, v + .1, f"{v:.1f}%", ha="center")
ax[1].bar(names, dow.mean_delay, color=[CYAN] * 5 + [AMBER] * 2); ax[1].set_title("Mean Delay by Day of Week"); ax[1].set_ylabel("Mean delay (min)")
save(fig, "eda_09_day_of_week")

# 10 correlation
cols = ["DEP_DELAY", "DELAYED", "DISTANCE", "CRS_ELAPSED_TIME", "SCHEDULED_HOUR", "DAY_OF_WEEK", "MONTH", "YEAR", "IS_WEEKEND", "HOLIDAY_DIST"]
hc = pd.concat([num.reset_index(drop=True), hist.drop(columns=["DELAYED", "DEP_DELAY"]).reset_index(drop=True)], axis=1).corr()
fig, ax = plt.subplots(1, 2, figsize=(18, 7.5))
im = ax[0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax[0].set_xticks(range(len(cols))); ax[0].set_xticklabels(cols, rotation=60, ha="right"); ax[0].set_yticks(range(len(cols))); ax[0].set_yticklabels(cols)
for i in range(len(cols)):
    for j in range(len(cols)): ax[0].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax[0].set_title("Pearson Correlation: Scheduled Attributes"); ax[0].grid(False)
hcols = list(hc.columns)
ax[1].imshow(hc.loc[["DEP_DELAY", "DELAYED"], hcols].values, cmap="RdBu_r", vmin=-.5, vmax=.5, aspect="auto")
ax[1].set_xticks(range(len(hcols))); ax[1].set_xticklabels(hcols, rotation=70, ha="right", fontsize=8); ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["DEP_DELAY", "DELAYED"])
for i, r in enumerate(["DEP_DELAY", "DELAYED"]):
    for j, c in enumerate(hcols): ax[1].text(j, i, f"{hc.loc[r, c]:.2f}", ha="center", va="center", fontsize=6)
ax[1].set_title("Correlation of Targets with Causal History Features"); ax[1].grid(False)
fig.colorbar(im, ax=ax[0], fraction=0.046)
save(fig, "eda_10_correlation_heatmap")

# class distribution (also used as classification chart 1)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
cd = S["class_distribution"]["overall"]
ax[0].bar(["On-time (<15 min)", "Delayed (>=15 min)"], [cd["on_time"], cd["delayed"]], color=[BLUE, RED])
for i, v in enumerate([cd["on_time"], cd["delayed"]]): ax[0].text(i, v, f"{v:,}\n({v / n * 100:.1f}%)", ha="center", va="bottom")
ax[0].set_title("Class Distribution (all valid flights)"); ax[0].set_ylabel("Flights"); ax[0].set_ylim(0, cd["on_time"] * 1.15)
sp = ["fit", "val", "test"]
ax[1].bar(["Fit\n(2019-01 → 2021-12)", "Validation\n(2021-12 → 2022-09)", "Test\n(2022-09 → 2023-08)"], [S["class_distribution"][s]["delayed_pct"] for s in sp], color=[SLATE, AMBER, PURPLE])
for i, s in enumerate(sp): ax[1].text(i, S["class_distribution"][s]["delayed_pct"] + .2, f"{S['class_distribution'][s]['delayed_pct']:.2f}%", ha="center")
ax[1].set_title("Delayed-Class Share by Chronological Split"); ax[1].set_ylabel("Delayed %")
save(fig, "clf_01_class_distribution")
print("EDA charts saved")
