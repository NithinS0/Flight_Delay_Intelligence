"""Step 1: load flight_data.csv, audit it, clean it, engineer features (incl. causal history) and save the model table."""
import time, sys
import numpy as np
import pandas as pd
from common import *
from hist_features import causal_features, global_priors
from src.preprocessing import clean_flight_data
from src.feature_engineering import engineer_advanced_features

t0 = time.time()
CAT_COLS = ["AIRLINE", "AIRLINE_DOT", "AIRLINE_CODE", "ORIGIN", "ORIGIN_CITY", "DEST", "DEST_CITY", "CANCELLATION_CODE"]
print("Loading", DATA_CSV, flush=True)
raw = pd.read_csv(DATA_CSV, dtype={c: "category" for c in CAT_COLS})
print("loaded", raw.shape, f"{time.time()-t0:.0f}s", flush=True)

# ---------------- RAW AUDIT ----------------
fl = pd.to_datetime(raw["FL_DATE"], errors="coerce")
raw_stats = {
    "file": str(DATA_CSV.name),
    "file_size_mb": round(DATA_CSV.stat().st_size / 1e6, 1),
    "rows": int(len(raw)), "columns": int(raw.shape[1]),
    "column_names": list(raw.columns),
    "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
    "date_min": str(fl.min().date()), "date_max": str(fl.max().date()),
    "missing": {c: {"count": int(v), "pct": round(v / len(raw) * 100, 3)} for c, v in raw.isna().sum().items() if v > 0},
    "full_row_duplicates": int(raw.duplicated().sum()),
    "unique_airlines": int(raw["AIRLINE"].nunique()),
    "unique_origins": int(raw["ORIGIN"].nunique()),
    "unique_destinations": int(raw["DEST"].nunique()),
    "unique_routes": int((raw["ORIGIN"].astype(str) + "_" + raw["DEST"].astype(str)).nunique()),
    "cancelled": int((raw["CANCELLED"] == 1).sum()), "diverted": int((raw["DIVERTED"] == 1).sum()),
    "dep_delay_missing_total": int(raw["DEP_DELAY"].isna().sum()),
    "dep_delay_missing_among_non_cancelled": int(raw.loc[raw["CANCELLED"] != 1, "DEP_DELAY"].isna().sum()),
    "dep_delay_missing_among_diverted": int(raw.loc[raw["DIVERTED"] == 1, "DEP_DELAY"].isna().sum()),
}
print("raw audit done", f"{time.time()-t0:.0f}s", flush=True)

KEEP = ["FL_DATE", "FL_NUMBER", "AIRLINE", "ORIGIN", "DEST", "CRS_DEP_TIME", "DEP_DELAY", "CANCELLED", "DIVERTED",
        "CRS_ELAPSED_TIME", "DISTANCE"]
df = raw[KEEP].copy()
del raw
for c in ["AIRLINE", "ORIGIN", "DEST"]:
    df[c] = df[c].astype(str).astype("category")

# ---------------- CLEAN (project's existing cleaning logic) ----------------
df, summary = clean_flight_data(df)
print("cleaned", df.shape, f"{time.time()-t0:.0f}s", flush=True)
summary["cleaning_missing_crs_elapsed"] = int(df["CRS_ELAPSED_TIME"].isna().sum())
# CRS_ELAPSED_TIME is missing for a handful of rows only -> fill with the median for the same ORIGIN-DEST (fallback: DISTANCE based)
if df["CRS_ELAPSED_TIME"].isna().any():
    med = df.groupby(["ORIGIN", "DEST"], observed=True)["CRS_ELAPSED_TIME"].transform("median")
    df["CRS_ELAPSED_TIME"] = df["CRS_ELAPSED_TIME"].fillna(med).fillna(df["CRS_ELAPSED_TIME"].median())
df = engineer_advanced_features(df)
df.drop(columns=["FL_NUMBER", "CANCELLED"], inplace=True)

# ---------------- extra calendar features (known in advance) ----------------
from pandas.tseries.holiday import USFederalHolidayCalendar
hol = USFederalHolidayCalendar().holidays(start="2018-12-01", end="2023-12-31")
dates = df["FL_DATE"].values.astype("datetime64[D]")
hol_arr = hol.values.astype("datetime64[D]")
pos = np.searchsorted(hol_arr, dates)
prev = hol_arr[np.clip(pos - 1, 0, len(hol_arr) - 1)]
nxt = hol_arr[np.clip(pos, 0, len(hol_arr) - 1)]
dist = np.minimum(np.abs((dates - prev).astype(int)), np.abs((nxt - dates).astype(int)))
df["HOLIDAY_DIST"] = np.clip(dist, 0, 7).astype(np.int8)
doy = df["FL_DATE"].dt.dayofyear.values
df["DOY_SIN"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
df["DOY_COS"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
df["DATE_ORD"] = (df["FL_DATE"] - pd.Timestamp("2019-01-01")).dt.days.astype(np.int64)

# ---------------- chronological split ----------------
train_mask = df["FL_DATE"] < TEST_START
n_train = int(train_mask.sum())
train_dates = df.loc[train_mask, "FL_DATE"]
cum = np.arange(1, n_train + 1) / n_train
val_start = train_dates.iloc[int(np.searchsorted(cum, FIT_FRACTION))]
df["SPLIT"] = np.where(df["FL_DATE"] >= TEST_START, "test", np.where(df["FL_DATE"] >= val_start, "val", "fit"))
split_info = {
    "test_start": str(TEST_START.date()), "val_start": str(val_start.date()),
    "fit_rows": int((df.SPLIT == "fit").sum()), "val_rows": int((df.SPLIT == "val").sum()),
    "test_rows": int((df.SPLIT == "test").sum()), "train_rows_fit_plus_val": n_train,
    "fit_date_range": [str(df.loc[df.SPLIT == "fit", "FL_DATE"].min().date()), str(df.loc[df.SPLIT == "fit", "FL_DATE"].max().date())],
    "val_date_range": [str(df.loc[df.SPLIT == "val", "FL_DATE"].min().date()), str(df.loc[df.SPLIT == "val", "FL_DATE"].max().date())],
    "test_date_range": [str(df.loc[df.SPLIT == "test", "FL_DATE"].min().date()), str(df.loc[df.SPLIT == "test", "FL_DATE"].max().date())],
    "original_project_split": {"train": 11744, "test": 3241, "validation": 2594, "fit": 9150},
}
print(split_info, flush=True)

# ---------------- causal historical features ----------------
df["DEP_DELAY_W"] = df["DEP_DELAY"].clip(-30, 180).astype(np.float32)   # winsorised ONLY inside history means
fit_mask = df["SPLIT"] == "fit"
default_rate = float(df.loc[fit_mask, "DELAYED"].mean())
default_mean = float(df.loc[fit_mask, "DEP_DELAY_W"].mean())
date = df["DATE_ORD"].values
delay = df["DEP_DELAY_W"].values.astype(np.float64)
flag = df["DELAYED"].values.astype(np.float64)

WINDOWS_ALL = [None, 30, 7, 1]
priors, gfeats = global_priors(date, delay, flag, WINDOWS_ALL, default_rate, default_mean)
for lab in ["1D", "7D", "30D"]:
    df[f"H_GLOBAL_RATE_{lab}"] = gfeats[lab][0]
    df[f"H_GLOBAL_MEAN_{lab}"] = gfeats[lab][1]
df["H_GLOBAL_RATE_ALL"] = gfeats["ALL"][0]
print("global priors done", f"{time.time()-t0:.0f}s", flush=True)

ent_codes = {
    "AIRLINE": (df["AIRLINE"].cat.codes.values.astype(np.int64), WINDOWS_ALL),
    "ORIGIN": (df["ORIGIN"].cat.codes.values.astype(np.int64), WINDOWS_ALL),
    "DEST": (df["DEST"].cat.codes.values.astype(np.int64), WINDOWS_ALL),
    "ROUTE": (df["ROUTE"].astype("category").cat.codes.values.astype(np.int64), [None, 30]),
    "ORIGINHOUR": (df["ORIGIN"].cat.codes.values.astype(np.int64) * 24 + df["SCHEDULED_HOUR"].values.astype(np.int64), [None, 30]),
    "AIRLINEROUTE": (df["AIRLINE_ROUTE"].astype("category").cat.codes.values.astype(np.int64), [None]),
}
for name, (codes, wins) in ent_codes.items():
    feats = causal_features(date, codes, delay, flag, name, wins, priors, None)
    for c, v in feats.items():
        df[c] = v
    print(f"  {name} history features done", f"{time.time()-t0:.0f}s", flush=True)

for c in ["ROUTE", "AIRLINE_ROUTE", "DAY_PART"]:
    df[c] = df[c].astype("category")
df["DELAYED"] = df["DELAYED"].astype(np.int8)
df["IS_WEEKEND"] = df["IS_WEEKEND"].astype(np.int8)

hist_cols = [c for c in df.columns if c.startswith("H_")]
print("history feature count:", len(hist_cols))
df.to_parquet(MODEL_TABLE, index=False)
meta = {"history_feature_columns": hist_cols, "default_rate_fit": default_rate, "default_mean_fit": default_mean,
        "split": split_info, "cleaning_summary": summary, "raw_audit": raw_stats,
        "winsorisation_for_history_means": [-30, 180], "k_smooth": 20}
save_json(meta, METRICS_DIR / "data_prep_meta.json")
save_json(raw_stats, METRICS_DIR / "raw_dataset_stats.json")
print("saved", MODEL_TABLE, df.shape, f"total {time.time()-t0:.0f}s")
