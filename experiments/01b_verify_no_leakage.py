"""Leakage verification: brute-force recomputation of causal history features from first principles."""
import numpy as np, pandas as pd
from common import *

df = pd.read_parquet(MODEL_TABLE)
meta = load_json(METRICS_DIR / "data_prep_meta.json")
k = 20.0
prior_rate_default, prior_mean_default = meta["default_rate_fit"], meta["default_mean_fit"]
rng = np.random.default_rng(0)
idx = rng.choice(len(df), 150, replace=False)
dord = df["DATE_ORD"].values
w = df["DEP_DELAY_W"].values.astype(float)
f = df["DELAYED"].values.astype(float)
air = df["AIRLINE"].astype(str).values
org = df["ORIGIN"].astype(str).values
res = {"rows_checked": len(idx), "max_abs_error": {}}
errs = {"H_AIRLINE_RATE_ALL": [], "H_ORIGIN_RATE_7D": [], "H_ORIGIN_N_1D": [], "H_GLOBAL_RATE_30D": [], "H_ORIGIN_MEAN_30D": []}
for i in idx:
    d = dord[i]
    prev = dord < d
    # global priors
    def glob(win):
        m = prev & (dord >= d - win) if win else prev
        n = m.sum()
        return (f[m].mean() if n else prior_rate_default), (w[m].mean() if n else prior_mean_default)
    # AIRLINE ALL rate
    m = prev & (air == air[i]); n = m.sum(); pr, _ = glob(None)
    errs["H_AIRLINE_RATE_ALL"].append(abs((f[m].sum() + k * pr) / (n + k) - df["H_AIRLINE_RATE_ALL"].iat[i]))
    # ORIGIN 7D rate
    m = prev & (dord >= d - 7) & (org == org[i]); n = m.sum(); pr, pm = glob(7)
    errs["H_ORIGIN_RATE_7D"].append(abs((f[m].sum() + k * pr) / (n + k) - df["H_ORIGIN_RATE_7D"].iat[i]))
    # ORIGIN count 1D (yesterday)
    m = (dord == d - 1) & (org == org[i])
    errs["H_ORIGIN_N_1D"].append(abs(m.sum() - df["H_ORIGIN_N_1D"].iat[i]))
    # GLOBAL 30D rate
    pr, pm = glob(30)
    errs["H_GLOBAL_RATE_30D"].append(abs(pr - df["H_GLOBAL_RATE_30D"].iat[i]))
    # ORIGIN 30D mean
    m = prev & (dord >= d - 30) & (org == org[i]); n = m.sum()
    errs["H_ORIGIN_MEAN_30D"].append(abs((w[m].sum() + k * pm) / (n + k) - df["H_ORIGIN_MEAN_30D"].iat[i]))
res["max_abs_error"] = {kk: float(np.max(v)) for kk, v in errs.items()}
res["passes_tolerance_1e-4"] = bool(all(v < 1e-4 for v in res["max_abs_error"].values()))
# same-day independence: features on the first day of the dataset have no history
first = df["DATE_ORD"] == df["DATE_ORD"].min()
res["first_day_rows"] = int(first.sum())
res["first_day_hist_counts_all_zero"] = bool((df.loc[first, [c for c in df.columns if c.startswith("H_") and "_N_" in c]] == 0).all().all())
# no feature is a (near) perfect proxy of the target
hc = [c for c in df.columns if c.startswith("H_")]
cm = df[hc + ["DELAYED"]].sample(300000, random_state=0).corr()["DELAYED"].drop("DELAYED").abs()
res["max_abs_corr_history_feature_with_target"] = float(cm.max()); res["feature_with_max_corr"] = cm.idxmax()
# split sanity
sp = df.groupby("SPLIT")["FL_DATE"].agg(["min", "max", "size"])
res["split_ranges"] = {s: {"min": str(r["min"].date()), "max": str(r["max"].date()), "rows": int(r["size"])} for s, r in sp.iterrows()}
res["splits_chronological_no_overlap"] = bool(sp.loc["fit", "max"] < sp.loc["val", "min"] and sp.loc["val", "max"] < sp.loc["test", "min"])
save_json(res, METRICS_DIR / "leakage_check.json")
print(json.dumps(res, indent=1, default=str) if False else res)
