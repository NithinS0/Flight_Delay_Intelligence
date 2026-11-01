"""Step 8: persist the end-of-data history snapshot used by the dashboard predictors."""
import numpy as np, pandas as pd, joblib
from common import *

df = pd.read_parquet(MODEL_TABLE, columns=["DATE_ORD", "AIRLINE", "ORIGIN", "DEST", "ROUTE", "AIRLINE_ROUTE", "SCHEDULED_HOUR", "DELAYED", "DEP_DELAY_W", "SPLIT"])
meta = load_json(METRICS_DIR / "data_prep_meta.json")
for c in ["AIRLINE", "ORIGIN", "DEST", "ROUTE", "AIRLINE_ROUTE"]:
    df[c] = df[c].astype(str)
last = int(df.DATE_ORD.max())
df["ORIGINHOUR"] = df.ORIGIN + "|" + df.SCHEDULED_HOUR.astype(int).astype(str)
df["AIRLINEROUTE"] = df.AIRLINE_ROUTE
wins = {"ALL": None, "30D": 30, "7D": 7, "1D": 1}
snap = {"last_date": str((pd.Timestamp("2019-01-01") + pd.Timedelta(days=last)).date()), "default_rate": meta["default_rate_fit"], "default_mean": meta["default_mean_fit"]}
glob = {}
for lab, w in wins.items():
    d = df if w is None else df[df.DATE_ORD > last - w]
    glob[lab] = (float(len(d)), float(d.DEP_DELAY_W.sum()), float(d.DELAYED.sum()))
snap["global"] = glob
for ent in ["AIRLINE", "ORIGIN", "DEST", "ROUTE", "ORIGINHOUR", "AIRLINEROUTE"]:
    tab = {}
    for lab, w in wins.items():
        d = df if w is None else df[df.DATE_ORD > last - w]
        g = d.groupby(ent).agg(n=("DELAYED", "size"), s=("DEP_DELAY_W", "sum"), f=("DELAYED", "sum"))
        for k, r in g.iterrows():
            tab.setdefault(k, {})[lab] = (float(r.n), float(r.s), float(r.f))
    snap[ent] = tab
    print(ent, len(tab))
# vocab for UI drop-downs
snap["vocab"] = {"AIRLINE": sorted(df.AIRLINE.unique()), "ORIGIN": sorted(df.ORIGIN.unique()), "DEST": sorted(df.DEST.unique())}
joblib.dump(snap, MODELS_DIR / "history_snapshot.joblib", compress=3)
print("snapshot saved", snap["last_date"])
