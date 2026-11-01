import pandas as pd, numpy as np, json, sys
from pathlib import Path

ROOT = Path(r"d:\ISPA")
out = {}
for name, p in [("sample", ROOT/"data/raw/benchmark_sample.csv"), ("cleaned", ROOT/"data/processed/cleaned_flights.csv"), ("full", ROOT/"flight_data.csv")]:
    df = pd.read_csv(p, usecols=lambda c: c in ["FL_DATE","AIRLINE","ORIGIN","DEST","DEP_DELAY","CANCELLED","DIVERTED","CRS_DEP_TIME","DISTANCE","CRS_ELAPSED_TIME"] )
    d = pd.to_datetime(df["FL_DATE"], errors="coerce")
    info = {
        "rows": len(df), "cols_loaded": list(df.columns),
        "date_min": str(d.min()), "date_max": str(d.max()),
        "airlines": int(df.AIRLINE.nunique()), "origins": int(df.ORIGIN.nunique()), "dests": int(df.DEST.nunique()),
    }
    if "CANCELLED" in df:
        info["cancelled"] = int((df.CANCELLED==1).sum()); info["diverted"] = int((df.DIVERTED==1).sum())
    nc = df[df.get("CANCELLED", 0) != 1] if "CANCELLED" in df else df
    nc = nc.dropna(subset=["DEP_DELAY"])
    info["valid_rows"] = len(nc)
    nd = pd.to_datetime(nc["FL_DATE"])
    info["train_lt_split"] = int((nd < "2022-09-26").sum()); info["test_ge_split"] = int((nd >= "2022-09-26").sum())
    info["delayed_rate"] = float((nc.DEP_DELAY >= 15).mean())
    info["max_delay"] = float(nc.DEP_DELAY.max())
    print(name, json.dumps(info, indent=1)); sys.stdout.flush()
