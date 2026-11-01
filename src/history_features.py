"""Runtime helper: turns the persisted history snapshot into the causal history features used by the final models.

The snapshot (models/history_snapshot.joblib) summarises all realised flight outcomes up to the last day in flight_data.csv
(31 Aug 2023).  For an interactive "what-if" prediction the features are evaluated as if the flight departs on the day AFTER
the snapshot, i.e. using only past information -- exactly like the training features.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "models" / "history_snapshot.joblib"
K = 20.0
ENTITY_WINDOWS = {
    "AIRLINE": ["ALL", "30D", "7D", "1D"], "ORIGIN": ["ALL", "30D", "7D", "1D"], "DEST": ["ALL", "30D", "7D", "1D"],
    "ROUTE": ["ALL", "30D"], "ORIGINHOUR": ["ALL", "30D"], "AIRLINEROUTE": ["ALL"],
}


def load_snapshot():
    return joblib.load(SNAPSHOT_PATH)


def _smooth(f, s, n, pr, pm):
    return (f + K * pr) / (n + K), (s + K * pm) / (n + K)


def history_features(snapshot, airline, origin, dest, hour):
    """Return {feature_name: value} for one prospective flight."""
    keys = {"AIRLINE": airline, "ORIGIN": origin, "DEST": dest, "ROUTE": f"{origin}_{dest}", "ORIGINHOUR": f"{origin}|{int(hour)}",
            "AIRLINEROUTE": f"{airline}_{origin}_{dest}"}
    g = snapshot["global"]          # {lab: (n, s, f)}
    pri = {}
    for lab, (n, s, f) in g.items():
        pri[lab] = (f / n if n else snapshot["default_rate"], s / n if n else snapshot["default_mean"])
    out = {}
    for ent, labs in ENTITY_WINDOWS.items():
        tab = snapshot[ent]
        row = tab.get(keys[ent], {})
        for lab in labs:
            n, s, f = row.get(lab, (0.0, 0.0, 0.0))
            pr, pm = pri[lab]
            rate, mean = _smooth(f, s, n, pr, pm)
            out[f"H_{ent}_RATE_{lab}"] = rate; out[f"H_{ent}_MEAN_{lab}"] = mean; out[f"H_{ent}_N_{lab}"] = n
    for lab in ["1D", "7D", "30D"]:
        out[f"H_GLOBAL_RATE_{lab}"] = pri[lab][0]; out[f"H_GLOBAL_MEAN_{lab}"] = pri[lab][1]
    out["H_GLOBAL_RATE_ALL"] = pri["ALL"][0]
    return out


def calendar_features(date, hour, minute):
    """Calendar features identical to the training pipeline."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    date = pd.Timestamp(date)
    hol = USFederalHolidayCalendar().holidays(start=date - pd.Timedelta(days=15), end=date + pd.Timedelta(days=15))
    dist = int(min(np.abs((hol - date).days).min(), 7)) if len(hol) else 7
    tot = hour * 60 + minute
    doy = date.dayofyear
    day_part = "Night" if hour < 6 else "Morning" if hour < 12 else "Afternoon" if hour < 18 else "Evening"
    return {
        "MONTH": date.month, "DAY_OF_WEEK": date.dayofweek, "SCHEDULED_HOUR": int(hour), "SCHEDULED_MINUTE": int(minute),
        "TIME_SIN": float(np.sin(2 * np.pi * tot / 1440.0)), "TIME_COS": float(np.cos(2 * np.pi * tot / 1440.0)),
        "IS_WEEKEND": int(date.dayofweek >= 5), "HOLIDAY_DIST": dist,
        "DOY_SIN": float(np.sin(2 * np.pi * doy / 365.25)), "DOY_COS": float(np.cos(2 * np.pi * doy / 365.25)), "DAY_PART": day_part,
    }
