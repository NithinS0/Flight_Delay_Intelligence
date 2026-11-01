"""Leakage-safe ("causal") historical delay features.

For a flight departing on calendar day D, every feature below is computed ONLY from flights that
departed on days strictly before D (days < D).  No same-day outcomes and no future outcomes are used.

Operational assumption (documented in the report): by the morning of day D, the realised delay outcomes
of all flights from days <= D-1 are available to the operations centre.

Statistics are Bayesian-smoothed towards the global rate for the same window, so entities with little
history shrink to the network-wide behaviour:
    smoothed = (sum + k * prior) / (n + k)
"""
import numpy as np
import pandas as pd

K_SMOOTH = 20.0


def _cum_lookup(date, ent, delay, flag, lags):
    """Return {lag: (n, s, f)} cumulative totals over days < (date - lag) for each row's entity."""
    d = pd.DataFrame({"ent": ent, "date": date, "n": 1.0, "s": delay, "f": flag})
    daily = d.groupby(["ent", "date"], sort=True, as_index=False).agg(n=("n", "sum"), s=("s", "sum"), f=("f", "sum"))
    cum = daily.groupby("ent")[["n", "s", "f"]].cumsum()
    daily["cn"], daily["cs"], daily["cf"] = cum["n"].values, cum["s"].values, cum["f"].values
    daily = daily.sort_values("date", kind="stable")[["ent", "date", "cn", "cs", "cf"]]
    out = {}
    base = pd.DataFrame({"ent": ent, "idx": np.arange(len(date))})
    for lag in lags:
        left = base.copy()
        left["qd"] = date - lag
        left = left.sort_values("qd", kind="stable")
        m = pd.merge_asof(left, daily, left_on="qd", right_on="date", by="ent", allow_exact_matches=False)
        m = m.sort_values("idx")
        out[lag] = (m["cn"].fillna(0).values, m["cs"].fillna(0).values, m["cf"].fillna(0).values)
    return out


def _window_totals(cl, w):
    """Totals over the window of the previous w days (w=None -> all history)."""
    n0, s0, f0 = cl[0]
    if w is None:
        return n0, s0, f0
    nw, sw, fw = cl[w]
    return n0 - nw, s0 - sw, f0 - fw


def causal_features(date, ent, delay, flag, name, windows, glob, prior_default, k=K_SMOOTH):
    """Build smoothed rate / mean / count features for one entity type.

    glob: dict {window_label: (prior_rate_array, prior_mean_array)} -- network-wide priors per window.
    """
    lags = [0] + [w for w in windows if w is not None]
    cl = _cum_lookup(date, ent, delay, flag, lags)
    feats = {}
    for w in windows:
        n, s, f = _window_totals(cl, w)
        lab = "ALL" if w is None else f"{w}D"
        pr, pm = glob[lab]
        feats[f"H_{name}_RATE_{lab}"] = ((f + k * pr) / (n + k)).astype(np.float32)
        feats[f"H_{name}_MEAN_{lab}"] = ((s + k * pm) / (n + k)).astype(np.float32)
        feats[f"H_{name}_N_{lab}"] = n.astype(np.float32)
    return feats


def global_priors(date, delay, flag, windows, default_rate, default_mean):
    """Network-wide rate / mean per window (unsmoothed, fallback to defaults when no history)."""
    ent = np.zeros(len(date), dtype=np.int64)
    lags = [0] + [w for w in windows if w is not None]
    cl = _cum_lookup(date, ent, delay, flag, lags)
    priors, feats = {}, {}
    for w in windows:
        n, s, f = _window_totals(cl, w)
        lab = "ALL" if w is None else f"{w}D"
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(n > 0, f / np.maximum(n, 1), default_rate)
            mean = np.where(n > 0, s / np.maximum(n, 1), default_mean)
        priors[lab] = (rate, mean)
        feats[lab] = (rate.astype(np.float32), mean.astype(np.float32), n.astype(np.float32))
    return priors, feats
