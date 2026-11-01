"""Dedicated Hypothesis Testing module for Flight Delay Intelligence.
Provides precomputed full-dataset (N = 2,920,859) hypothesis test results with Benjamini-Hochberg FDR correction,
as well as on-demand interactive hypothesis testing functions for custom user subsets in Streamlit.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"


def apply_benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Applies Benjamini-Hochberg False Discovery Rate (FDR) correction to an array of p-values."""
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = len(p_arr)
    if m == 0:
        return p_arr
    order = np.argsort(p_arr)
    bh = np.empty(m, dtype=np.float64)
    prev = 1.0
    for r_rev in range(m - 1, -1, -1):
        idx = order[r_rev]
        rank = r_rev + 1
        val = min(prev, p_arr[idx] * m / rank)
        bh[idx] = val
        prev = val
    return bh


def load_hypothesis_test_results() -> pd.DataFrame:
    """Loads the 14 full-dataset hypothesis tests with Benjamini-Hochberg FDR q-values."""
    csv_path = ARTIFACTS_DIR / "hypothesis_testing_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    json_path = METRICS_DIR / "hypothesis_testing_results.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    return pd.DataFrame()


def run_interactive_two_group_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    label_a: str,
    label_b: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Runs Student's t-test, Welch's t-test, and Mann-Whitney U test between two user-selected groups."""
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    med_a, med_b = float(np.median(a)), float(np.median(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))

    t_student, p_student = stats.ttest_ind(a, b, equal_var=True)
    t_welch, p_welch = stats.ttest_ind(a, b, equal_var=False)

    rng = np.random.default_rng(42)
    sa = rng.choice(a, size=min(50_000, n_a), replace=False) if n_a > 50_000 else a
    sb = rng.choice(b, size=min(50_000, n_b), replace=False) if n_b > 50_000 else b
    u_stat, p_mwu = stats.mannwhitneyu(sa, sb, alternative="two-sided")
    rank_biserial = float(2.0 * (u_stat / (len(sa) * len(sb))) - 1.0)

    pooled_sd = float(np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)))
    cohens_d = float((mean_a - mean_b) / pooled_sd) if pooled_sd > 0 else 0.0

    return {
        "group_a": label_a,
        "group_b": label_b,
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": round(mean_a, 3),
        "mean_b": round(mean_b, 3),
        "median_a": round(med_a, 3),
        "median_b": round(med_b, 3),
        "diff_mean": round(mean_a - mean_b, 3),
        "diff_median": round(med_a - med_b, 3),
        "cohens_d": round(cohens_d, 4),
        "rank_biserial_r": round(rank_biserial, 4),
        "student_t": round(float(t_student), 3),
        "student_p": float(p_student),
        "welch_t": round(float(t_welch), 3),
        "welch_p": float(p_welch),
        "mwu_u": float(u_stat),
        "mwu_p": float(p_mwu),
        "alpha": alpha,
        "significant_welch": bool(p_welch < alpha),
        "significant_mwu": bool(p_mwu < alpha),
    }


def run_custom_two_group_test(df: pd.DataFrame, group_col: str, val_a: Any, val_b: Any, alpha: float = 0.05) -> Dict[str, Any]:
    """Runs Welch's t-test and Mann-Whitney U test on two subsets of `df` defined by `group_col`."""
    a = df.loc[df[group_col] == val_a, "DEP_DELAY"].dropna().to_numpy()
    b = df.loc[df[group_col] == val_b, "DEP_DELAY"].dropna().to_numpy()
    if len(a) < 5 or len(b) < 5:
        return {"error": "Insufficient sample size in one or both groups."}
    res = run_interactive_two_group_test(a, b, str(val_a), str(val_b), alpha=alpha)
    res["mann_whitney_u"] = res["mwu_u"]
    res["mann_whitney_p"] = res["mwu_p"]
    dec = "Reject H₀ (statistically significant difference)" if res["significant_welch"] else "Fail to reject H₀ (no significant difference)"
    res["plain_interpretation"] = (
        f"{dec}: {val_a} (Mean = {res['mean_a']:.2f}m, Median = {res['median_a']:.1f}m) vs "
        f"{val_b} (Mean = {res['mean_b']:.2f}m, Median = {res['median_b']:.1f}m) — "
        f"ΔMean = {res['diff_mean']:+.2f} min, Cohen's d = {res['cohens_d']:.4f}, Welch p = {res['welch_p']:.3e}."
    )
    return res


