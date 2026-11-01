"""Dedicated ANOVA & Post-Hoc Analysis module for Flight Delay Intelligence.
Provides precomputed full-dataset One-Way ANOVA, Welch's ANOVA, Kruskal-Wallis, Two-Way ANOVA (with interaction),
Daily Cluster-Robust Block-Mean ANOVA, and Games-Howell post-hoc tables, plus interactive ANOVA utilities.
"""
from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"


def load_anova_bundle() -> Dict[str, Any]:
    """Loads precomputed ANOVA bundle from artifacts/metrics/anova_results.json."""
    json_path = METRICS_DIR / "anova_results.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"one_way_anova": [], "two_way_anova_daypart_x_weekend": [], "two_way_cell_means": [], "games_howell_posthoc": [], "clustering_note": ""}


def load_oneway_anova_df() -> pd.DataFrame:
    """Loads One-Way ANOVA results table."""
    csv_path = ARTIFACTS_DIR / "anova_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    bundle = load_anova_bundle()
    return pd.DataFrame(bundle.get("one_way_anova", []))


def load_twoway_anova_df() -> pd.DataFrame:
    """Loads Two-Way ANOVA (Day Part x Weekend) table."""
    csv_path = ARTIFACTS_DIR / "two_way_anova_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    bundle = load_anova_bundle()
    return pd.DataFrame(bundle.get("two_way_anova_daypart_x_weekend", []))


def load_posthoc_df() -> pd.DataFrame:
    """Loads Games-Howell Post-Hoc comparison results."""
    csv_path = ARTIFACTS_DIR / "posthoc_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    bundle = load_anova_bundle()
    return pd.DataFrame(bundle.get("games_howell_posthoc", []))

