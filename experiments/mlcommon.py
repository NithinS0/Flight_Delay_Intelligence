"""Shared modelling utilities: feature sets, preprocessors, metrics, sampling."""
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
                             median_absolute_error, explained_variance_score)
from common import *

# ---------------- feature sets ----------------
CORE_CAT = ["AIRLINE", "ORIGIN", "DEST"]
CORE_NUM = ["DISTANCE", "CRS_ELAPSED_TIME", "YEAR", "MONTH", "DAY_OF_WEEK", "SCHEDULED_HOUR", "SCHEDULED_MINUTE", "TIME_SIN", "TIME_COS"]
ADV_CAT = ["AIRLINE", "ORIGIN", "DEST", "ROUTE", "AIRLINE_ROUTE", "DAY_PART"]
ADV_NUM = CORE_NUM + ["IS_WEEKEND"]
# improved (no history): YEAR removed (cannot generalise to unseen year values), calendar features added
ENH_CAT = ["AIRLINE", "ORIGIN", "DEST", "DAY_PART"]
ENH_NUM = [c for c in CORE_NUM if c != "YEAR"] + ["IS_WEEKEND", "HOLIDAY_DIST", "DOY_SIN", "DOY_COS"]


def hist_cols(df):
    return [c for c in df.columns if c.startswith("H_")]


def feature_sets(df):
    h = hist_cols(df)
    return {
        "CORE": (CORE_CAT, CORE_NUM),
        "ADV": (ADV_CAT, ADV_NUM),
        "ENH": (ENH_CAT, ENH_NUM),
        "HIST": (ENH_CAT, ENH_NUM + h),
    }


def ordinal_pre(cat, num):
    """Tree models: ordinal codes fitted on the training fold (unseen -> -1) + passthrough numerics."""
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32), cat),
        ("num", "passthrough", num)], sparse_threshold=0.0)


def onehot_pre(cat, num, min_frequency=None):
    """Linear / SVM / original-project replicas: sparse one-hot + standardised numerics."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=min_frequency, dtype=np.float32), cat),
        ("num", StandardScaler(), num)], sparse_threshold=1.0)


# ---------------- data ----------------
def load_table():
    df = pd.read_parquet(MODEL_TABLE)
    for c in ["AIRLINE", "ORIGIN", "DEST", "ROUTE", "AIRLINE_ROUTE", "DAY_PART"]:
        df[c] = df[c].astype(str)   # plain strings -> sklearn encoders are fitted on training folds only
    return df


def split_views(df):
    fit = df[df.SPLIT == "fit"]
    val = df[df.SPLIT == "val"]
    test = df[df.SPLIT == "test"]
    train = df[df.SPLIT != "test"]
    return fit, val, test, train


def time_ordered_sample(n_rows, cap, seed=SEED):
    """Random sample of positions kept in chronological order (df is sorted by date)."""
    if cap is None or n_rows <= cap:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, cap, replace=False))


# ---------------- metrics ----------------
def score_of(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def best_f1_threshold(y, s, n_grid=400):
    qs = np.unique(np.quantile(s, np.linspace(0.02, 0.995, n_grid)))
    best_t, best_f = qs[0], -1.0
    for t in qs:
        p = s >= t
        tp = np.sum(p & (y == 1)); fp = np.sum(p & (y == 0)); fn = np.sum(~p & (y == 1))
        f = 2 * tp / max(2 * tp + fp + fn, 1)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), float(best_f)


def clf_metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
        "balanced_accuracy": balanced_accuracy_score(y, pred), "confusion_matrix": cm.tolist(),
        "threshold": float(thr), "predicted_delayed": int(pred.sum()),
    }


def reg_metrics(y, p):
    return {
        "mae": mean_absolute_error(y, p), "rmse": float(np.sqrt(mean_squared_error(y, p))), "r2": r2_score(y, p),
        "median_ae": median_absolute_error(y, p), "explained_variance": explained_variance_score(y, p),
    }


class Timer:
    def __enter__(self):
        self.t = time.time(); return self
    def __exit__(self, *a):
        self.s = time.time() - self.t
