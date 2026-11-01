import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.feature_engineering import (
    CORE_CATEGORICAL_FEATURES, CORE_NUMERICAL_FEATURES, CORE_ALL_FEATURES,
    engineer_advanced_features
)
from src.history_features import load_snapshot, history_features, calendar_features

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models" / "classification"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RF_PATH = MODEL_DIR / "rf_classifier.joblib"
DT_PATH = MODEL_DIR / "dt_classifier.joblib"
SVM_PATH = MODEL_DIR / "svm_classifier.joblib"
XGB_HIST_PATH = MODEL_DIR / "XGBoost_tuned.joblib"
ENS_HIST_PATH = MODEL_DIR / "Soft_Voting_Ensemble_XGBplusLGBplusHGB.joblib"

SPLIT_DATE = pd.to_datetime("2022-09-26")

_SNAPSHOT_CACHE = None


def _get_snapshot():
    global _SNAPSHOT_CACHE
    if _SNAPSHOT_CACHE is None:
        try:
            _SNAPSHOT_CACHE = load_snapshot()
        except Exception:
            _SNAPSHOT_CACHE = None
    return _SNAPSHOT_CACHE


def enrich_with_causal_history(df_in: pd.DataFrame) -> pd.DataFrame:
    """Enriches a user inference DataFrame with calendar and causal historical features from the persisted snapshot."""
    df = df_in.copy()
    if "FL_DATE" not in df.columns:
        yr = int(df["YEAR"].iloc[0]) if "YEAR" in df.columns else 2023
        mo = int(df["MONTH"].iloc[0]) if "MONTH" in df.columns else 7
        df["FL_DATE"] = pd.Timestamp(f"{yr:04d}-{mo:02d}-15")
    if "CRS_DEP_TIME" not in df.columns:
        hr = int(df["SCHEDULED_HOUR"].iloc[0]) if "SCHEDULED_HOUR" in df.columns else 12
        mn = int(df["SCHEDULED_MINUTE"].iloc[0]) if "SCHEDULED_MINUTE" in df.columns else 0
        df["CRS_DEP_TIME"] = hr * 100 + mn
    df = engineer_advanced_features(df)
    snap = _get_snapshot()
    if snap is None:
        return df

    rows_hist = []
    for _, row in df.iterrows():
        airline = str(row.get("AIRLINE", "Delta Air Lines Inc."))
        origin = str(row.get("ORIGIN", "ATL"))
        dest = str(row.get("DEST", "LAX"))
        hour = int(row.get("SCHEDULED_HOUR", 12))
        minute = int(row.get("SCHEDULED_MINUTE", 0))
        fl_date = row.get("FL_DATE", pd.Timestamp(f"{int(row.get('YEAR', 2023))}-{int(row.get('MONTH', 7)):02d}-15"))
        cal = calendar_features(fl_date, hour, minute)
        h = history_features(snap, airline, origin, dest, hour)
        rows_hist.append({**cal, **h})

    hist_df = pd.DataFrame(rows_hist, index=df.index)
    for col in hist_df.columns:
        if col not in df.columns:
            df[col] = hist_df[col]
    return df


def get_classification_preprocessor() -> ColumnTransformer:
    """Creates leakage-safe ColumnTransformer for CORE classification."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CORE_CATEGORICAL_FEATURES),
            ("num", StandardScaler(), CORE_NUMERICAL_FEATURES),
        ]
    )


def build_rf_pipeline(
    min_samples_split: int = 40,
    min_samples_leaf: int = 40,
    max_features: str = "sqrt",
    max_depth: int = 16,
    class_weight: str = "balanced_subsample",
    random_state: int = 42,
    n_estimators: int = 100,
) -> Pipeline:
    """Builds Random Forest Classification Pipeline."""
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", get_classification_preprocessor()), ("classifier", classifier)])


def build_dt_pipeline(max_depth: int = 12, min_samples_split: int = 50, random_state: int = 42) -> Pipeline:
    """Builds Decision Tree Classification Pipeline."""
    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocessor", get_classification_preprocessor()), ("classifier", classifier)])


def build_svm_pipeline(C: float = 1.0, random_state: int = 42) -> Pipeline:
    """Builds Linear Support Vector Machine Pipeline."""
    classifier = LinearSVC(
        C=C,
        class_weight="balanced",
        random_state=random_state,
        max_iter=3000,
        dual=False,
    )
    return Pipeline(steps=[("preprocessor", get_classification_preprocessor()), ("classifier", classifier)])


def get_or_train_rf_model(train_df: Optional[pd.DataFrame] = None) -> Pipeline:
    """Loads existing fitted Random Forest model or fits on training data."""
    if RF_PATH.exists():
        try:
            return joblib.load(RF_PATH)
        except Exception:
            pass

    if train_df is not None:
        pipe = build_rf_pipeline()
        X_train = train_df[CORE_ALL_FEATURES]
        y_train = (train_df["DEP_DELAY"] >= 15.0).astype(int)
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, RF_PATH)
        return pipe

    raise FileNotFoundError("RF model not found and no training data provided.")


def load_classifier(model_path: Optional[str] = None) -> Any:
    """Loads fitted classification pipeline or bundle."""
    path = Path(model_path) if model_path else (XGB_HIST_PATH if XGB_HIST_PATH.exists() else RF_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Classifier model not found at {path}")
    obj = joblib.load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]
    return obj


def evaluate_classifier_model(pipeline: Pipeline, df_test: pd.DataFrame) -> Dict[str, Any]:
    """Evaluates classifier on test dataset and returns metrics and confusion matrix."""
    X_test = df_test[CORE_ALL_FEATURES]
    y_test = (df_test["DEP_DELAY"] >= 15.0).astype(int)
    y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    rep = classification_report(y_test, y_pred, target_names=["On-Time", "Delayed"], output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1, 4),
        "confusion_matrix": cm,
        "report": rep,
        "test_records": len(df_test),
        "actual_delayed": int(y_test.sum()),
        "predicted_delayed": int((y_pred == 1).sum()),
    }


def load_all_classifier_records() -> list:
    """Loads the top 4 selected classification model records sorted by validation F1:
    1. XGBoost Classifier
    2. HistGradientBoosting Classifier
    3. LightGBM Classifier
    4. Logistic Regression
    """
    work_dir = ROOT_DIR / "artifacts" / "work"
    target_map = {
        "clf_XGBoost_tuned.json": "XGBoost Classifier",
        "clf_HistGradientBoosting_tuned.json": "HistGradientBoosting Classifier",
        "clf_LightGBM_tuned.json": "LightGBM Classifier",
        "clf_Logistic_Regression_tuned.json": "Logistic Regression",
    }
    records = []
    for fname, model_label in target_map.items():
        p = work_dir / fname
        if p.exists():
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                rec["model"] = model_label
                records.append(rec)
            except Exception:
                pass
    records.sort(key=lambda r: float(r["val"].get("f1", 0.0)), reverse=True)
    return records


def get_classifier_record_by_name(model_name: str) -> Optional[Dict[str, Any]]:
    """Returns a single classifier's full evaluation record by name."""
    recs = load_all_classifier_records()
    norm = model_name.lower().replace(" ", "").replace("_", "")
    for r in recs:
        r_norm = r.get("model", "").lower().replace(" ", "").replace("_", "")
        if r.get("model") == model_name or norm in r_norm or r_norm in norm:
            return r
    return recs[0] if recs else None


def predict_flight_delay(
    flight_data: Optional[pd.DataFrame] = None,
    pipeline: Optional[Pipeline] = None,
    use_champion_hist: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Generates delay classification and calibrated probability for user inference.
    Accepts either a `flight_data` DataFrame OR keyword arguments (`airline`, `origin`, `dest`, `month`, `day_of_week`, `scheduled_hour`, `scheduled_minute`, `distance`, `crs_elapsed_time`).
    """
    if flight_data is None:
        airline = kwargs.get("airline", "Delta Air Lines Inc.")
        origin = kwargs.get("origin", "ATL")
        dest = kwargs.get("dest", "LAX")
        month = int(kwargs.get("month", 7))
        dow = int(kwargs.get("day_of_week", 4))
        hour = int(kwargs.get("scheduled_hour", 17))
        minute = int(kwargs.get("scheduled_minute", 30))
        dist = float(kwargs.get("distance", 1000.0))
        elap = float(kwargs.get("crs_elapsed_time", 150.0))
        tot_m = hour * 60 + minute
        flight_data = pd.DataFrame([{
            "AIRLINE": airline, "ORIGIN": origin, "DEST": dest,
            "DISTANCE": dist, "CRS_ELAPSED_TIME": elap, "YEAR": 2023,
            "MONTH": month, "DAY_OF_WEEK": dow, "SCHEDULED_HOUR": hour, "SCHEDULED_MINUTE": minute,
            "TIME_SIN": float(np.sin(2.0 * np.pi * tot_m / 1440.0)),
            "TIME_COS": float(np.cos(2.0 * np.pi * tot_m / 1440.0)),
            "ROUTE": f"{origin}_{dest}", "AIRLINE_ROUTE": f"{airline}_{origin}_{dest}",
        }])

    if pipeline is None and use_champion_hist and XGB_HIST_PATH.exists():
        try:
            df_enriched = enrich_with_causal_history(flight_data)
            bundle = joblib.load(XGB_HIST_PATH)
            pipe = bundle["pipeline"]
            cols = bundle["columns"]
            thr = float(bundle.get("threshold", 0.2372))
            X = df_enriched[cols]
            prob = float(pipe.predict_proba(X)[0, 1])
            pred = int(prob >= thr)
            risk_level = "High Disruption Risk" if prob >= thr * 1.35 else ("Moderate Risk" if prob >= thr else "Low Risk (Normal Operations)")
            status_str = "DELAYED (≥ 15 min)" if pred == 1 else "ON-TIME (< 15 min)"
            return {
                "delayed": pred,
                "predicted_class": pred,
                "status": "Delayed (≥ 15 mins)" if pred == 1 else "On-Time / Minor Buffer (< 15 mins)",
                "status_label": status_str,
                "probability_delayed": round(prob * 100.0, 1),
                "delay_probability": round(prob, 4),
                "threshold": round(thr, 4),
                "decision_threshold_pct": round(thr * 100.0, 1),
                "risk_level": risk_level,
                "model_used": "XGBoost (Tuned, Causal History)",
                "history_context": {
                    "airline_hist_delay_rate_pct": round(float(df_enriched["H_AIRLINE_RATE_ALL"].iloc[0]) * 100.0, 1) if "H_AIRLINE_RATE_ALL" in df_enriched else None,
                    "origin_30d_delay_rate_pct": round(float(df_enriched["H_ORIGIN_RATE_30D"].iloc[0]) * 100.0, 1) if "H_ORIGIN_RATE_30D" in df_enriched else None,
                    "route_hist_delay_rate_pct": round(float(df_enriched["H_ROUTE_RATE_ALL"].iloc[0]) * 100.0, 1) if "H_ROUTE_RATE_ALL" in df_enriched else None,
                },
            }
        except Exception:
            pass

    if pipeline is None:
        pipeline = get_or_train_rf_model()

    X = flight_data[CORE_ALL_FEATURES]
    pred = int(pipeline.predict(X)[0])

    prob = 0.5
    if hasattr(pipeline, "named_steps") and "classifier" in pipeline.named_steps:
        clf_step = pipeline.named_steps["classifier"]
        pre_step = pipeline.named_steps["preprocessor"]
        if hasattr(clf_step, "predict_proba"):
            probs = clf_step.predict_proba(pre_step.transform(X))[0]
            prob = float(probs[1])

    return {
        "delayed": pred,
        "predicted_class": pred,
        "status": "Delayed (≥ 15 mins)" if pred == 1 else "On-Time / Minor Buffer (< 15 mins)",
        "status_label": "DELAYED (≥ 15 min)" if pred == 1 else "ON-TIME (< 15 min)",
        "probability_delayed": round(prob * 100.0, 1),
        "delay_probability": round(prob, 4),
        "threshold": 0.5,
        "decision_threshold_pct": 50.0,
        "risk_level": "High Risk" if pred == 1 else "Low Risk",
        "model_used": "Random Forest Classifier (Core Baseline)",
    }
