import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.feature_engineering import (
    CORE_CATEGORICAL_FEATURES,
    CORE_NUMERICAL_FEATURES,
    CORE_ALL_FEATURES,
    ADVANCED_CATEGORICAL_FEATURES,
    ADVANCED_NUMERICAL_FEATURES,
    ADVANCED_ALL_FEATURES,
)
from src.classification import enrich_with_causal_history

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models" / "regression"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
GBR_PATH = MODEL_DIR / "transformed_gbr.joblib"
LR_PATH = MODEL_DIR / "linear_regression.joblib"
RF_REG_PATH = MODEL_DIR / "rf_regressor.joblib"
LGB_L1_PATH = MODEL_DIR / "LightGBM_L1.joblib"
TWO_STAGE_PATH = MODEL_DIR / "two_stage_lightgbm_l1.joblib"
XGB_CLF_PATH = ROOT_DIR / "models" / "classification" / "XGBoost_tuned.joblib"

SPLIT_DATE = pd.to_datetime("2022-09-26")


def signed_log_transform(y: np.ndarray) -> np.ndarray:
    """Signed logarithmic transformation: sign(y) * log1p(abs(y)) to compress severe right skew."""
    return np.sign(y) * np.log1p(np.abs(y))


def signed_log_inverse(z: np.ndarray) -> np.ndarray:
    """Inverse signed logarithmic transformation: sign(z) * expm1(abs(z))."""
    return np.sign(z) * np.expm1(np.abs(z))


def get_regression_preprocessor(advanced: bool = True) -> ColumnTransformer:
    """Creates ColumnTransformer for regression features."""
    cat_feats = ADVANCED_CATEGORICAL_FEATURES if advanced else CORE_CATEGORICAL_FEATURES
    num_feats = ADVANCED_NUMERICAL_FEATURES if advanced else CORE_NUMERICAL_FEATURES
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
            ("num", StandardScaler(), num_feats),
        ]
    )


def build_linear_regression_pipeline() -> Pipeline:
    """Builds standard Linear Regression pipeline."""
    return Pipeline(steps=[
        ("preprocessor", get_regression_preprocessor(advanced=False)),
        ("regressor", Ridge(alpha=10.0)),
    ])


def build_rf_regressor_pipeline(n_estimators: int = 100, max_depth: int = 12, random_state: int = 42) -> Pipeline:
    """Builds Random Forest Regressor pipeline."""
    return Pipeline(steps=[
        ("preprocessor", get_regression_preprocessor(advanced=False)),
        ("regressor", RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)),
    ])


def build_transformed_gbr_pipeline(
    n_estimators: int = 150,
    subsample: float = 0.8,
    min_samples_split: int = 20,
    min_samples_leaf: int = 5,
    max_features: str = "sqrt",
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Pipeline:
    """Builds Transformed Gradient Boosting Regressor Pipeline with signed log target transform."""
    preprocessor = get_regression_preprocessor(advanced=True)
    base_gbr = GradientBoostingRegressor(
        n_estimators=n_estimators,
        subsample=subsample,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    transformed_regressor = TransformedTargetRegressor(
        regressor=base_gbr,
        func=signed_log_transform,
        inverse_func=signed_log_inverse,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", transformed_regressor)])


def get_or_train_gbr_model(train_df: Optional[pd.DataFrame] = None) -> Pipeline:
    """Loads existing fitted Transformed GBR model or fits on training data."""
    if GBR_PATH.exists():
        try:
            return joblib.load(GBR_PATH)
        except Exception:
            pass

    if train_df is not None:
        pipe = build_transformed_gbr_pipeline()
        X_train = train_df[ADVANCED_ALL_FEATURES]
        y_train = train_df["DEP_DELAY"].values
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, GBR_PATH)
        return pipe

    raise FileNotFoundError("GBR model not found and no training data provided.")


def load_regressor(model_path: Optional[str] = None) -> Any:
    """Loads fitted regression pipeline."""
    path = Path(model_path) if model_path else (LGB_L1_PATH if LGB_L1_PATH.exists() else GBR_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Regressor model not found at {path}")
    obj = joblib.load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]
    return obj


def evaluate_regression_model(pipeline: Pipeline, df_test: pd.DataFrame, is_advanced: bool = True) -> Dict[str, Any]:
    """Evaluates regression pipeline against test set and calculates MAE, RMSE, R²."""
    feats = ADVANCED_ALL_FEATURES if is_advanced else CORE_ALL_FEATURES
    X_test = df_test[feats]
    y_test = df_test["DEP_DELAY"].values

    y_pred = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    residuals = y_test - y_pred

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "y_true": y_test,
        "y_pred": y_pred,
        "residuals": residuals,
        "test_records": len(df_test),
    }


def _categorize_delay(pred: float) -> str:
    if pred < -5:
        return "Early Departure"
    elif -5 <= pred < 15:
        return "On-Time / Minor Buffer"
    elif 15 <= pred < 60:
        return "Moderate Delay"
    else:
        return "Severe Delay Warning"


def load_all_regressor_records() -> list:
    """Loads the top 4 selected regression model records sorted by validation MAE:
    1. LightGBM Regressor (Champion)
    2. HistGradientBoosting Regressor
    3. XGBoost Regressor
    4. Gradient Boosting Regressor
    """
    import json
    work_dir = ROOT_DIR / "artifacts" / "work"
    target_map = {
        "reg_LightGBM_L1.json": "LightGBM Regressor",
        "reg_HistGradientBoosting_L1.json": "HistGradientBoosting Regressor",
        "reg_XGBoost_L1.json": "XGBoost Regressor",
        "reg_Gradient_Boosting_L1.json": "Gradient Boosting Regressor",
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
    records.sort(key=lambda r: float(r["val"].get("mae", 999.0)))
    return records


def get_regressor_record_by_name(model_name: str) -> Optional[Dict[str, Any]]:
    """Returns a single regressor's full evaluation record by name."""
    recs = load_all_regressor_records()
    norm = model_name.lower().replace(" ", "").replace("_", "")
    for r in recs:
        r_norm = r.get("model", "").lower().replace(" ", "").replace("_", "")
        if r.get("model") == model_name or norm in r_norm or r_norm in norm:
            return r
    return recs[0] if recs else None


def predict_delay_duration(
    flight_data: pd.DataFrame,
    pipeline: Optional[Pipeline] = None,
) -> Dict[str, Any]:
    """Generates predicted departure delay duration in minutes for user inference
    using the Champion LightGBM Regressor (L1 loss).
    """
    if pipeline is None and LGB_L1_PATH.exists():
        try:
            df_enriched = enrich_with_causal_history(flight_data)
            lgb_bundle = joblib.load(LGB_L1_PATH)
            cols = lgb_bundle["columns"]
            X = df_enriched[cols]
            direct_pred = float(lgb_bundle["pipeline"].predict(X)[0])

            return {
                "predicted_delay_minutes": round(direct_pred, 1),
                "direct_l1_minutes": round(direct_pred, 1),
                "category": _categorize_delay(direct_pred),
                "is_delayed_threshold": direct_pred >= 15.0,
                "model_used": "LightGBM Regressor (L1 Loss, Causal History)",
            }
        except Exception:
            pass

    if pipeline is None:
        pipeline = get_or_train_gbr_model()

    df_adv = enrich_with_causal_history(flight_data)
    X = df_adv[ADVANCED_ALL_FEATURES]
    pred = float(pipeline.predict(X)[0])

    return {
        "predicted_delay_minutes": round(pred, 1),
        "direct_l1_minutes": round(pred, 1),
        "category": _categorize_delay(pred),
        "is_delayed_threshold": pred >= 15.0,
        "model_used": "Transformed Gradient Boosting Regressor",
    }
