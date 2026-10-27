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
    ADVANCED_ALL_FEATURES
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "regression"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
GBR_PATH = MODEL_DIR / "transformed_gbr.joblib"
LR_PATH = MODEL_DIR / "linear_regression.joblib"
RF_REG_PATH = MODEL_DIR / "rf_regressor.joblib"

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
        ("regressor", Ridge(alpha=10.0))
    ])

def build_rf_regressor_pipeline(n_estimators: int = 100, max_depth: int = 12, random_state: int = 42) -> Pipeline:
    """Builds Random Forest Regressor pipeline."""
    return Pipeline(steps=[
        ("preprocessor", get_regression_preprocessor(advanced=False)),
        ("regressor", RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1))
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
    """Loads existing fitted champion Transformed GBR model or fits on training data."""
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

def load_regressor(model_path: Optional[str] = None) -> Pipeline:
    """Loads fitted Transformed GBR regression pipeline."""
    path = Path(model_path) if model_path else GBR_PATH
    if not path.exists():
        raise FileNotFoundError(f"Regressor model not found at {path}")
    return joblib.load(path)


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

def predict_delay_duration(
    flight_data: pd.DataFrame, 
    pipeline: Optional[Pipeline] = None
) -> Dict[str, Any]:
    """Generates predicted departure delay duration in minutes for user inference."""
    if pipeline is None:
        pipeline = get_or_train_gbr_model()
        
    X = flight_data[ADVANCED_ALL_FEATURES]
    pred = float(pipeline.predict(X)[0])
    
    if pred < -5:
        category = "Early Departure"
    elif -5 <= pred < 15:
        category = "On-Time / Minor Buffer"
    elif 15 <= pred < 60:
        category = "Moderate Delay"
    else:
        category = "Severe Delay Warning"
        
    return {
        "predicted_delay_minutes": round(pred, 1),
        "category": category,
        "is_delayed_threshold": pred >= 15.0,
    }
