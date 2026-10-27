import os
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from src.feature_engineering import CORE_CATEGORICAL_FEATURES, CORE_NUMERICAL_FEATURES, CORE_ALL_FEATURES

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "classification"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RF_PATH = MODEL_DIR / "rf_classifier.joblib"
DT_PATH = MODEL_DIR / "dt_classifier.joblib"
SVM_PATH = MODEL_DIR / "svm_classifier.joblib"

SPLIT_DATE = pd.to_datetime("2022-09-26")

def get_classification_preprocessor() -> ColumnTransformer:
    """Creates leakage-safe ColumnTransformer for classification."""
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
    """Loads existing fitted champion Random Forest model or fits on training data."""
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

def load_classifier(model_path: Optional[str] = None) -> Pipeline:
    """Loads fitted classification pipeline."""
    path = Path(model_path) if model_path else RF_PATH
    if not path.exists():
        raise FileNotFoundError(f"Classifier model not found at {path}")
    return joblib.load(path)


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

def predict_flight_delay(
    flight_data: pd.DataFrame, 
    pipeline: Optional[Pipeline] = None
) -> Dict[str, Any]:
    """Generates delay classification and probability for user inference."""
    if pipeline is None:
        pipeline = get_or_train_rf_model()
        
    X = flight_data[CORE_ALL_FEATURES]
    pred = int(pipeline.predict(X)[0])
    
    prob = None
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        probs = pipeline.named_steps["classifier"].predict_proba(pipeline.named_steps["preprocessor"].transform(X))[0]
        prob = float(probs[1])
        
    return {
        "delayed": pred,
        "status": "Delayed (≥ 15 mins)" if pred == 1 else "On-Time / Minor Buffer (< 15 mins)",
        "probability_delayed": round(prob * 100, 1) if prob is not None else None,
    }
