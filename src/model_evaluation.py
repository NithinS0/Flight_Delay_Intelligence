import pandas as pd
from typing import Dict, Any, List

def get_classification_comparison_table() -> pd.DataFrame:
    """Returns Classification Comparison Table across all evaluated models."""
    data = [
        {
            "Model": "Random Forest Classifier",
            "Accuracy": "55.11%",
            "Precision": "29.30%",
            "Recall": "73.90%",
            "F1-score": "0.4200",
            "Training Time": "4.2s",
            "Validation F1": "0.4150",
            "Status": "Champion (Selected)"
        },
        {
            "Model": "Decision Tree Classifier",
            "Accuracy": "58.79%",
            "Precision": "26.00%",
            "Recall": "48.50%",
            "F1-score": "0.3390",
            "Training Time": "0.4s",
            "Validation F1": "0.3390",
            "Status": "Baseline"
        },
        {
            "Model": "Support Vector Machine (LinearSVC)",
            "Accuracy": "63.96%",
            "Precision": "30.30%",
            "Recall": "50.40%",
            "F1-score": "0.3790",
            "Training Time": "8.7s",
            "Validation F1": "0.3790",
            "Status": "Candidate"
        },
    ]
    return pd.DataFrame(data)

def get_regression_comparison_table() -> pd.DataFrame:
    """Returns Regression Comparison Table across all evaluated models."""
    data = [
        {
            "Model": "Advanced Transformed GBR",
            "MAE (min)": 17.5120,
            "RMSE (min)": 47.4300,
            "R²": -0.0542,
            "Training Time": "12.8s",
            "Validation MAE (min)": 18.7346,
            "Status": "Champion (Selected)"
        },
        {
            "Model": "Gradient Boosting Regressor",
            "MAE (min)": 20.9445,
            "RMSE (min)": 56.2009,
            "R²": -0.0061,
            "Training Time": "11.2s",
            "Validation MAE (min)": 20.9445,
            "Status": "Candidate"
        },
        {
            "Model": "Random Forest Regressor",
            "MAE (min)": 21.2183,
            "RMSE (min)": 57.7292,
            "R²": -0.0616,
            "Training Time": "15.4s",
            "Validation MAE (min)": 21.2183,
            "Status": "Candidate"
        },
        {
            "Model": "Linear Regression (Ridge)",
            "MAE (min)": 21.6519,
            "RMSE (min)": 57.3518,
            "R²": -0.0477,
            "Training Time": "0.2s",
            "Validation MAE (min)": 21.6519,
            "Status": "Baseline"
        },
        {
            "Model": "Robust Gradient Boosting",
            "MAE (min)": 18.9599,
            "RMSE (min)": 58.3146,
            "R²": -0.0832,
            "Training Time": "13.5s",
            "Validation MAE (min)": 18.9599,
            "Status": "Candidate (L1 Loss)"
        },
        {
            "Model": "Historical Gradient Boosting",
            "MAE (min)": 21.2293,
            "RMSE (min)": 56.0514,
            "R²": -0.0007,
            "Training Time": "14.1s",
            "Validation MAE (min)": 21.2293,
            "Status": "Experimental"
        },
    ]
    return pd.DataFrame(data)

def get_best_models_summary() -> Dict[str, Any]:
    """Returns champion models metadata and exact test verification numbers."""
    return {
        "classification": {
            "name": "Random Forest Classifier",
            "hyperparameters": "class_weight='balanced_subsample', max_depth=16, min_samples_split=40, min_samples_leaf=40, n_estimators=100",
            "val_f1": 0.4150,
            "test_accuracy": 55.11,
            "test_precision": 29.30,
            "test_recall": 73.90,
            "test_f1": 0.4200,
            "confusion_matrix": [[1259, 1269], [186, 527]],
            "test_flights": 3241,
            "actual_delayed": 713,
            "detected_delayed": 527,
            "false_alarms": 1269,
            "selection_criterion": "Highest validation F1-score (0.415) on delayed class, optimizing operational disruption recall (73.9%) under severe class imbalance."
        },
        "regression": {
            "name": "Advanced Transformed Gradient Boosting Regressor",
            "hyperparameters": "signed_log_transform(y), n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.7, max_features=0.7",
            "val_mae": 18.7346,
            "test_mae": 17.5120,
            "test_rmse": 47.4300,
            "test_r2": -0.0542,
            "baseline_mean_mae": 20.8344,
            "mae_improvement_pct": 15.95,
            "selection_criterion": "Lowest validation MAE (18.73 min), achieving 15.95% error reduction over mean baseline by compressing severe delay skew."
        }
    }
