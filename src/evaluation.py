from typing import Dict, Any, List

def get_classification_benchmarks() -> Dict[str, Any]:
    """Returns exact benchmark results for Phase 2 Classification."""
    initial_models = [
        {
            "Model": "Decision Tree Classifier",
            "Accuracy": "56.56%",
            "Precision (Delayed)": 0.28,
            "Recall (Delayed)": 0.63,
            "F1-Score": 0.39,
            "Notes": "Fast baseline; high variance and overfitting tendency"
        },
        {
            "Model": "Random Forest Classifier",
            "Accuracy": "56.43%",
            "Precision (Delayed)": 0.30,
            "Recall (Delayed)": 0.72,
            "F1-Score": 0.42,
            "Notes": "Strong ensemble stability with high delay sensitivity"
        },
        {
            "Model": "Support Vector Machine (LinearSVC)",
            "Accuracy": "53.29%",
            "Precision (Delayed)": 0.28,
            "Recall (Delayed)": 0.69,
            "F1-Score": 0.39,
            "Notes": "Linear separation struggle on non-linear delay clusters"
        }
    ]
    
    validation_models = [
        {
            "Model": "Decision Tree",
            "Accuracy": "58.79%",
            "Precision (Delayed)": 0.260,
            "Recall (Delayed)": 0.485,
            "F1-Score": 0.339,
            "Selected": "No"
        },
        {
            "Model": "Random Forest Classifier",
            "Accuracy": "62.53%",
            "Precision (Delayed)": 0.314,
            "Recall (Delayed)": 0.611,
            "F1-Score": 0.415,
            "Selected": "Yes (Champion)"
        },
        {
            "Model": "Support Vector Machine",
            "Accuracy": "63.96%",
            "Precision (Delayed)": 0.303,
            "Recall (Delayed)": 0.504,
            "F1-Score": 0.379,
            "Selected": "No"
        }
    ]
    
    hyperparameters = {
        "min_samples_split": 40,
        "min_samples_leaf": 40,
        "max_features": "sqrt",
        "max_depth": 16,
        "class_weight": "balanced_subsample",
        "best_cv_f1": 0.28,
    }
    
    final_test = {
        "accuracy": 55.11,
        "precision": 29.3,
        "recall": 73.9,
        "f1": 0.42,
        "confusion_matrix": [
            [1259, 1269],
            [186, 527]
        ],
        "true_negatives": 1259,
        "false_positives": 1269,
        "false_negatives": 186,
        "true_positives": 527,
        "interpretation": (
            "The classifier detected 527 of the 713 actual delayed flights in the test period (73.9% recall). "
            "To capture 73.9% of disruptions, the balanced class weighting yields 1,269 false alarms, "
            "which represents a proactive operational alert posture."
        )
    }
    
    return {
        "initial_models": initial_models,
        "validation_models": validation_models,
        "hyperparameters": hyperparameters,
        "final_test": final_test,
    }

def get_regression_benchmarks() -> Dict[str, Any]:
    """Returns exact benchmark results for Phase 3 and Phase 4 Regression."""
    validation_models = [
        {
            "Model Architecture": "Linear Regression",
            "Validation MAE (min)": 21.6519,
            "Validation RMSE (min)": 57.3518,
            "Validation R²": -0.0477,
            "Selection Outcome": "Baseline"
        },
        {
            "Model Architecture": "Random Forest Regressor",
            "Validation MAE (min)": 21.2183,
            "Validation RMSE (min)": 57.7292,
            "Validation R²": -0.0616,
            "Selection Outcome": "Candidate (CV MAE: 18.19m)"
        },
        {
            "Model Architecture": "Original Gradient Boosting",
            "Validation MAE (min)": 20.9445,
            "Validation RMSE (min)": 56.2009,
            "Validation R²": -0.0061,
            "Selection Outcome": "Candidate (CV MAE: 18.37m)"
        },
        {
            "Model Architecture": "Advanced Transformed Gradient Boosting",
            "Validation MAE (min)": 18.7346,
            "Validation RMSE (min)": 57.5528,
            "Validation R²": -0.0551,
            "Selection Outcome": "SELECTED CHAMPION (Lowest MAE)"
        },
        {
            "Model Architecture": "Robust Gradient Boosting",
            "Validation MAE (min)": 18.9599,
            "Validation RMSE (min)": 58.3146,
            "Validation R²": -0.0832,
            "Selection Outcome": "Candidate (loss='absolute_error')"
        },
        {
            "Model Architecture": "Historical Gradient Boosting",
            "Validation MAE (min)": 21.2293,
            "Validation RMSE (min)": 56.0514,
            "Validation R²": -0.0007,
            "Selection Outcome": "Candidate (No leakage history)"
        }
    ]
    
    hyperparameters = {
        "subsample": 0.7,
        "n_estimators": 300,
        "min_samples_split": 20,
        "min_samples_leaf": 5,
        "max_features": 0.7,
        "max_depth": 4,
        "learning_rate": 0.05,
        "cv_mae": 13.8860
    }
    
    final_test = {
        "baseline_mean": {
            "mae": 20.834384,
            "rmse": 46.285194,
            "r2": -0.003929
        },
        "selected_model": {
            "mae": 17.511991,
            "rmse": 47.429980,
            "r2": -0.054204
        },
        "mae_improvement": 15.95,
        "interpretation": (
            "The selected model achieves a 15.95% reduction in Mean Absolute Error (17.51m vs 20.83m baseline). "
            "However, its RMSE is slightly higher than baseline (47.43m vs 46.29m) and R² is negative (-0.0542). "
            "This confirms the model performs effectively on normal delays but underpredicts extreme tail events (>120 mins). "
            "In accordance with methodological standards, regression is NOT described as highly accurate in R² terms."
        )
    }
    
    return {
        "validation_models": validation_models,
        "hyperparameters": hyperparameters,
        "final_test": final_test,
    }

def get_combined_comparison() -> List[Dict[str, Any]]:
    """Returns Phase 5 combined comparison between Classification and Regression."""
    return [
        {
            "Modeling Dimension": "Prediction Objective",
            "Classification (Random Forest)": "Predict whether departure delay ≥ 15 mins",
            "Regression (Transformed GBR)": "Estimate departure delay duration in minutes"
        },
        {
            "Modeling Dimension": "Target Variable",
            "Classification (Random Forest)": "DELAYED (Binary: 0 or 1)",
            "Regression (Transformed GBR)": "DEP_DELAY (Continuous: -41 to 1,180 min)"
        },
        {
            "Modeling Dimension": "Selected Algorithm",
            "Classification (Random Forest)": "Tuned Random Forest (balanced_subsample)",
            "Regression (Transformed GBR)": "Advanced Transformed Gradient Boosting"
        },
        {
            "Modeling Dimension": "Primary Metric",
            "Classification (Random Forest)": "Delayed Recall: 73.9% | F1-Score: 0.42",
            "Regression (Transformed GBR)": "Test MAE: 17.51 mins (+15.95% vs baseline)"
        },
        {
            "Modeling Dimension": "Secondary Metrics",
            "Classification (Random Forest)": "Accuracy: 55.11% | Precision: 29.3%",
            "Regression (Transformed GBR)": "RMSE: 47.43 mins | R²: -0.0542"
        },
        {
            "Modeling Dimension": "Operational Utility",
            "Classification (Random Forest)": "Early warning triage for dispatchers & gate crews",
            "Regression (Transformed GBR)": "Turnaround buffer planning & passenger alerts"
        },
        {
            "Modeling Dimension": "Known Limitation",
            "Classification (Random Forest)": "Elevated false alarm rate (1,269 false positives)",
            "Regression (Transformed GBR)": "Underpredicts extreme black-swan delays (>120m)"
        }
    ]
