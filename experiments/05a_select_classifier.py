"""Step 5a: Aggregate all classification results (16 models), include all 11 metrics (Accuracy, Precision, Recall,
F1, Specificity, Balanced Accuracy, ROC-AUC, PR-AUC, MCC, Cohen's Kappa, Log Loss), select the final classifier
on VALIDATION delayed-class F1, and export comparison CSV + JSON.
"""
import numpy as np, pandas as pd
from common import *

recs = [load_json(p) for p in sorted(WORK_DIR.glob("clf_*.json"))]
failed = [r for r in recs if r.get("failed")]
recs = [r for r in recs if not r.get("failed")]

def safe_fn(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")

rows = []
for r in recs:
    v, t, td = r["val"], r["test"], r["test_default_threshold"]
    rows.append({
        "Model": r["model"],
        "Group": r["group"],
        "Feature set": r["feature_set"],
        "Features": r["n_features"],
        "Train rows used (fit stage)": r["fit_rows_used"],
        "Accuracy": t["accuracy"],
        "Precision": t["precision"],
        "Recall": t["recall"],
        "F1-score": t["f1"],
        "Specificity": t.get("specificity", 0.0),
        "Balanced Accuracy": t["balanced_accuracy"],
        "ROC-AUC": t["roc_auc"],
        "PR-AUC": t["pr_auc"],
        "MCC": t.get("mcc", 0.0),
        "Cohen's Kappa": t.get("cohen_kappa", 0.0),
        "Log Loss": t.get("log_loss", np.nan),
        "Training Time (s)": r["train_time_s"],
        "Validation Score (F1)": v["f1"],
        "Test Score (F1)": t["f1"],
        "Val Precision": v["precision"],
        "Val Recall": v["recall"],
        "Val Specificity": v.get("specificity", 0.0),
        "Val ROC-AUC": v["roc_auc"],
        "Val PR-AUC": v["pr_auc"],
        "Val MCC": v.get("mcc", 0.0),
        "Val Cohen's Kappa": v.get("cohen_kappa", 0.0),
        "Val Log Loss": v.get("log_loss", np.nan),
        "Val Accuracy": v["accuracy"],
        "Val Balanced Accuracy": v["balanced_accuracy"],
        "Decision Threshold (val-optimised)": t["threshold"],
        "Test F1 @default predict()": td["f1"],
        "Test Precision @default": td["precision"],
        "Test Recall @default": td["recall"],
        "Test Accuracy @default": td["accuracy"],
        "Test Balanced Acc @default": td["balanced_accuracy"],
        "Test TN": t["confusion_matrix"][0][0],
        "Test FP": t["confusion_matrix"][0][1],
        "Test FN": t["confusion_matrix"][1][0],
        "Test TP": t["confusion_matrix"][1][1],
        "Best params": str(r["best_params"]),
    })

tab = pd.DataFrame(rows).sort_values("Validation Score (F1)", ascending=False).reset_index(drop=True)
tab.insert(0, "Rank (val F1)", np.arange(1, len(tab) + 1))
tab.to_csv(ARTIFACTS / "model_comparison_classification.csv", index=False)

best_row = tab.iloc[0]
best = next(r for r in recs if r["model"] == best_row["Model"])
# Also record best single estimator (for direct pipeline + TreeSHAP)
single_rows = tab[tab["Group"] != "ensemble"]
best_single_row = single_rows.iloc[0]
best_single = next(r for r in recs if r["model"] == best_single_row["Model"])

safe = safe_fn(best["model"])
sel = {
    "model": best["model"],
    "safe_name": safe,
    "best_single_model": best_single["model"],
    "best_single_safe_name": safe_fn(best_single["model"]),
    "threshold": best["test"]["threshold"],
    "feature_set": best["feature_set"],
    "best_params": best["best_params"],
    "val": best["val"],
    "test": best["test"],
    "test_default_threshold": best["test_default_threshold"],
    "train_time_s": best["train_time_s"],
    "tuning": best.get("tuning"),
    "fit_rows_used": best["fit_rows_used"],
    "refit_rows_used": best["refit_rows_used"],
    "selection_rule": "highest validation delayed-class F1 (threshold optimised on validation); recall/precision/PR-AUC/MCC inspected as tie-breakers",
    "runner_up": tab.iloc[1]["Model"] if len(tab) > 1 else None,
    "runner_up_val_f1": float(tab.iloc[1]["Validation Score (F1)"]) if len(tab) > 1 else None,
    "failed_models": [f["model"] for f in failed],
}

# old-vs-new comparison
old = next((r for r in recs if r["model"] == "OLD Random Forest"), None)
cmp = {}
if old:
    def pct(new, base):
        return float((new - base) / base * 100) if base else None
    def block(oldm, newm):
        return {
            k: {"old": oldm[k], "new": newm[k], "abs_change": newm[k] - oldm[k], "pct_change": pct(newm[k], oldm[k])}
            for k in ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "pr_auc", "balanced_accuracy", "mcc", "cohen_kappa"]
            if k in oldm and k in newm
        }
    od = dict(old["test_default_threshold"])
    for k_extra in ["roc_auc", "pr_auc", "specificity", "mcc", "cohen_kappa"]:
        if k_extra in old["test"]:
            od[k_extra] = old["test"][k_extra]
    cmp["old_rf_default_predict_vs_new_best"] = block(od, best["test"])
    cmp["old_rf_valthreshold_vs_new_best"] = block(old["test"], best["test"])
    cmp["note"] = (
        "OLD Random Forest = original-project RF configuration re-run on the full flight_data.csv (100k-row fit sample, CORE features). "
        "'default_predict' mirrors the original operating mode; 'valthreshold' gives the old model the same threshold-optimisation as the new one."
    )
cmp["original_project_reported_on_15k_sample"] = {
    "source": "hard-coded in src/model_evaluation.py (benchmark_sample.csv, 3,241 test rows)",
    "model": "Random Forest", "accuracy_pct": 55.11, "precision_pct": 29.30, "recall_pct": 73.90, "f1": 0.4200, "validation_f1": 0.4150
}
sel["comparison_vs_old"] = cmp
save_json(sel, ARTIFACTS / "classification_selection.json")
print(tab[["Rank (val F1)", "Model", "Group", "Validation Score (F1)", "Test Score (F1)", "Precision", "Recall", "ROC-AUC", "PR-AUC", "MCC", "Training Time (s)"]].to_string())
print("BEST:", best["model"], "thr", sel["threshold"], "| BEST SINGLE:", best_single["model"])
