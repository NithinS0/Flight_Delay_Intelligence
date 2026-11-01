"""Step 5b: Aggregate all regression results (+ baselines + ensemble + two-stage), include all 9 regression metrics
(MAE, MSE, RMSE, R², Adjusted R², Median AE, Explained Variance, Max Error, Mean Bias Error), and select the
final regressor on VALIDATION MAE.
"""
import numpy as np, pandas as pd
from common import *

recs = [load_json(p) for p in sorted(WORK_DIR.glob("reg_*.json")) if "baselines" not in p.name]
failed = [r for r in recs if r.get("failed")]
recs = [r for r in recs if not r.get("failed")]
base = load_json(WORK_DIR / "reg_baselines.json")
ts = load_json(METRICS_DIR / "two_stage_results.json") if (METRICS_DIR / "two_stage_results.json").exists() else {"variants": {}}
allrecs = recs + list(base.values()) + list(ts["variants"].values())

def sf(n):
    return n.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("+", "plus")

mean_t = base["Training mean baseline"]["test"]["mae"]
med_t = base["Training median baseline"]["test"]["mae"]
mean_v = base["Training mean baseline"]["val"]["mae"]
med_v = base["Training median baseline"]["val"]["mae"]
old = next((r for r in recs if r["model"] == "OLD Transformed GBR (signed-log)"), None)
old_t = old["test"]["mae"] if old else np.nan

rows = []
for r in allrecs:
    v, t = r["val"], r["test"]
    rows.append({
        "Model": r["model"],
        "Group": r["group"],
        "Feature set": r.get("feature_set", "-"),
        "Target transform": r.get("target_transform", "none"),
        "Train rows used (fit stage)": r.get("fit_rows_used", np.nan),
        "MAE": t["mae"],
        "MSE": t.get("mse", t["rmse"] ** 2),
        "RMSE": t["rmse"],
        "R2": t["r2"],
        "Adjusted R2": t.get("adj_r2", t["r2"]),
        "Median Absolute Error": t["median_ae"],
        "Explained Variance": t["explained_variance"],
        "Max Error": t.get("max_error", np.nan),
        "Mean Bias Error": t.get("mean_bias_error", np.nan),
        "Training Time (s)": r.get("train_time_s", 0.0),
        "Validation MAE": v["mae"],
        "Test MAE": t["mae"],
        "Val MSE": v.get("mse", v["rmse"] ** 2),
        "Val RMSE": v["rmse"],
        "Val R2": v["r2"],
        "Val Adjusted R2": v.get("adj_r2", v["r2"]),
        "Val Median AE": v["median_ae"],
        "Val Explained Variance": v["explained_variance"],
        "Val Max Error": v.get("max_error", np.nan),
        "Val Mean Bias Error": v.get("mean_bias_error", np.nan),
        "Test MAE vs mean baseline (%)": (mean_t - t["mae"]) / mean_t * 100.0,
        "Test MAE vs median baseline (%)": (med_t - t["mae"]) / med_t * 100.0,
        "Test MAE vs OLD transformed GBR (%)": (old_t - t["mae"]) / old_t * 100.0 if old else np.nan,
        "Best params": str(r.get("best_params", r.get("gate_threshold", ""))),
    })

tab = pd.DataFrame(rows)
cand = tab[tab.Group.isin(["original-replica", "improved", "ensemble", "two-stage"])].sort_values("Validation MAE")
tab["_o"] = tab["Group"].map({"baseline": 1}).fillna(0)
tab = tab.sort_values(["_o", "Validation MAE"]).drop(columns="_o").reset_index(drop=True)
tab.insert(0, "Rank (val MAE)", np.arange(1, len(tab) + 1))
tab.to_csv(ARTIFACTS / "model_comparison_regression.csv", index=False)

# Best single direct estimator (has .joblib pipeline directly in models/regression/)
direct_single = tab[tab.Group.isin(["original-replica", "improved"])].sort_values("Validation MAE").iloc[0]
overall = cand.iloc[0]

def pack(row):
    r = next(x for x in allrecs if x["model"] == row["Model"])
    return {
        "model": r["model"],
        "safe_name": sf(r["model"]),
        "group": r["group"],
        "feature_set": r.get("feature_set"),
        "best_params": r.get("best_params"),
        "val": r["val"],
        "test": r["test"],
        "train_time_s": r.get("train_time_s"),
        "tuning": r.get("tuning"),
        "target_transform": r.get("target_transform"),
        "fit_rows_used": r.get("fit_rows_used"),
        "refit_rows_used": r.get("refit_rows_used"),
    }

sel = {
    "best_direct": pack(direct_single),
    "best_overall": pack(overall),
    "selection_rule": "lowest validation MAE among original replicas, improved direct models, ensembles and two-stage systems (baselines excluded from selection, reported for comparison)",
    "baselines": {k: {"val": v["val"], "test": v["test"], "val_constant": v["val_constant_minutes"], "test_constant": v["test_constant_minutes"]} for k, v in base.items()},
    "failed_models": [f["model"] for f in failed],
    "two_stage": ts,
}

def pct(old_, new_):
    return float((old_ - new_) / old_ * 100.0)

sel["improvement"] = {
    "best_direct_vs_mean_baseline_pct_test_mae": pct(mean_t, direct_single["Test MAE"]),
    "best_direct_vs_median_baseline_pct_test_mae": pct(med_t, direct_single["Test MAE"]),
    "best_direct_vs_old_transformed_gbr_replica_pct_test_mae": pct(old_t, direct_single["Test MAE"]) if old else None,
    "best_overall_vs_mean_baseline_pct_test_mae": pct(mean_t, overall["Test MAE"]),
    "best_overall_vs_median_baseline_pct_test_mae": pct(med_t, overall["Test MAE"]),
    "best_overall_vs_old_transformed_gbr_replica_pct_test_mae": pct(old_t, overall["Test MAE"]) if old else None,
    "old_transformed_gbr_replica_vs_mean_baseline_pct_test_mae": pct(mean_t, old_t) if old else None,
    "old_transformed_gbr_replica_vs_median_baseline_pct_test_mae": pct(med_t, old_t) if old else None,
    "validation_mae": {
        "mean_baseline": mean_v,
        "median_baseline": med_v,
        "old_transformed_gbr_replica": old["val"]["mae"] if old else None,
        "best_direct": direct_single["Validation MAE"],
        "best_overall": overall["Validation MAE"],
    },
    "original_project_reported_on_15k_sample": {
        "source": "hard-coded in src/model_evaluation.py (benchmark_sample.csv, 3,241 test rows)",
        "model": "Advanced Transformed GBR",
        "test_mae": 17.512, "rmse": 47.43, "r2": -0.0542, "baseline_mean_mae": 20.8344, "mae_improvement_pct_vs_mean": 15.95,
    },
}

save_json(sel, ARTIFACTS / "regression_selection.json")
pd.set_option("display.width", 250)
print(tab[["Rank (val MAE)", "Model", "Group", "Validation MAE", "Test MAE", "RMSE", "R2", "Median Absolute Error", "Mean Bias Error", "Training Time (s)"]].round(3).to_string())
print("BEST DIRECT:", direct_single["Model"], "| BEST OVERALL:", overall["Model"], "| failed:", sel["failed_models"])
