"""Step 4c: Complete the full Regression Model Suite (Linear, Ridge, Lasso, ElasticNet, Decision Tree,
Random Forest, Extra Trees, Gradient Boosting L2/Signed-Log/Huber/L1, HistGradientBoosting L2/Signed-Log/L1,
LightGBM L2/Signed-Log/L1/Huber/Quantile-0.5, XGBoost L2/L1/Pseudo-Huber, LinearSVR, Ensemble Regressor,
and Two-Stage Delay Duration Estimator).
Also backfills all 9 Regression Metrics (MAE, MSE, RMSE, R², Adjusted R², Median AE, Explained Variance,
Max Error, Mean Bias Error) across all models, computes Exact TreeSHAP for regression, and saves Q-Q & SHAP charts.
"""
import sys, time, warnings, joblib
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import randint, uniform, loguniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, explained_variance_score, max_error
)
import lightgbm as lgb
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlcommon import *
from chartstyle import *
from src.regression import signed_log_transform, signed_log_inverse

warnings.filterwarnings("ignore")
t0 = time.time()
NJ = 12

df = load_table()
fit, val, test, train = split_views(df)
FS = feature_sets(df)
yf, yv, yt, ytr = [d.DEP_DELAY.values.astype(np.float64) for d in (fit, val, test, train)]
REG_DIR = MODELS_DIR / "regression"
REG_DIR.mkdir(exist_ok=True, parents=True)
cv3 = TimeSeriesSplit(n_splits=3)

def slog(reg):
    return TransformedTargetRegressor(regressor=reg, func=signed_log_transform, inverse_func=signed_log_inverse, check_inverse=False)

def full_reg_metrics(y_true, y_pred, n_features=74):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    p = max(int(n_features), 1)
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    adj_r2 = float(1.0 - (1.0 - r2) * (n - 1.0) / max(n - p - 1.0, 1.0))
    med_ae = float(median_absolute_error(y_true, y_pred))
    exp_var = float(explained_variance_score(y_true, y_pred))
    mx_err = float(max_error(y_true, y_pred))
    mbe = float(np.mean(y_pred - y_true))  # Mean Bias Error (predicted - actual)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "adj_r2": adj_r2,
        "median_ae": med_ae,
        "explained_variance": exp_var,
        "max_error": mx_err,
        "mean_bias_error": mbe,
    }

SPECS = {}
def spec(name, group, fs, enc, make, wrap=False, cap=None, space=None, n_iter=0, tune_cap=60_000, note=""):
    SPECS[name] = dict(name=name, group=group, fs=fs, enc=enc, make=make, wrap=wrap, cap=cap, space=space, n_iter=n_iter, tune_cap=tune_cap, note=note)

# Fast linear, tree, SVR, and gradient boosting specs
spec("Linear Regression (OLS)", "improved", "HIST", "ohe", lambda: LinearRegression(n_jobs=NJ), cap=200_000, note="Ordinary Least Squares linear baseline on HIST features")
spec("Lasso Regression (tuned)", "improved", "HIST", "ohe", lambda: Lasso(max_iter=1000, random_state=SEED), space=dict(alpha=loguniform(0.01, 10.0)), n_iter=3, tune_cap=50_000, cap=200_000, note="L1-penalized sparse linear regression")
spec("ElasticNet Regression (tuned)", "improved", "HIST", "ohe", lambda: ElasticNet(max_iter=1000, random_state=SEED), space=dict(alpha=loguniform(0.01, 10.0), l1_ratio=[0.2, 0.5, 0.8]), n_iter=3, tune_cap=50_000, cap=200_000, note="Combined L1+L2 regularized linear regression")
spec("Decision Tree Regressor (tuned)", "improved", "HIST", "ord", lambda: DecisionTreeRegressor(random_state=SEED), space=dict(max_depth=randint(6, 16), min_samples_leaf=randint(50, 300)), n_iter=4, tune_cap=50_000, cap=200_000, note="Single recursive regression tree")
spec("Support Vector Regressor (LinearSVR, L1)", "improved", "HIST", "ohe", lambda: LinearSVR(loss="epsilon_insensitive", epsilon=0.0, max_iter=2000, random_state=SEED), space=dict(C=loguniform(0.01, 5.0)), n_iter=3, tune_cap=40_000, cap=150_000, note="Linear Support Vector Regression optimizing L1 epsilon-insensitive loss")

gbr_space = dict(n_estimators=randint(70, 120), learning_rate=loguniform(0.05, 0.14), max_depth=randint(3, 6), min_samples_leaf=randint(40, 180), subsample=uniform(0.7, 0.3), max_features=["sqrt", 0.5])
spec("Gradient Boosting (L2, signed-log target)", "improved", "HIST", "ord", lambda: slog(GradientBoostingRegressor(loss="squared_error", random_state=SEED)), wrap=True, space=gbr_space, n_iter=2, tune_cap=20_000, cap=35_000)
spec("Gradient Boosting (Huber)", "improved", "HIST", "ord", lambda: GradientBoostingRegressor(loss="huber", random_state=SEED), space={**gbr_space, "alpha": [0.85, 0.9]}, n_iter=2, tune_cap=20_000, cap=35_000)
spec("Gradient Boosting (L1)", "improved", "HIST", "ord", lambda: GradientBoostingRegressor(loss="absolute_error", random_state=SEED), space=gbr_space, n_iter=2, tune_cap=20_000, cap=35_000)

hgb_space = dict(max_iter=randint(120, 240), learning_rate=loguniform(0.04, 0.14), max_leaf_nodes=randint(31, 127), min_samples_leaf=randint(30, 200), l2_regularization=loguniform(1e-2, 10))
spec("HistGradientBoosting (L2)", "improved", "HIST", "ord", lambda: HistGradientBoostingRegressor(loss="squared_error", early_stopping=False, random_state=SEED), space=hgb_space, n_iter=3, tune_cap=60_000, cap=200_000)
spec("HistGradientBoosting (L2, signed-log target)", "improved", "HIST", "ord", lambda: slog(HistGradientBoostingRegressor(loss="squared_error", early_stopping=False, random_state=SEED)), wrap=True, space=hgb_space, n_iter=3, tune_cap=60_000, cap=200_000)
spec("HistGradientBoosting (L1)", "improved", "HIST", "ord", lambda: HistGradientBoostingRegressor(loss="absolute_error", early_stopping=False, random_state=SEED), space=hgb_space, n_iter=3, tune_cap=60_000, cap=200_000)

lgb_space = dict(n_estimators=randint(150, 320), learning_rate=loguniform(0.03, 0.12), num_leaves=randint(31, 127), min_child_samples=randint(30, 200), subsample=uniform(0.7, 0.3), colsample_bytree=uniform(0.6, 0.4), reg_lambda=loguniform(0.1, 20))
LG = dict(random_state=SEED, n_jobs=NJ, verbose=-1, subsample_freq=1)
spec("LightGBM (L2)", "improved", "HIST", "ord", lambda: lgb.LGBMRegressor(objective="regression", **LG), space=lgb_space, n_iter=4, tune_cap=60_000, cap=300_000)
spec("LightGBM (L2, signed-log target)", "improved", "HIST", "ord", lambda: slog(lgb.LGBMRegressor(objective="regression", **LG)), wrap=True, space=lgb_space, n_iter=4, tune_cap=60_000, cap=300_000)
spec("LightGBM (L1)", "improved", "HIST", "ord", lambda: lgb.LGBMRegressor(objective="regression_l1", **LG), space=lgb_space, n_iter=4, tune_cap=60_000, cap=300_000)
spec("LightGBM (Huber)", "improved", "HIST", "ord", lambda: lgb.LGBMRegressor(objective="huber", **LG), space={**lgb_space, "alpha": [10.0, 20.0]}, n_iter=4, tune_cap=60_000, cap=300_000)
spec("LightGBM (Quantile 0.5)", "improved", "HIST", "ord", lambda: lgb.LGBMRegressor(objective="quantile", alpha=0.5, **LG), space=lgb_space, n_iter=3, tune_cap=60_000, cap=300_000)

xgb_space = dict(n_estimators=randint(150, 280), learning_rate=loguniform(0.03, 0.12), max_depth=randint(4, 8), min_child_weight=randint(5, 40), subsample=uniform(0.7, 0.3), colsample_bytree=uniform(0.6, 0.4), reg_lambda=loguniform(0.5, 30))
XG = dict(random_state=SEED, n_jobs=NJ, tree_method="hist")
spec("XGBoost (L2)", "improved", "HIST", "ord", lambda: xgb.XGBRegressor(objective="reg:squarederror", **XG), space=xgb_space, n_iter=3, tune_cap=60_000, cap=250_000)
spec("XGBoost (L1)", "improved", "HIST", "ord", lambda: xgb.XGBRegressor(objective="reg:absoluteerror", **XG), space=xgb_space, n_iter=3, tune_cap=60_000, cap=250_000)
spec("XGBoost (Pseudo-Huber)", "improved", "HIST", "ord", lambda: xgb.XGBRegressor(objective="reg:pseudohubererror", huber_slope=15.0, **XG), space=xgb_space, n_iter=3, tune_cap=60_000, cap=250_000)

def pre_for(sp):
    cat, num = FS[sp["fs"]]
    return ordinal_pre(cat, num) if sp["enc"] == "ord" else onehot_pre(cat, num, min_frequency=30)

def safe_fn(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("+", "plus")

def run_spec(sp):
    name = sp["name"]
    safe = safe_fn(name)
    out_json = WORK_DIR / f"reg_{safe}.json"
    if out_json.exists():
        print("cached", name, flush=True)
        return load_json(out_json)
    t_all = time.time()
    cat, num = FS[sp["fs"]]
    cols = cat + num
    prefix = "reg__regressor__" if sp["wrap"] else "reg__"
    inner = "regressor__" if sp["wrap"] else ""
    rec = {
        "model": name, "group": sp["group"], "feature_set": sp["fs"], "n_features": len(cols),
        "encoding": sp["enc"], "note": sp["note"], "target_transform": "signed-log" if sp["wrap"] else "none"
    }
    pipe = Pipeline([("pre", pre_for(sp)), ("reg", sp["make"]())])
    best = {}
    if sp["space"] and sp["n_iter"] > 0:
        ti = time_ordered_sample(len(fit), sp["tune_cap"])
        rs = RandomizedSearchCV(
            pipe, {prefix + k: v for k, v in sp["space"].items()},
            n_iter=sp["n_iter"], cv=cv3, scoring="neg_mean_absolute_error",
            random_state=SEED, n_jobs=1, refit=False, error_score="raise"
        )
        with Timer() as tm:
            rs.fit(fit.iloc[ti][cols], yf[ti])
        best = {k.replace(prefix, ""): (v.item() if hasattr(v, "item") else v) for k, v in rs.best_params_.items()}
        rec["tuning"] = {
            "method": "RandomizedSearchCV", "cv": "TimeSeriesSplit(n_splits=3)",
            "scoring": "neg_mean_absolute_error", "n_iter": sp["n_iter"],
            "tuning_rows": int(len(ti)), "best_cv_mae": float(-rs.best_score_),
            "seconds": tm.s, "random_state": SEED
        }
        print(f"[{name}] tuned {tm.s:.0f}s cv MAE={-rs.best_score_:.3f} {best}", flush=True)
    rec["best_params"] = best

    def build():
        est = sp["make"]()
        if best:
            est.set_params(**{inner + k: v for k, v in best.items()})
        return Pipeline([("pre", pre_for(sp)), ("reg", est)])

    fi = time_ordered_sample(len(fit), sp["cap"])
    pf = build()
    with Timer() as tm:
        pf.fit(fit.iloc[fi][cols], yf[fi])
    rec["train_time_s"] = tm.s
    rec["fit_rows_used"] = int(len(fi))
    pv = pf.predict(val[cols])
    rec["val"] = full_reg_metrics(yv, pv, len(cols))

    ri = time_ordered_sample(len(train), sp["cap"])
    pt = build()
    with Timer() as tm2:
        pt.fit(train.iloc[ri][cols], ytr[ri])
    rec["refit_time_s"] = tm2.s
    rec["refit_rows_used"] = int(len(ri))
    ps = pt.predict(test[cols])
    rec["test"] = full_reg_metrics(yt, ps, len(cols))

    np.savez_compressed(WORK_DIR / f"reg_pred_{safe}.npz", val=pv.astype(np.float32), test=ps.astype(np.float32))
    joblib.dump({"pipeline": pt, "columns": cols, "feature_set": sp["fs"]}, REG_DIR / f"{safe}.joblib", compress=3)
    rec["total_seconds"] = time.time() - t_all
    save_json(rec, out_json)
    print(f"[{name}] VAL MAE={rec['val']['mae']:.3f} RMSE={rec['val']['rmse']:.2f} | TEST MAE={rec['test']['mae']:.3f} RMSE={rec['test']['rmse']:.2f} | {rec['total_seconds']:.0f}s", flush=True)
    return rec

for nm, sp in SPECS.items():
    run_spec(sp)

# ==============================================================================
# 2. BUILD ENSEMBLE REGRESSOR (Blending Top L1 / Signed-Log Models on Validation MAE)
# ==============================================================================
ens_reg_out = WORK_DIR / "reg_Ensemble_Regressor_LGB-L1plusXGB-L1plusHGB-L1.json"
if not ens_reg_out.exists():
    print("Building Ensemble Regressor (LGB-L1 + XGB-L1 + HGB-L1)...", flush=True)
    p_lgb = np.load(WORK_DIR / "reg_pred_LightGBM_L1.npz")
    p_xgb = np.load(WORK_DIR / "reg_pred_XGBoost_L1.npz")
    p_hgb = np.load(WORK_DIR / "reg_pred_HistGradientBoosting_L1.npz")
    r_l = load_json(WORK_DIR / "reg_LightGBM_L1.json")
    r_x = load_json(WORK_DIR / "reg_XGBoost_L1.json")
    r_h = load_json(WORK_DIR / "reg_HistGradientBoosting_L1.json")

    best_w, best_mae = (0.45, 0.35, 0.20), 1e9
    for wl in [0.3, 0.4, 0.5, 0.6]:
        for wx in [0.2, 0.3, 0.4]:
            wh = round(1.0 - wl - wx, 2)
            if wh < 0.0:
                continue
            pv_b = wl * p_lgb["val"] + wx * p_xgb["val"] + wh * p_hgb["val"]
            m_v = float(mean_absolute_error(yv, pv_b))
            if m_v < best_mae:
                best_mae, best_w = m_v, (wl, wx, wh)

    wl, wx, wh = best_w
    pv_ens = wl * p_lgb["val"].astype(np.float64) + wx * p_xgb["val"].astype(np.float64) + wh * p_hgb["val"].astype(np.float64)
    ps_ens = wl * p_lgb["test"].astype(np.float64) + wx * p_xgb["test"].astype(np.float64) + wh * p_hgb["test"].astype(np.float64)
    tot_tr = r_l["train_time_s"] + r_x["train_time_s"] + r_h["train_time_s"]

    rec_er = {
        "model": "Ensemble Regressor (LGB-L1+XGB-L1+HGB-L1)",
        "group": "ensemble", "feature_set": "HIST", "n_features": 74, "encoding": "ord",
        "target_transform": "none (L1 loss blend)",
        "note": f"Convex L1-loss ensemble blending LightGBM-L1 ({wl}), XGBoost-L1 ({wx}), and HistGradientBoosting-L1 ({wh})",
        "best_params": {"weights_lgb_xgb_hgb": list(best_w)},
        "tuning": {"method": "Validation MAE simplex grid search", "best_val_mae": best_mae},
        "train_time_s": tot_tr, "fit_rows_used": r_l["fit_rows_used"],
        "refit_time_s": r_l["refit_time_s"] + r_x["refit_time_s"] + r_h["refit_time_s"], "refit_rows_used": r_l["refit_rows_used"],
        "val": full_reg_metrics(yv, pv_ens, 74),
        "test": full_reg_metrics(yt, ps_ens, 74),
        "total_seconds": tot_tr,
    }
    np.savez_compressed(WORK_DIR / "reg_pred_Ensemble_Regressor_LGB-L1plusXGB-L1plusHGB-L1.npz", val=pv_ens.astype(np.float32), test=ps_ens.astype(np.float32))
    save_json(rec_er, ens_reg_out)
    print(f"[Ensemble Regressor] VAL MAE={rec_er['val']['mae']:.3f} | TEST MAE={rec_er['test']['mae']:.3f}", flush=True)

# ==============================================================================
# 3. BACKFILL ALL 9 METRICS ON ALL EXISTING CACHED REGRESSION FILES & BASELINES
# ==============================================================================
print("Upgrading all cached reg_*.json and reg_baselines.json with full 9 metrics...", flush=True)
for p in sorted(WORK_DIR.glob("reg_*.json")):
    if p.name == "reg_baselines.json":
        base = load_json(p)
        for nm, b_rec in base.items():
            pred_f = WORK_DIR / f"reg_pred_{nm.replace(' ', '_')}.npz"
            if pred_f.exists():
                pr = np.load(pred_f)
                b_rec["val"] = full_reg_metrics(yv, pr["val"], 0)
                b_rec["test"] = full_reg_metrics(yt, pr["test"], 0)
        save_json(base, p)
        continue
    r = load_json(p)
    if r.get("failed"):
        continue
    if "adj_r2" in r["test"] and "mean_bias_error" in r["test"] and "mse" in r["test"]:
        continue
    safe = p.stem.replace("reg_", "")
    pr_path = WORK_DIR / f"reg_pred_{safe}.npz"
    if pr_path.exists():
        pr = np.load(pr_path)
        nf = r.get("n_features", 74)
        r["val"] = full_reg_metrics(yv, pr["val"], nf)
        r["test"] = full_reg_metrics(yt, pr["test"], nf)
        save_json(r, p)
        print(f"  Upgraded {r['model']}")

# ==============================================================================
# 4. COMPUTE EXACT TREESHAP FOR REGRESSION & RESIDUAL Q-Q PLOT
# ==============================================================================
print("Computing Exact TreeSHAP for Regression & Residual Q-Q plot...", flush=True)
cat_h, num_h = FS["HIST"]
cols_h = cat_h + num_h
lgb_bundle = joblib.load(REG_DIR / "LightGBM_L1.joblib")
lgb_pipe = lgb_bundle["pipeline"]
lgb_est = lgb_pipe.named_steps["reg"]
feat_names = [c.split("__", 1)[1] for c in lgb_pipe.named_steps["pre"].get_feature_names_out()]

shap_idx = time_ordered_sample(len(test), 10_000)
X_test_tr = lgb_pipe.named_steps["pre"].transform(test.iloc[shap_idx][cols_h])
reg_shap_contribs = lgb_est.predict(X_test_tr, pred_contrib=True)  # (10000, n_features + 1)
reg_shap_vals = reg_shap_contribs[:, :-1]
reg_mean_abs_shap = np.mean(np.abs(reg_shap_vals), axis=0)
reg_shap_pct = reg_mean_abs_shap / max(reg_mean_abs_shap.sum(), 1e-12) * 100.0
gain_imp = lgb_est.booster_.feature_importance("gain").astype(np.float64)
gain_imp_pct = gain_imp / max(gain_imp.sum(), 1e-12) * 100.0

reg_fi_df = pd.DataFrame({
    "feature": feat_names,
    "importance_pct": np.round(gain_imp_pct, 4),
    "mean_abs_shap_minutes": np.round(reg_mean_abs_shap, 4),
    "shap_share_pct": np.round(reg_shap_pct, 4),
}).sort_values("mean_abs_shap_minutes", ascending=False).reset_index(drop=True)
reg_fi_df.to_csv(ARTIFACTS / "feature_importance_regression.csv", index=False)

# Plot Residual Q-Q Plot + Regression TreeSHAP Summary
pred_best_l1 = np.load(WORK_DIR / "reg_pred_LightGBM_L1.npz")["test"].astype(np.float64)
res_l1 = yt - pred_best_l1
rng = np.random.default_rng(42)
res_sample = rng.choice(res_l1, size=15_000, replace=False)

fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
stats.probplot(res_sample, dist="norm", plot=ax[0])
ax[0].set_title("Residual Normal Q-Q Plot — LightGBM (L1)\n(Heavy positive tail = unpredicted extreme black-swan delays)")
ax[0].get_lines()[0].set_color(BLUE); ax[0].get_lines()[0].set_markersize(2.5)
ax[0].get_lines()[1].set_color(RED)

top20_rs = reg_fi_df.head(20).iloc[::-1]
ax[1].barh(top20_rs["feature"], top20_rs["mean_abs_shap_minutes"], color=GREEN)
for i, v in enumerate(top20_rs["mean_abs_shap_minutes"]):
    ax[1].text(v + 0.015, i, f"{v:.2f}m", va="center", fontsize=8)
ax[1].set_title("Exact TreeSHAP Feature Attribution — LightGBM (L1)\n(Mean |SHAP| in minutes of departure delay, n = 10,000 test flights)")
ax[1].set_xlabel("Mean |SHAP value| (minutes)")
save(fig, "reg_12_residual_qq_and_shap")

print(f"Step 4c finished in {time.time() - t0:.1f}s!", flush=True)

