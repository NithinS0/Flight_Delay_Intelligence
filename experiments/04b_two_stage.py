"""Step 4b: two-stage delay model  (Stage 1: delayed? -> Stage 2: duration) vs direct regression.

Stage-1 scores come from the selected classifier (already produced by 03_classification.py).
Stage-2 regressors (LightGBM, fixed a-priori hyper-parameters -- NOT tuned on test) are trained on:
    * flights that truly were on-time (<15 min)   -> "on-time regressor"
    * flights that truly were delayed (>=15 min)  -> "delayed regressor"
Prediction: hard gate  pred = delayed_regressor(x) if stage1_score >= t else on_time_regressor(x).
Gate thresholds evaluated:  t_F1  (maximises delayed-class F1 on validation)  and  t_MAE (minimises validation MAE).
"""
import sys, warnings, joblib
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import max_error, mean_squared_error
from mlcommon import *
warnings.filterwarnings("ignore")

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
    return {
        "mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "adj_r2": adj_r2,
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
        "mean_bias_error": float(np.mean(y_pred - y_true)),
    }

clf_sel = load_json(ARTIFACTS / "classification_selection.json")
safe = clf_sel["safe_name"]
sc = np.load(WORK_DIR / f"clf_scores_{safe}.npz")
sv, st, thr_f1 = sc["val"].astype(float), sc["test"].astype(float), clf_sel["threshold"]

df = load_table()
fit, val, test, train = split_views(df)
FS = feature_sets(df); cat, num = FS["HIST"]; cols = cat + num
yf, yv, yt, ytr = [d.DEP_DELAY.values.astype(float) for d in (fit, val, test, train)]
dfit, dtr = fit.DELAYED.values, train.DELAYED.values
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
         reg_lambda=5.0, random_state=SEED, n_jobs=12, verbose=-1)
res = {"stage1_model": clf_sel["model"], "stage1_threshold_f1": thr_f1, "variants": {}}


def fit_stage2(X, y, flag, objective):
    models = {}
    fi = time_ordered_sample(len(X), 300_000)
    Xs, ys, fs = X.iloc[fi], y[fi], flag[fi]
    for cls in (0, 1):
        m = fs == cls
        pipe = Pipeline([("pre", ordinal_pre(cat, num)), ("reg", lgb.LGBMRegressor(objective=objective, **P))])
        pipe.fit(Xs[m], ys[m]); models[cls] = pipe
    return models

out_pred = {}
for obj, label in [("regression_l1", "L1"), ("regression", "L2")]:
    with Timer() as tm:
        m_fit = fit_stage2(fit[cols], yf, dfit, obj)
    pv0, pv1 = m_fit[0].predict(val[cols]), m_fit[1].predict(val[cols])
    m_tr = fit_stage2(train[cols], ytr, dtr, obj)
    ps0, ps1 = m_tr[0].predict(test[cols]), m_tr[1].predict(test[cols])
    # threshold minimising validation MAE
    grid = np.unique(np.quantile(sv, np.linspace(0.05, 0.995, 120)))
    maes = [np.mean(np.abs(yv - np.where(sv >= t, pv1, pv0))) for t in grid]
    t_mae = float(grid[int(np.argmin(maes))])
    for tname, t in [("F1-optimal gate", thr_f1), ("MAE-optimal gate", t_mae)]:
        name = f"Two-stage (LightGBM {label}, {tname})"
        pv = np.where(sv >= t, pv1, pv0); ps = np.where(st >= t, ps1, ps0)
        res["variants"][name] = {"model": name, "group": "two-stage", "feature_set": "HIST", "n_features": len(cols), "gate_threshold": float(t), "stage2_objective": label,
                                 "val": full_reg_metrics(yv, pv, len(cols)), "test": full_reg_metrics(yt, ps, len(cols)), "train_time_s": tm.s, "fit_rows_used": 300_000, "refit_rows_used": 300_000,
                                 "val_share_routed_delayed": float((sv >= t).mean()), "test_share_routed_delayed": float((st >= t).mean())}
        out_pred[name] = (pv, ps)
        print(name, "VAL MAE", round(res["variants"][name]["val"]["mae"], 3), "TEST MAE", round(res["variants"][name]["test"]["mae"], 3), flush=True)
    # oracle (perfect Stage-1) -- upper bound of what Stage 1 could bring; uses true labels, reported as diagnostic only, never selected
    pv_or = np.where(val.DELAYED.values == 1, pv1, pv0); ps_or = np.where(test.DELAYED.values == 1, ps1, ps0)
    res.setdefault("oracle_diagnostic_not_a_model", {})[label] = {"val_mae": float(np.mean(np.abs(yv - pv_or))), "test_mae": float(np.mean(np.abs(yt - ps_or)))}
    # stage-2 conditional accuracy diagnostics
    dv, dt = val.DELAYED.values == 1, test.DELAYED.values == 1
    res.setdefault("stage2_conditional_mae", {})[label] = {
        "val_delayed_only_mae": float(np.mean(np.abs(yv[dv] - pv1[dv]))), "val_ontime_only_mae": float(np.mean(np.abs(yv[~dv] - pv0[~dv]))),
        "test_delayed_only_mae": float(np.mean(np.abs(yt[dt] - ps1[dt]))), "test_ontime_only_mae": float(np.mean(np.abs(yt[~dt] - ps0[~dt])))}
    if label == "L1":
        joblib.dump({"ontime": m_tr[0], "delayed": m_tr[1], "columns": cols, "stage1": clf_sel["model"], "gate_threshold_f1": thr_f1, "gate_threshold_mae": t_mae},
                    MODELS_DIR / "regression" / "two_stage_lightgbm_l1.joblib", compress=3)
for name, (pv, ps) in out_pred.items():
    np.savez_compressed(WORK_DIR / f"reg_pred_{name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.npz", val=pv.astype(np.float32), test=ps.astype(np.float32))
save_json(res, METRICS_DIR / "two_stage_results.json")
print("two-stage done")
