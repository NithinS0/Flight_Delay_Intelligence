"""Step 4: regression experiments (target = DEP_DELAY minutes).

Same leakage-safe protocol as the classification step (tune on FIT -> validate -> refit on FIT+VAL -> single TEST evaluation).
Primary selection metric: validation MAE.
"""
import sys, time, warnings, joblib
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import numpy as np, pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
import lightgbm as lgb
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
from mlcommon import *
from src.regression import signed_log_transform, signed_log_inverse

warnings.filterwarnings("ignore")
NJ = 12
df = load_table()
fit, val, test, train = split_views(df)
FS = feature_sets(df)
yf, yv, yt, ytr = [d.DEP_DELAY.values.astype(np.float64) for d in (fit, val, test, train)]
print("rows fit/val/test:", len(fit), len(val), len(test), flush=True)
REG_DIR = MODELS_DIR / "regression"; REG_DIR.mkdir(exist_ok=True, parents=True)
cv3 = TimeSeriesSplit(n_splits=3)


def slog(reg):
    return TransformedTargetRegressor(regressor=reg, func=signed_log_transform, inverse_func=signed_log_inverse, check_inverse=False)


SPECS = {}
def spec(name, group, fs, enc, make, wrap=False, cap=None, space=None, n_iter=0, tune_cap=300_000, note=""):
    SPECS[name] = dict(name=name, group=group, fs=fs, enc=enc, make=make, wrap=wrap, cap=cap, space=space, n_iter=n_iter, tune_cap=tune_cap, note=note)

OLD_CAP = 100_000
OLD_GBR = dict(n_estimators=100, subsample=0.8, min_samples_split=20, min_samples_leaf=5, max_features="sqrt", max_depth=4, learning_rate=0.05, random_state=SEED)
# ---- original-project replicas ----
spec("OLD Linear Regression (Ridge)", "original-replica", "CORE", "ohe", lambda: Ridge(alpha=10.0), cap=OLD_CAP)
spec("OLD Random Forest Regressor", "original-replica", "CORE", "ohe", lambda: RandomForestRegressor(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=NJ), cap=OLD_CAP)
spec("OLD Gradient Boosting Regressor", "original-replica", "CORE", "ohe", lambda: GradientBoostingRegressor(**OLD_GBR), cap=OLD_CAP,
     note="hyper-parameters assumed = those of the original transformed GBR, without target transform")
spec("OLD Transformed GBR (signed-log)", "original-replica", "ADV", "ohe", lambda: slog(GradientBoostingRegressor(**OLD_GBR)), wrap=True, cap=OLD_CAP)
spec("OLD Robust GBR (L1 loss)", "original-replica", "ADV", "ohe", lambda: GradientBoostingRegressor(**{**OLD_GBR, "loss": "absolute_error"}), cap=OLD_CAP,
     note="original implementation not in repository; reconstructed as L1-loss GBR on the advanced features")

# ---- improved (history features) ----
spec("Ridge (HIST)", "improved", "HIST", "ohe", lambda: Ridge(), space=dict(alpha=loguniform(0.1, 1000)), n_iter=4, tune_cap=60_000, cap=200_000)
spec("Random Forest (tuned, L2)", "improved", "HIST", "ord", lambda: RandomForestRegressor(random_state=SEED, n_jobs=NJ),
     space=dict(n_estimators=[100, 180], max_depth=randint(8, 20), min_samples_leaf=randint(30, 200), max_features=[0.3, 0.5, "sqrt"]),
     n_iter=4, tune_cap=40_000, cap=150_000)
spec("Extra Trees (tuned, L2)", "improved", "HIST", "ord", lambda: ExtraTreesRegressor(random_state=SEED, n_jobs=NJ),
     space=dict(n_estimators=[100, 180], max_depth=randint(8, 20), min_samples_leaf=randint(30, 200), max_features=[0.3, 0.5, "sqrt"]),
     n_iter=4, tune_cap=40_000, cap=150_000)
gbr_space = dict(n_estimators=randint(80, 150), learning_rate=loguniform(0.04, 0.15), max_depth=randint(3, 6), min_samples_leaf=randint(30, 200),
                 subsample=uniform(0.7, 0.3), max_features=["sqrt", 0.5])
spec("Gradient Boosting (L2, signed-log target)", "improved", "HIST", "ord", lambda: slog(GradientBoostingRegressor(loss="squared_error", random_state=SEED)),
     wrap=True, space=gbr_space, n_iter=3, tune_cap=30_000, cap=80_000)
spec("Gradient Boosting (Huber)", "improved", "HIST", "ord", lambda: GradientBoostingRegressor(loss="huber", random_state=SEED),
     space={**gbr_space, "alpha": [0.85, 0.9]}, n_iter=3, tune_cap=30_000, cap=80_000)
spec("Gradient Boosting (L1)", "improved", "HIST", "ord", lambda: GradientBoostingRegressor(loss="absolute_error", random_state=SEED),
     space=gbr_space, n_iter=3, tune_cap=30_000, cap=80_000)
hgb_space = dict(max_iter=randint(100, 250), learning_rate=loguniform(0.04, 0.15), max_leaf_nodes=randint(31, 127), min_samples_leaf=randint(30, 200),
                 l2_regularization=loguniform(1e-2, 10))
spec("HistGradientBoosting (L2)", "improved", "HIST", "ord", lambda: HistGradientBoostingRegressor(loss="squared_error", early_stopping=False, random_state=SEED),
     space=hgb_space, n_iter=4, tune_cap=60_000, cap=200_000)
spec("HistGradientBoosting (L2, signed-log target)", "improved", "HIST", "ord", lambda: slog(HistGradientBoostingRegressor(loss="squared_error", early_stopping=False, random_state=SEED)),
     wrap=True, space=hgb_space, n_iter=4, tune_cap=60_000, cap=200_000)
spec("HistGradientBoosting (L1)", "improved", "HIST", "ord", lambda: HistGradientBoostingRegressor(loss="absolute_error", early_stopping=False, random_state=SEED),
     space=hgb_space, n_iter=4, tune_cap=60_000, cap=200_000)
lgb_space = dict(n_estimators=randint(150, 350), learning_rate=loguniform(0.03, 0.12), num_leaves=randint(31, 127), min_child_samples=randint(30, 200),
                 subsample=uniform(0.7, 0.3), colsample_bytree=uniform(0.6, 0.4), reg_lambda=loguniform(0.1, 20))
LG = dict(random_state=SEED, n_jobs=NJ, verbose=-1, subsample_freq=1)
spec("LightGBM (L2)", "improved", "HIST", "ord", lambda: lgb.LGBMRegressor(objective="regression", **LG), space=lgb_space, n_iter=4, tune_cap=60_000, cap=300_000)
spec("LightGBM (L2, signed-log target)", "improved", "HIST", "ord", lambda: slog(lgb.LGBMRegressor(objective="regression", **LG)), wrap=True, space=lgb_space, n_iter=4, tune_cap=60_000, cap=300_000)
spec("LightGBM (L1)", "improved", "HIST", "ord", lambda: lgb.LGBMRegressor(objective="regression_l1", **LG), space=lgb_space, n_iter=4, tune_cap=60_000, cap=300_000)
spec("LightGBM (Huber)", "improved", "HIST", "ord", lambda: lgb.LGBMRegressor(objective="huber", **LG), space={**lgb_space, "alpha": [10.0, 20.0]}, n_iter=4, tune_cap=60_000, cap=300_000)
spec("LightGBM (Quantile 0.5)", "improved", "HIST", "ord", lambda: lgb.LGBMRegressor(objective="quantile", alpha=0.5, **LG), space=lgb_space, n_iter=4, tune_cap=60_000, cap=300_000)
xgb_space = dict(n_estimators=randint(150, 300), learning_rate=loguniform(0.03, 0.12), max_depth=randint(4, 8), min_child_weight=randint(5, 40),
                 subsample=uniform(0.7, 0.3), colsample_bytree=uniform(0.6, 0.4), reg_lambda=loguniform(0.5, 30))
XG = dict(random_state=SEED, n_jobs=NJ, tree_method="hist")
spec("XGBoost (L2)", "improved", "HIST", "ord", lambda: xgb.XGBRegressor(objective="reg:squarederror", **XG), space=xgb_space, n_iter=4, tune_cap=60_000, cap=250_000)
spec("XGBoost (L1)", "improved", "HIST", "ord", lambda: xgb.XGBRegressor(objective="reg:absoluteerror", **XG), space=xgb_space, n_iter=4, tune_cap=60_000, cap=250_000)
spec("XGBoost (Pseudo-Huber)", "improved", "HIST", "ord", lambda: xgb.XGBRegressor(objective="reg:pseudohubererror", huber_slope=15.0, **XG), space=xgb_space, n_iter=4, tune_cap=60_000, cap=250_000)
cb_space = dict(iterations=randint(200, 400), learning_rate=loguniform(0.04, 0.15), depth=randint(5, 8), l2_leaf_reg=loguniform(1, 20))
if CatBoostRegressor is not None:
    spec("CatBoost (MAE)", "improved", "HIST", "ord", lambda: CatBoostRegressor(loss_function="MAE", **CBK), space=cb_space, n_iter=3, tune_cap=40_000, cap=150_000)
    spec("CatBoost (RMSE)", "improved", "HIST", "ord", lambda: CatBoostRegressor(loss_function="RMSE", **CBK), space=cb_space, n_iter=3, tune_cap=40_000, cap=150_000)


def pre_for(sp):
    cat, num = FS[sp["fs"]]
    return ordinal_pre(cat, num) if sp["enc"] == "ord" else onehot_pre(cat, num, min_frequency=30)


def run(sp):
    name = sp["name"]; safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    out_json = WORK_DIR / f"reg_{safe}.json"
    if out_json.exists():
        print("cached", name, flush=True); return load_json(out_json)
    t_all = time.time()
    cat, num = FS[sp["fs"]]; cols = cat + num
    prefix = "reg__regressor__" if sp["wrap"] else "reg__"
    inner = "regressor__" if sp["wrap"] else ""
    rec = {"model": name, "group": sp["group"], "feature_set": sp["fs"], "n_features": len(cols), "encoding": sp["enc"], "note": sp["note"],
           "target_transform": "signed-log" if sp["wrap"] else "none"}
    pipe = Pipeline([("pre", pre_for(sp)), ("reg", sp["make"]())])
    best = {}
    if sp["space"]:
        ti = time_ordered_sample(len(fit), sp["tune_cap"])
        rs = RandomizedSearchCV(pipe, {prefix + k: v for k, v in sp["space"].items()}, n_iter=sp["n_iter"], cv=cv3, scoring="neg_mean_absolute_error",
                                random_state=SEED, n_jobs=1, refit=False, error_score="raise")
        with Timer() as tm:
            rs.fit(fit.iloc[ti][cols], yf[ti])
        best = {k.replace(prefix, ""): (v.item() if hasattr(v, "item") else v) for k, v in rs.best_params_.items()}
        rec["tuning"] = {"method": "RandomizedSearchCV", "cv": "TimeSeriesSplit(n_splits=3)", "scoring": "neg_mean_absolute_error", "n_iter": sp["n_iter"],
                         "tuning_rows": int(len(ti)), "best_cv_mae": float(-rs.best_score_), "seconds": tm.s, "random_state": SEED}
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
    rec["train_time_s"] = tm.s; rec["fit_rows_used"] = int(len(fi))
    pv = pf.predict(val[cols])
    rec["val"] = reg_metrics(yv, pv)
    ri = time_ordered_sample(len(train), sp["cap"])
    pt = build()
    with Timer() as tm2:
        pt.fit(train.iloc[ri][cols], ytr[ri])
    rec["refit_time_s"] = tm2.s; rec["refit_rows_used"] = int(len(ri))
    ps = pt.predict(test[cols])
    rec["test"] = reg_metrics(yt, ps)
    np.savez_compressed(WORK_DIR / f"reg_pred_{safe}.npz", val=pv.astype(np.float32), test=ps.astype(np.float32))
    joblib.dump({"pipeline": pt, "columns": cols, "feature_set": sp["fs"]}, REG_DIR / f"{safe}.joblib", compress=3)
    rec["total_seconds"] = time.time() - t_all
    save_json(rec, out_json)
    print(f"[{name}] VAL MAE={rec['val']['mae']:.3f} RMSE={rec['val']['rmse']:.2f} R2={rec['val']['r2']:.4f} | TEST MAE={rec['test']['mae']:.3f} RMSE={rec['test']['rmse']:.2f} R2={rec['test']['r2']:.4f} | {rec['total_seconds']:.0f}s", flush=True)
    return rec


def baselines():
    out_json = WORK_DIR / "reg_baselines.json"
    rec = {}
    for nm, fnc in [("Training mean baseline", np.mean), ("Training median baseline", np.median)]:
        cv_, ct_ = float(fnc(yf)), float(fnc(ytr))
        pv = np.full(len(yv), cv_); ps = np.full(len(yt), ct_)
        rec[nm] = {"model": nm, "group": "baseline", "val_constant_minutes": cv_, "test_constant_minutes": ct_, "val": reg_metrics(yv, pv), "test": reg_metrics(yt, ps), "train_time_s": 0.0}
        np.savez_compressed(WORK_DIR / f"reg_pred_{nm.replace(' ', '_')}.npz", val=pv.astype(np.float32), test=ps.astype(np.float32))
    save_json(rec, out_json); print("baselines", {k: (v["val"]["mae"], v["test"]["mae"]) for k, v in rec.items()}, flush=True)


if __name__ == "__main__":
    baselines()
    wanted = sys.argv[1:] or list(SPECS)
    for n in wanted:
        try:
            run(SPECS[n])
        except Exception as e:
            import traceback; traceback.print_exc()
            save_json({"model": n, "failed": True, "error": repr(e)}, WORK_DIR / f"reg_FAILED_{n.replace(' ', '_')}.json")
            print(f"[{n}] FAILED: {e!r}", flush=True)
