"""Step 3: classification experiments (DELAYED = DEP_DELAY >= 15).

Protocol (no test leakage):
  1. hyper-parameters tuned with RandomizedSearchCV + TimeSeriesSplit(3) on the FIT period only (scoring = PR-AUC),
  2. model re-fitted on the FIT period with the chosen parameters -> scored on the VALIDATION period,
  3. decision threshold = value that maximises delayed-class F1 on VALIDATION,
  4. model refitted on FIT+VAL (= original "training" period) -> scored ONCE on the untouched TEST period using the
     validation-chosen threshold.
Resume-safe: finished models are cached in artifacts/work/clf_<name>.json and skipped on re-run.
"""
import sys, time, warnings, joblib
import numpy as np, pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
import lightgbm as lgb
import xgboost as xgb
from mlcommon import *

warnings.filterwarnings("ignore")
NJ = 12
df = load_table()
fit, val, test, train = split_views(df)
FS = feature_sets(df)
yf, yv, yt, ytr = fit.DELAYED.values, val.DELAYED.values, test.DELAYED.values, train.DELAYED.values
print("rows fit/val/test:", len(fit), len(val), len(test), flush=True)
CLF_DIR = MODELS_DIR / "classification"; CLF_DIR.mkdir(exist_ok=True, parents=True)

cv3 = TimeSeriesSplit(n_splits=3)


def P(pre, est):
    return Pipeline([("pre", pre), ("clf", est)])


SPECS = {}
def spec(name, group, fs, enc, make, cap=None, space=None, n_iter=0, tune_cap=300_000, note=""):
    SPECS[name] = dict(name=name, group=group, fs=fs, enc=enc, make=make, cap=cap, space=space, n_iter=n_iter, tune_cap=tune_cap, note=note)

# ---------------- original-project replicas (identical features & hyper-parameters, re-run on the new data) -----------
OLD_CAP = 100_000
spec("OLD Decision Tree", "original-replica", "CORE", "ohe",
     lambda: DecisionTreeClassifier(max_depth=12, min_samples_split=50, class_weight="balanced", random_state=SEED), cap=OLD_CAP)
spec("OLD Random Forest", "original-replica", "CORE", "ohe",
     lambda: RandomForestClassifier(n_estimators=100, min_samples_split=40, min_samples_leaf=40, max_features="sqrt", max_depth=16,
                                    class_weight="balanced_subsample", random_state=SEED, n_jobs=NJ), cap=OLD_CAP)
spec("OLD Linear SVM", "original-replica", "CORE", "ohe",
     lambda: LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, max_iter=2000, dual=False), cap=OLD_CAP)

# ---------------- improved models (history features) ----------------
spec("Decision Tree (tuned)", "improved", "HIST", "ord", lambda: DecisionTreeClassifier(random_state=SEED),
     space=dict(max_depth=randint(6, 20), min_samples_leaf=randint(20, 400), class_weight=[None, "balanced"]),
     n_iter=6, tune_cap=40_000, cap=150_000)
spec("Random Forest (tuned)", "improved", "HIST", "ord", lambda: RandomForestClassifier(random_state=SEED, n_jobs=NJ),
     space=dict(n_estimators=[100, 180], max_depth=randint(10, 22), min_samples_leaf=randint(20, 150), max_features=["sqrt", 0.3],
                class_weight=["balanced_subsample", None]), n_iter=4, tune_cap=40_000, cap=150_000)
spec("Extra Trees (tuned)", "improved", "HIST", "ord", lambda: ExtraTreesClassifier(random_state=SEED, n_jobs=NJ),
     space=dict(n_estimators=[100, 180], max_depth=randint(10, 22), min_samples_leaf=randint(20, 150), max_features=["sqrt", 0.3],
                class_weight=["balanced_subsample", None]), n_iter=4, tune_cap=40_000, cap=150_000)
spec("Gradient Boosting (sklearn, tuned)", "improved", "HIST", "ord", lambda: GradientBoostingClassifier(random_state=SEED),
     space=dict(n_estimators=randint(80, 150), learning_rate=loguniform(0.04, 0.15), max_depth=randint(3, 6), min_samples_leaf=randint(30, 200),
                subsample=uniform(0.7, 0.3), max_features=["sqrt", 0.4]), n_iter=3, tune_cap=30_000, cap=80_000)
spec("HistGradientBoosting (tuned)", "improved", "HIST", "ord", lambda: HistGradientBoostingClassifier(random_state=SEED, early_stopping=False),
     space=dict(max_iter=randint(100, 250), learning_rate=loguniform(0.04, 0.15), max_leaf_nodes=randint(31, 127), min_samples_leaf=randint(30, 200),
                l2_regularization=loguniform(1e-2, 10), class_weight=[None, "balanced"]), n_iter=4, tune_cap=60_000, cap=200_000)
spec("LightGBM (tuned)", "improved", "HIST", "ord", lambda: lgb.LGBMClassifier(random_state=SEED, n_jobs=NJ, verbose=-1, subsample_freq=1),
     space=dict(n_estimators=randint(150, 350), learning_rate=loguniform(0.03, 0.12), num_leaves=randint(31, 127), min_child_samples=randint(30, 200),
                subsample=uniform(0.7, 0.3), colsample_bytree=uniform(0.6, 0.4), reg_lambda=loguniform(0.1, 20), scale_pos_weight=[1.0, 2.0, 4.0]),
     n_iter=5, tune_cap=60_000, cap=300_000)
spec("XGBoost (tuned)", "improved", "HIST", "ord", lambda: xgb.XGBClassifier(random_state=SEED, n_jobs=NJ, tree_method="hist", eval_metric="logloss"),
     space=dict(n_estimators=randint(150, 300), learning_rate=loguniform(0.03, 0.12), max_depth=randint(4, 8), min_child_weight=randint(5, 40),
                subsample=uniform(0.7, 0.3), colsample_bytree=uniform(0.6, 0.4), reg_lambda=loguniform(0.5, 30), scale_pos_weight=[1.0, 2.0, 4.0]),
     n_iter=4, tune_cap=60_000, cap=250_000)
spec("Linear SVM (tuned)", "improved", "HIST", "ohe", lambda: LinearSVC(random_state=SEED, max_iter=2000, dual=False),
     space=dict(C=loguniform(1e-2, 5), class_weight=[None, "balanced"]), n_iter=4, tune_cap=40_000, cap=150_000)


def pre_for(sp):
    cat, num = FS[sp["fs"]]
    return ordinal_pre(cat, num) if sp["enc"] == "ord" else onehot_pre(cat, num, min_frequency=30)


def run(sp):
    name = sp["name"]; safe = name.replace(" ", "_").replace("(", "").replace(")", "")
    out_json = WORK_DIR / f"clf_{safe}.json"
    if out_json.exists():
        print("cached", name, flush=True); return load_json(out_json)
    t_all = time.time()
    cat, num = FS[sp["fs"]]; cols = cat + num
    rec = {"model": name, "group": sp["group"], "feature_set": sp["fs"], "n_features": len(cols), "encoding": sp["enc"], "note": sp["note"]}
    pipe = P(pre_for(sp), sp["make"]())
    # ---------- 1. tuning on FIT only ----------
    best = {}
    if sp["space"]:
        ti = time_ordered_sample(len(fit), sp["tune_cap"])
        Xtu, ytu = fit.iloc[ti][cols], yf[ti]
        rs = RandomizedSearchCV(pipe, {f"clf__{k}": v for k, v in sp["space"].items()}, n_iter=sp["n_iter"], cv=cv3, scoring="average_precision",
                                random_state=SEED, n_jobs=1, refit=False, error_score="raise")
        with Timer() as tm:
            rs.fit(Xtu, ytu)
        best = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v) for k, v in rs.best_params_.items()}
        rec.update({"tuning": {"method": "RandomizedSearchCV", "cv": "TimeSeriesSplit(n_splits=3)", "scoring": "average_precision (PR-AUC)",
                               "n_iter": sp["n_iter"], "tuning_rows": int(len(ti)), "best_cv_pr_auc": float(rs.best_score_), "seconds": tm.s,
                               "random_state": SEED}})
        print(f"[{name}] tuned in {tm.s:.0f}s best cv PR-AUC={rs.best_score_:.4f} {best}", flush=True)
    rec["best_params"] = best
    est = sp["make"]().set_params(**best)
    # ---------- 2. FIT -> VALIDATION ----------
    fi = time_ordered_sample(len(fit), sp["cap"])
    pf = P(pre_for(sp), est)
    with Timer() as tm:
        pf.fit(fit.iloc[fi][cols], yf[fi])
    rec["train_time_s"] = tm.s; rec["fit_rows_used"] = int(len(fi))
    sv = score_of(pf, val[cols])
    default_val = (pf.predict(val[cols]) == 1).astype(int)
    thr, f1v = best_f1_threshold(yv, sv)
    rec["val"] = clf_metrics(yv, sv, thr)
    rec["val_default_threshold"] = {"f1": float(f1_score(yv, default_val, zero_division=0)), "precision": float(precision_score(yv, default_val, zero_division=0)),
                                    "recall": float(recall_score(yv, default_val, zero_division=0)), "accuracy": float(accuracy_score(yv, default_val))}
    # ---------- 3. FIT+VAL -> TEST (single evaluation) ----------
    ri = time_ordered_sample(len(train), sp["cap"])
    pt = P(pre_for(sp), sp["make"]().set_params(**best))
    with Timer() as tm2:
        pt.fit(train.iloc[ri][cols], ytr[ri])
    rec["refit_time_s"] = tm2.s; rec["refit_rows_used"] = int(len(ri))
    st = score_of(pt, test[cols])
    default_test = (pt.predict(test[cols]) == 1).astype(int)
    rec["test"] = clf_metrics(yt, st, thr)
    rec["test_default_threshold"] = {"f1": float(f1_score(yt, default_test, zero_division=0)), "precision": float(precision_score(yt, default_test, zero_division=0)),
                                     "recall": float(recall_score(yt, default_test, zero_division=0)), "accuracy": float(accuracy_score(yt, default_test)),
                                     "balanced_accuracy": float(balanced_accuracy_score(yt, default_test)), "confusion_matrix": confusion_matrix(yt, default_test, labels=[0, 1]).tolist()}
    rec["test_best_possible_f1_oracle"] = None   # intentionally not computed: would use test labels
    np.savez_compressed(WORK_DIR / f"clf_scores_{safe}.npz", val=sv.astype(np.float32), test=st.astype(np.float32))
    joblib.dump({"pipeline": pt, "threshold": thr, "columns": cols, "feature_set": sp["fs"]}, CLF_DIR / f"{safe}.joblib", compress=3)
    rec["total_seconds"] = time.time() - t_all
    save_json(rec, out_json)
    print(f"[{name}] VAL f1={rec['val']['f1']:.4f} P={rec['val']['precision']:.3f} R={rec['val']['recall']:.3f} AUC={rec['val']['roc_auc']:.4f} PR={rec['val']['pr_auc']:.4f} | "
          f"TEST f1={rec['test']['f1']:.4f} P={rec['test']['precision']:.3f} R={rec['test']['recall']:.3f} AUC={rec['test']['roc_auc']:.4f} PR={rec['test']['pr_auc']:.4f} | {rec['total_seconds']:.0f}s", flush=True)
    return rec


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(SPECS)
    for n in wanted:
        try:
            run(SPECS[n])
        except Exception as e:
            import traceback; traceback.print_exc()
            save_json({"model": n, "failed": True, "error": repr(e)}, WORK_DIR / f"clf_FAILED_{n.replace(' ', '_')}.json")
            print(f"[{n}] FAILED: {e!r}", flush=True)
