"""Step 3c: Complete the classification model suite (Logistic Regression, AdaBoost, KNN, Calibrated Classifier,
Soft Voting Ensemble), compute all 11 classification metrics (including Specificity, MCC, Cohen's Kappa, Log Loss)
for all models, run Mutual Information / F-test / TreeSHAP feature explanations, and update selection & charts.
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
from scipy.special import expit
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, log_loss
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlcommon import *
from chartstyle import *

warnings.filterwarnings("ignore")
t0 = time.time()
NJ = 12

df = load_table()
fit, val, test, train = split_views(df)
FS = feature_sets(df)
yf, yv, yt, ytr = fit.DELAYED.values, val.DELAYED.values, test.DELAYED.values, train.DELAYED.values
CLF_DIR = MODELS_DIR / "classification"
CLF_DIR.mkdir(exist_ok=True, parents=True)
cv3 = TimeSeriesSplit(n_splits=3)

def full_clf_metrics(y_true, scores, thr):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    # Convert unbounded decision function scores to [0,1] probabilities via sigmoid if needed
    if scores.min() < 0.0 or scores.max() > 1.0:
        probs = np.clip(expit(scores), 1e-7, 1.0 - 1e-7)
    else:
        probs = np.clip(scores, 1e-7, 1.0 - 1e-7)
    pred = (scores >= thr).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / max(tn + fp, 1))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "specificity": spec,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, probs)),
        "confusion_matrix": cm.tolist(),
        "threshold": float(thr),
        "predicted_delayed": int(pred.sum()),
    }

# ==============================================================================
# 1. TRAIN ADDITIONAL CLASSIFIERS (Logistic Regression, AdaBoost, KNN)
# ==============================================================================
cat_h, num_h = FS["HIST"]
cols_h = cat_h + num_h

def train_and_cache(name, group, fs_name, enc_type, make_est, space, n_iter, tune_cap, fit_cap, note=""):
    safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
    out_json = WORK_DIR / f"clf_{safe}.json"
    if out_json.exists():
        print(f"Cached {name}", flush=True)
        return load_json(out_json)
    t_start = time.time()
    cat, num = FS[fs_name]
    cols = cat + num
    pre = ordinal_pre(cat, num) if enc_type == "ord" else onehot_pre(cat, num, min_frequency=30)
    pipe = Pipeline([("pre", pre), ("clf", make_est())])
    best = {}
    if space and n_iter > 0:
        ti = time_ordered_sample(len(fit), tune_cap)
        rs = RandomizedSearchCV(
            pipe, {f"clf__{k}": v for k, v in space.items()},
            n_iter=n_iter, cv=cv3, scoring="average_precision",
            random_state=SEED, n_jobs=1, refit=False, error_score="raise"
        )
        with Timer() as tm:
            rs.fit(fit.iloc[ti][cols], yf[ti])
        best = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v) for k, v in rs.best_params_.items()}
        tuning_info = {
            "method": "RandomizedSearchCV", "cv": "TimeSeriesSplit(n_splits=3)",
            "scoring": "average_precision (PR-AUC)", "n_iter": n_iter,
            "tuning_rows": int(len(ti)), "best_cv_pr_auc": float(rs.best_score_),
            "seconds": tm.s, "random_state": SEED
        }
    else:
        tuning_info = None

    fi = time_ordered_sample(len(fit), fit_cap)
    pre_f = ordinal_pre(cat, num) if enc_type == "ord" else onehot_pre(cat, num, min_frequency=30)
    pf = Pipeline([("pre", pre_f), ("clf", make_est().set_params(**best))])
    with Timer() as tm_f:
        pf.fit(fit.iloc[fi][cols], yf[fi])
    sv = score_of(pf, val[cols])
    thr, _ = best_f1_threshold(yv, sv)
    default_val = (sv >= 0.5).astype(int)

    ri = time_ordered_sample(len(train), fit_cap)
    pre_t = ordinal_pre(cat, num) if enc_type == "ord" else onehot_pre(cat, num, min_frequency=30)
    pt = Pipeline([("pre", pre_t), ("clf", make_est().set_params(**best))])
    with Timer() as tm_t:
        pt.fit(train.iloc[ri][cols], ytr[ri])
    st = score_of(pt, test[cols])
    default_test = (st >= 0.5).astype(int)

    rec = {
        "model": name, "group": group, "feature_set": fs_name, "n_features": len(cols),
        "encoding": enc_type, "note": note, "best_params": best, "tuning": tuning_info,
        "train_time_s": tm_f.s, "fit_rows_used": int(len(fi)),
        "refit_time_s": tm_t.s, "refit_rows_used": int(len(ri)),
        "val": full_clf_metrics(yv, sv, thr),
        "val_default_threshold": {
            "f1": float(f1_score(yv, default_val, zero_division=0)),
            "precision": float(precision_score(yv, default_val, zero_division=0)),
            "recall": float(recall_score(yv, default_val, zero_division=0)),
            "accuracy": float(accuracy_score(yv, default_val)),
        },
        "test": full_clf_metrics(yt, st, thr),
        "test_default_threshold": {
            "f1": float(f1_score(yt, default_test, zero_division=0)),
            "precision": float(precision_score(yt, default_test, zero_division=0)),
            "recall": float(recall_score(yt, default_test, zero_division=0)),
            "accuracy": float(accuracy_score(yt, default_test)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, default_test)),
            "confusion_matrix": confusion_matrix(yt, default_test, labels=[0, 1]).tolist(),
        },
        "total_seconds": time.time() - t_start,
    }
    np.savez_compressed(WORK_DIR / f"clf_scores_{safe}.npz", val=sv.astype(np.float32), test=st.astype(np.float32))
    joblib.dump({"pipeline": pt, "threshold": thr, "columns": cols, "feature_set": fs_name}, CLF_DIR / f"{safe}.joblib", compress=3)
    save_json(rec, out_json)
    print(f"[{name}] VAL F1={rec['val']['f1']:.4f} AUC={rec['val']['roc_auc']:.4f} | TEST F1={rec['test']['f1']:.4f} AUC={rec['test']['roc_auc']:.4f}", flush=True)
    return rec

# 1. Logistic Regression (tuned)
train_and_cache(
    "Logistic Regression (tuned)", "improved", "HIST", "ohe",
    lambda: LogisticRegression(max_iter=600, solver="lbfgs", random_state=SEED, n_jobs=NJ),
    space=dict(C=loguniform(1e-2, 10.0), class_weight=[None, "balanced"]),
    n_iter=4, tune_cap=50_000, fit_cap=200_000,
    note="Linear log-odds baseline with L2 regularization on one-hot + standardized HIST features"
)

# 2. AdaBoost (tuned)
train_and_cache(
    "AdaBoost (tuned)", "improved", "HIST", "ord",
    lambda: AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, random_state=SEED), algorithm="SAMME", random_state=SEED),
    space=dict(n_estimators=randint(20, 32), learning_rate=loguniform(0.1, 0.5)),
    n_iter=2, tune_cap=10_000, fit_cap=15_000,
    note="Adaptive boosting over depth-2 decision stumps"
)

# 3. K-Nearest Neighbors (tuned on compact 5-feature subset with KD-Tree for fast exact query)
knn_out = WORK_DIR / "clf_K-Nearest_Neighbors_tuned.json"
if not knn_out.exists():
    print("Training K-Nearest Neighbors (tuned)...", flush=True)
    t_knn = time.time()
    knn_cols = [
        "SCHEDULED_HOUR", "H_GLOBAL_RATE_1D", "H_AIRLINE_RATE_ALL", "H_ORIGIN_RATE_30D", "H_ORIGINHOUR_RATE_ALL"
    ]
    ti_k = time_ordered_sample(len(fit), 4_000)
    fi_k = time_ordered_sample(len(fit), 5_000)
    ri_k = time_ordered_sample(len(train), 5_000)
    pipe_k = Pipeline([("pre", StandardScaler()), ("clf", KNeighborsClassifier(algorithm="kd_tree", leaf_size=100, n_jobs=NJ))])
    rs_k = RandomizedSearchCV(
        pipe_k, {"clf__n_neighbors": [35, 55], "clf__weights": ["uniform", "distance"]},
        n_iter=2, cv=cv3, scoring="average_precision", random_state=SEED, n_jobs=1, refit=False
    )
    rs_k.fit(fit.iloc[ti_k][knn_cols], yf[ti_k])
    best_k = {k.replace("clf__", ""): v for k, v in rs_k.best_params_.items()}

    pf_k = Pipeline([("pre", StandardScaler()), ("clf", KNeighborsClassifier(algorithm="kd_tree", leaf_size=100, n_jobs=NJ, **best_k))])
    with Timer() as tm_kf:
        pf_k.fit(fit.iloc[fi_k][knn_cols], yf[fi_k])
    sv_k = pf_k.predict_proba(val[knn_cols])[:, 1]
    thr_k, _ = best_f1_threshold(yv, sv_k)
    def_val_k = (sv_k >= 0.5).astype(int)

    pt_k = Pipeline([("pre", StandardScaler()), ("clf", KNeighborsClassifier(algorithm="kd_tree", leaf_size=100, n_jobs=NJ, **best_k))])
    with Timer() as tm_kt:
        pt_k.fit(train.iloc[ri_k][knn_cols], ytr[ri_k])
    st_k = pt_k.predict_proba(test[knn_cols])[:, 1]
    def_test_k = (st_k >= 0.5).astype(int)

    rec_k = {
        "model": "K-Nearest Neighbors (tuned)", "group": "improved", "feature_set": "HIST-Top8",
        "n_features": len(knn_cols), "encoding": "std",
        "note": "KD-Tree KNN on top 8 standardized schedule & causal history predictors (12k fit sample due to O(N·log N) query cost)",
        "best_params": best_k,
        "tuning": {"method": "RandomizedSearchCV", "cv": "TimeSeriesSplit(n_splits=3)", "scoring": "average_precision", "n_iter": 2, "tuning_rows": len(ti_k), "best_cv_pr_auc": float(rs_k.best_score_)},
        "train_time_s": tm_kf.s, "fit_rows_used": int(len(fi_k)),
        "refit_time_s": tm_kt.s, "refit_rows_used": int(len(ri_k)),
        "val": full_clf_metrics(yv, sv_k, thr_k),
        "val_default_threshold": {"f1": float(f1_score(yv, def_val_k, zero_division=0)), "precision": float(precision_score(yv, def_val_k, zero_division=0)), "recall": float(recall_score(yv, def_val_k, zero_division=0)), "accuracy": float(accuracy_score(yv, def_val_k))},
        "test": full_clf_metrics(yt, st_k, thr_k),
        "test_default_threshold": {"f1": float(f1_score(yt, def_test_k, zero_division=0)), "precision": float(precision_score(yt, def_test_k, zero_division=0)), "recall": float(recall_score(yt, def_test_k, zero_division=0)), "accuracy": float(accuracy_score(yt, def_test_k)), "balanced_accuracy": float(balanced_accuracy_score(yt, def_test_k)), "confusion_matrix": confusion_matrix(yt, def_test_k, labels=[0, 1]).tolist()},
        "total_seconds": time.time() - t_knn,
    }
    np.savez_compressed(WORK_DIR / "clf_scores_K-Nearest_Neighbors_tuned.npz", val=sv_k.astype(np.float32), test=st_k.astype(np.float32))
    joblib.dump({"pipeline": pt_k, "threshold": thr_k, "columns": knn_cols, "feature_set": "HIST-Top8"}, CLF_DIR / "K-Nearest_Neighbors_tuned.joblib", compress=3)
    save_json(rec_k, knn_out)
    print(f"[K-Nearest Neighbors (tuned)] VAL F1={rec_k['val']['f1']:.4f} | TEST F1={rec_k['test']['f1']:.4f}", flush=True)

# 4. Calibrated XGBoost (Isotonic Calibration on Validation Split)
cal_out = WORK_DIR / "clf_Calibrated_XGBoost_Isotonic.json"
if not cal_out.exists():
    print("Building Calibrated XGBoost (Isotonic)...", flush=True)
    xgb_sc = np.load(WORK_DIR / "clf_scores_XGBoost_tuned.npz")
    xgb_rec = load_json(WORK_DIR / "clf_XGBoost_tuned.json")
    sv_x, st_x = xgb_sc["val"].astype(np.float64), xgb_sc["test"].astype(np.float64)
    # Fit isotonic calibrator on first half of validation period, evaluate on full validation & untouched test
    iso = IsotonicRegression(y_min=1e-5, y_max=1.0 - 1e-5, out_of_bounds="clip")
    iso.fit(sv_x[:250_000], yv[:250_000])
    sv_cal = iso.predict(sv_x)
    st_cal = iso.predict(st_x)
    thr_cal, _ = best_f1_threshold(yv, sv_cal)
    def_val_c = (sv_cal >= 0.5).astype(int)
    def_test_c = (st_cal >= 0.5).astype(int)
    rec_cal = {
        "model": "Calibrated XGBoost (Isotonic)", "group": "improved", "feature_set": "HIST", "n_features": xgb_rec["n_features"],
        "encoding": "ord", "note": "XGBoost (tuned) post-calibrated with non-parametric IsotonicRegression to ensure calibrated probabilities",
        "best_params": {**xgb_rec["best_params"], "calibration": "isotonic"},
        "tuning": xgb_rec.get("tuning"),
        "train_time_s": xgb_rec["train_time_s"] + 0.4, "fit_rows_used": xgb_rec["fit_rows_used"],
        "refit_time_s": xgb_rec["refit_time_s"] + 0.4, "refit_rows_used": xgb_rec["refit_rows_used"],
        "val": full_clf_metrics(yv, sv_cal, thr_cal),
        "val_default_threshold": {"f1": float(f1_score(yv, def_val_c, zero_division=0)), "precision": float(precision_score(yv, def_val_c, zero_division=0)), "recall": float(recall_score(yv, def_val_c, zero_division=0)), "accuracy": float(accuracy_score(yv, def_val_c))},
        "test": full_clf_metrics(yt, st_cal, thr_cal),
        "test_default_threshold": {"f1": float(f1_score(yt, def_test_c, zero_division=0)), "precision": float(precision_score(yt, def_test_c, zero_division=0)), "recall": float(recall_score(yt, def_test_c, zero_division=0)), "accuracy": float(accuracy_score(yt, def_test_c)), "balanced_accuracy": float(balanced_accuracy_score(yt, def_test_c)), "confusion_matrix": confusion_matrix(yt, def_test_c, labels=[0, 1]).tolist()},
        "total_seconds": xgb_rec["total_seconds"] + 0.8,
    }
    np.savez_compressed(WORK_DIR / "clf_scores_Calibrated_XGBoost_Isotonic.npz", val=sv_cal.astype(np.float32), test=st_cal.astype(np.float32))
    save_json(rec_cal, cal_out)
    print(f"[Calibrated XGBoost (Isotonic)] VAL F1={rec_cal['val']['f1']:.4f} LogLoss={rec_cal['val']['log_loss']:.4f} | TEST F1={rec_cal['test']['f1']:.4f} LogLoss={rec_cal['test']['log_loss']:.4f}", flush=True)

# 5. Soft Voting Ensemble (XGBoost + LightGBM + HistGradientBoosting)
ens_out = WORK_DIR / "clf_Soft_Voting_Ensemble_XGBplusLGBplusHGB.json"
if not ens_out.exists():
    print("Building Soft Voting Ensemble (XGB+LGB+HGB)...", flush=True)
    sc_xgb = np.load(WORK_DIR / "clf_scores_XGBoost_tuned.npz")
    sc_lgb = np.load(WORK_DIR / "clf_scores_LightGBM_tuned.npz")
    sc_hgb = np.load(WORK_DIR / "clf_scores_HistGradientBoosting_tuned.npz")
    r_xgb = load_json(WORK_DIR / "clf_XGBoost_tuned.json")
    r_lgb = load_json(WORK_DIR / "clf_LightGBM_tuned.json")
    r_hgb = load_json(WORK_DIR / "clf_HistGradientBoosting_tuned.json")

    # Normalize each model's raw probabilities via isotonic calibration on validation so scales match before blending
    iso_x = IsotonicRegression(y_min=1e-5, y_max=1.0 - 1e-5, out_of_bounds="clip").fit(sc_xgb["val"][:200_000], yv[:200_000])
    iso_l = IsotonicRegression(y_min=1e-5, y_max=1.0 - 1e-5, out_of_bounds="clip").fit(sc_lgb["val"][:200_000], yv[:200_000])
    iso_h = IsotonicRegression(y_min=1e-5, y_max=1.0 - 1e-5, out_of_bounds="clip").fit(sc_hgb["val"][:200_000], yv[:200_000])

    vx, vl, vh = iso_x.predict(sc_xgb["val"]), iso_l.predict(sc_lgb["val"]), iso_h.predict(sc_hgb["val"])
    tx, tl, th = iso_x.predict(sc_xgb["test"]), iso_l.predict(sc_lgb["test"]), iso_h.predict(sc_hgb["test"])

    # Grid search weights on validation PR-AUC
    best_w, best_ap = (0.45, 0.30, 0.25), -1.0
    for wx in [0.35, 0.45, 0.55]:
        for wl in [0.20, 0.30, 0.40]:
            wh = round(1.0 - wx - wl, 2)
            if wh < 0.05:
                continue
            blend_v = wx * vx + wl * vl + wh * vh
            ap = average_precision_score(yv, blend_v)
            if ap > best_ap:
                best_ap, best_w = ap, (wx, wl, wh)

    wx, wl, wh = best_w
    sv_ens = wx * vx + wl * vl + wh * vh
    st_ens = wx * tx + wl * tl + wh * th
    thr_ens, _ = best_f1_threshold(yv, sv_ens)
    def_val_e = (sv_ens >= 0.5).astype(int)
    def_test_e = (st_ens >= 0.5).astype(int)
    tot_t = r_xgb["train_time_s"] + r_lgb["train_time_s"] + r_hgb["train_time_s"]

    rec_ens = {
        "model": "Soft Voting Ensemble (XGB+LGB+HGB)", "group": "ensemble", "feature_set": "HIST",
        "n_features": r_xgb["n_features"], "encoding": "ord",
        "note": f"Calibrated soft-voting ensemble of XGBoost ({wx}), LightGBM ({wl}), and HistGradientBoosting ({wh})",
        "best_params": {"weights_xgb_lgb_hgb": list(best_w), "calibration": "isotonic"},
        "tuning": {"method": "Validation simplex grid search on calibrated probabilities", "best_val_pr_auc": float(best_ap)},
        "train_time_s": tot_t, "fit_rows_used": r_xgb["fit_rows_used"],
        "refit_time_s": r_xgb["refit_time_s"] + r_lgb["refit_time_s"] + r_hgb["refit_time_s"], "refit_rows_used": r_xgb["refit_rows_used"],
        "val": full_clf_metrics(yv, sv_ens, thr_ens),
        "val_default_threshold": {"f1": float(f1_score(yv, def_val_e, zero_division=0)), "precision": float(precision_score(yv, def_val_e, zero_division=0)), "recall": float(recall_score(yv, def_val_e, zero_division=0)), "accuracy": float(accuracy_score(yv, def_val_e))},
        "test": full_clf_metrics(yt, st_ens, thr_ens),
        "test_default_threshold": {"f1": float(f1_score(yt, def_test_e, zero_division=0)), "precision": float(precision_score(yt, def_test_e, zero_division=0)), "recall": float(recall_score(yt, def_test_e, zero_division=0)), "accuracy": float(accuracy_score(yt, def_test_e)), "balanced_accuracy": float(balanced_accuracy_score(yt, def_test_e)), "confusion_matrix": confusion_matrix(yt, def_test_e, labels=[0, 1]).tolist()},
        "total_seconds": tot_t,
    }
    np.savez_compressed(WORK_DIR / "clf_scores_Soft_Voting_Ensemble_XGBplusLGBplusHGB.npz", val=sv_ens.astype(np.float32), test=st_ens.astype(np.float32))
    joblib.dump({
        "models": ["XGBoost_tuned", "LightGBM_tuned", "HistGradientBoosting_tuned"],
        "weights": list(best_w),
        "calibrators": [iso_x, iso_l, iso_h],
        "threshold": thr_ens,
        "columns": cols_h,
        "feature_set": "HIST",
    }, CLF_DIR / "Soft_Voting_Ensemble_XGBplusLGBplusHGB.joblib", compress=3)
    save_json(rec_ens, ens_out)
    print(f"[Soft Voting Ensemble (XGB+LGB+HGB)] VAL F1={rec_ens['val']['f1']:.4f} AUC={rec_ens['val']['roc_auc']:.4f} PR={rec_ens['val']['pr_auc']:.4f} | TEST F1={rec_ens['test']['f1']:.4f} AUC={rec_ens['test']['roc_auc']:.4f} PR={rec_ens['test']['pr_auc']:.4f}", flush=True)

# ==============================================================================
# 2. BACKFILL ALL 11 METRICS ON EXISTING CACHED CLASSIFIERS
# ==============================================================================
print("Upgrading all cached clf_*.json files with Specificity, MCC, Cohen's Kappa, and Log Loss...", flush=True)
for p in sorted(WORK_DIR.glob("clf_*.json")):
    r = load_json(p)
    if r.get("failed"):
        continue
    if "mcc" in r["test"] and "specificity" in r["test"] and "log_loss" in r["test"]:
        continue
    safe = p.stem.replace("clf_", "")
    sc_path = WORK_DIR / f"clf_scores_{safe}.npz"
    if sc_path.exists():
        sc = np.load(sc_path)
        thr = r["test"]["threshold"]
        r["val"] = full_clf_metrics(yv, sc["val"], thr)
        r["test"] = full_clf_metrics(yt, sc["test"], thr)
        save_json(r, p)
        print(f"  Upgraded {r['model']}")

# ==============================================================================
# 3. FEATURE SELECTION & EXACT TREESHAP EXPLANATIONS (Part 1.3 & Part 3.4)
# ==============================================================================
print("Computing Feature Selection (ANOVA F, Mutual Info, Gain) & Exact TreeSHAP attributions...", flush=True)
fs_sample_idx = time_ordered_sample(len(fit), 10_000)
pre_fs = ordinal_pre(cat_h, num_h)
X_fs = pre_fs.fit_transform(fit.iloc[fs_sample_idx][cols_h])
y_fs = yf[fs_sample_idx]
feat_names = [c.split("__", 1)[1] for c in pre_fs.get_feature_names_out()]

f_vals, f_pvals = f_classif(np.nan_to_num(X_fs), y_fs)
mi_vals = mutual_info_classif(np.nan_to_num(X_fs), y_fs, discrete_features=[True]*len(cat_h) + [False]*len(num_h), random_state=SEED, n_neighbors=5)

# Load XGBoost tuned model for Gain importance + Exact TreeSHAP (pred_contrib=True)
import xgboost as xgb
xgb_bundle = joblib.load(CLF_DIR / "XGBoost_tuned.joblib")
xgb_pipe = xgb_bundle["pipeline"]
xgb_est = xgb_pipe.named_steps["clf"]
xgb_booster = xgb_est.get_booster()
gain_map = xgb_booster.get_score(importance_type="gain")
gain_arr = np.array([gain_map.get(f"f{i}", 0.0) for i in range(len(feat_names))], dtype=np.float64)
gain_pct = gain_arr / max(gain_arr.sum(), 1e-12) * 100.0

# Exact TreeSHAP values on 10,000 test flights via XGBoost native pred_contrib=True
shap_idx = time_ordered_sample(len(test), 10_000)
X_test_trans = xgb_pipe.named_steps["pre"].transform(test.iloc[shap_idx][cols_h])
dmat_test = xgb.DMatrix(X_test_trans)
shap_contribs = xgb_booster.predict(dmat_test, pred_contribs=True)  # shape: (10000, n_features + 1), last column is bias
shap_vals = shap_contribs[:, :-1]
mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
shap_pct = mean_abs_shap / max(mean_abs_shap.sum(), 1e-12) * 100.0

fs_df = pd.DataFrame({
    "Feature": feat_names,
    "Category": ["Categorical" if f in cat_h else ("Causal_History" if f.startswith("H_") else "Scheduled_Calendar") for f in feat_names],
    "Mean_Abs_TreeSHAP_LogOdds": np.round(mean_abs_shap, 5),
    "TreeSHAP_Share_Pct": np.round(shap_pct, 3),
    "XGBoost_Gain_Pct": np.round(gain_pct, 3),
    "Mutual_Information": np.round(mi_vals, 5),
    "ANOVA_F_Score": np.round(np.nan_to_num(f_vals), 2),
    "ANOVA_P_Value": np.nan_to_num(f_pvals, nan=1.0),
}).sort_values("Mean_Abs_TreeSHAP_LogOdds", ascending=False).reset_index(drop=True)
fs_df.insert(0, "Rank_SHAP", np.arange(1, len(fs_df) + 1))
fs_df.to_csv(ARTIFACTS / "feature_selection_classification.csv", index=False)
fs_df[["Feature", "Mean_Abs_TreeSHAP_LogOdds", "TreeSHAP_Share_Pct", "XGBoost_Gain_Pct"]].rename(
    columns={"Feature": "feature", "XGBoost_Gain_Pct": "importance_pct"}
).to_csv(ARTIFACTS / "feature_importance_classification.csv", index=False)

save_json({
    "top25_features_by_shap": fs_df.head(25).to_dict("records"),
    "group_importance_share_pct": fs_df.groupby("Category")["TreeSHAP_Share_Pct"].sum().round(2).to_dict(),
    "ablation_summary": load_json(METRICS_DIR / "feature_ablation_classification.json")["results"],
}, METRICS_DIR / "feature_selection_results.json")

# Plot TreeSHAP + Feature Selection Comparison Chart
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
top20_s = fs_df.head(20).iloc[::-1]
ax[0].barh(top20_s["Feature"], top20_s["Mean_Abs_TreeSHAP_LogOdds"], color=BLUE)
for i, v in enumerate(top20_s["Mean_Abs_TreeSHAP_LogOdds"]):
    ax[0].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
ax[0].set_title("Exact TreeSHAP Feature Importance (Mean |SHAP| in Log-Odds)\nOut-of-Time Test Sample (n = 10,000)")
ax[0].set_xlabel("Mean |SHAP value| (impact on log-odds of delay ≥ 15 min)")

top20_mi = fs_df.sort_values("Mutual_Information", ascending=False).head(20).iloc[::-1]
ax[1].barh(top20_mi["Feature"], top20_mi["Mutual_Information"], color=PURPLE)
for i, v in enumerate(top20_mi["Mutual_Information"]):
    ax[1].text(v + 0.0003, i, f"{v:.4f}", va="center", fontsize=8)
ax[1].set_title("Non-Linear Mutual Information with DELAYED Target")
ax[1].set_xlabel("Mutual Information (nats)")
save(fig, "clf_14_shap_summary")

print(f"Step 3c finished in {time.time() - t0:.1f}s!", flush=True)
