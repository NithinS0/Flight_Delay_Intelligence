"""Step 3b: class-imbalance study + feature-set ablation (validation-period scores only; test set untouched)."""
import warnings, time
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from mlcommon import *
warnings.filterwarnings("ignore")

df = load_table()
fit, val, test, train = split_views(df)
FS = feature_sets(df)
yf, yv = fit.DELAYED.values, val.DELAYED.values
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5.0, random_state=SEED, n_jobs=12, verbose=-1)

def val_report(s, y=yv):
    thr, _ = best_f1_threshold(y, s)
    m = clf_metrics(y, s, thr)
    m05 = clf_metrics(y, s, 0.5)
    return {"best_threshold": thr, "f1": m["f1"], "precision": m["precision"], "recall": m["recall"], "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
            "balanced_accuracy": m["balanced_accuracy"], "f1_at_0.5": m05["f1"], "recall_at_0.5": m05["recall"], "precision_at_0.5": m05["precision"]}

# ---------------- feature-set ablation ----------------
abl = {}
ti = time_ordered_sample(len(fit), 600_000)
for fs in ["CORE", "ENH", "HIST"]:
    cat, num = FS[fs]; cols = cat + num
    pipe = Pipeline([("pre", ordinal_pre(cat, num)), ("clf", lgb.LGBMClassifier(**PARAMS))])
    with Timer() as tm:
        pipe.fit(fit.iloc[ti][cols], yf[ti])
    s = pipe.predict_proba(val[cols])[:, 1]
    abl[fs] = {**val_report(s), "n_features": len(cols), "train_rows": int(len(ti)), "train_time_s": tm.s}
    print("ablation", fs, abl[fs], flush=True)
save_json({"description": "LightGBM fixed params, FIT-period training sample, VALIDATION-period scores", "results": abl}, METRICS_DIR / "feature_ablation_classification.json")

# ---------------- imbalance study (HIST features, 100k fit sample) ----------------
cat, num = FS["HIST"]; cols = cat + num
si = time_ordered_sample(len(fit), 100_000)
pre = ordinal_pre(cat, num)
Xs = pre.fit_transform(fit.iloc[si][cols]); ys = yf[si]
Xv = pre.transform(val[cols])
cat_idx = list(range(len(cat)))
neg, pos = int((ys == 0).sum()), int((ys == 1).sum())
res = {"sample_rows": int(len(si)), "sample_delayed_pct": float(ys.mean() * 100), "variants": {}}

def fitscore(X, y, **kw):
    p = dict(PARAMS); p.update(kw)
    m = lgb.LGBMClassifier(**p)
    with Timer() as tm:
        m.fit(X, y)
    return m.predict_proba(Xv)[:, 1], tm.s

variants = {}
variants["No resampling, no weights"] = lambda: fitscore(Xs, ys)
variants["Class weights (balanced)"] = lambda: fitscore(Xs, ys, scale_pos_weight=neg / pos)
def rus():
    X2, y2 = RandomUnderSampler(random_state=SEED).fit_resample(Xs, ys); return fitscore(X2, y2) + (len(y2),)
def rus3():
    X2, y2 = RandomUnderSampler(sampling_strategy=0.5, random_state=SEED).fit_resample(Xs, ys); return fitscore(X2, y2) + (len(y2),)
def ros():
    X2, y2 = RandomOverSampler(sampling_strategy=0.5, random_state=SEED).fit_resample(Xs, ys); return fitscore(X2, y2) + (len(y2),)
def smote():
    X2, y2 = SMOTENC(categorical_features=cat_idx, sampling_strategy=0.5, k_neighbors=5, random_state=SEED).fit_resample(Xs, ys); return fitscore(X2, y2) + (len(y2),)
variants["Random undersampling (1:1)"] = rus
variants["Random undersampling (2:1)"] = rus3
variants["Random oversampling (2:1)"] = ros
variants["SMOTENC oversampling (2:1)"] = smote
for name, fn in variants.items():
    try:
        out = fn()
        s, secs = out[0], out[1]
        rows = out[2] if len(out) > 2 else len(ys)
        res["variants"][name] = {**val_report(s), "train_rows_after_resampling": int(rows), "train_time_s": secs}
        print("imbalance", name, res["variants"][name], flush=True)
    except Exception as e:
        res["variants"][name] = {"failed": repr(e)}
        print("imbalance", name, "FAILED", e, flush=True)
save_json(res, METRICS_DIR / "imbalance_study.json")
print("done")
