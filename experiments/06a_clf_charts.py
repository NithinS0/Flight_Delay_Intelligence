"""Step 6a: classification presentation charts (all from computed results)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, precision_recall_curve
from common import *
from chartstyle import *

tab = pd.read_csv(ARTIFACTS / "model_comparison_classification.csv")
sel = load_json(ARTIFACTS / "classification_selection.json")
yt = pd.read_parquet(MODEL_TABLE, columns=["DELAYED", "SPLIT"])
yv_, yt_ = yt.loc[yt.SPLIT == "val", "DELAYED"].values, yt.loc[yt.SPLIT == "test", "DELAYED"].values
safe = lambda n: n.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
scores = {m: np.load(WORK_DIR / f"clf_scores_{safe(m)}.npz") for m in tab.Model}
best = sel["model"]
best_single_safe = sel.get("best_single_safe_name", "XGBoost_tuned")
best_single_name = sel.get("best_single_model", "XGBoost (tuned)")

def color_for(row):
    if row.Model == best: return GREEN
    return SLATE if row.Group == "original-replica" else BLUE

def bar_metric(col, title, fname, pct=True, ylabel=None, extra_default=None):
    d = tab.sort_values(col)
    fig, ax = plt.subplots(figsize=(11, 7))
    vals = d[col] * (100 if pct else 1)
    ax.barh(d.Model, vals, color=[color_for(r) for r in d.itertuples()])
    for i, v in enumerate(vals): ax.text(v + (0.3 if pct else 0.003), i, f"{v:.2f}{'%' if pct else ''}", va="center", fontsize=9)
    ax.set_title(title); ax.set_xlabel(ylabel or col + (" (%)" if pct else ""))
    ax.set_xlim(0, vals.max() * 1.14)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN, label="Selected model (best validation F1)"), Patch(color=BLUE, label="Improved models"), Patch(color=SLATE, label="Original-project replicas")], loc="lower right")
    save(fig, fname)

sub = "Test period 26-Sep-2022 → 31-Aug-2023 · threshold chosen on validation"
bar_metric("Accuracy", f"Accuracy Comparison\n{sub}", "clf_02_accuracy_comparison")
bar_metric("Precision", f"Delayed-Class Precision Comparison\n{sub}", "clf_03_precision_comparison")
bar_metric("Recall", f"Delayed-Class Recall Comparison\n{sub}", "clf_04_recall_comparison")
bar_metric("F1-score", f"Delayed-Class F1-score Comparison\n{sub}", "clf_05_f1_comparison", pct=False, ylabel="F1-score")

# ROC & PR
fig, ax = plt.subplots(figsize=(9, 8))
cmap = plt.get_cmap("tab20")
for i, r in enumerate(tab.itertuples()):
    fpr, tpr, _ = roc_curve(yt_, scores[r.Model]["test"])
    ax.plot(fpr, tpr, lw=3 if r.Model == best else 1.4, color=GREEN if r.Model == best else cmap(i), label=f"{r.Model} (AUC {r._asdict()['_11']:.3f})" if False else f"{r.Model} (AUC {r[tab.columns.get_loc('ROC-AUC') + 1]:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC Curves — Test Period"); ax.legend(fontsize=7.5, loc="lower right")
save(fig, "clf_06_roc_curves")

fig, ax = plt.subplots(figsize=(9, 8))
for i, r in enumerate(tab.itertuples()):
    p, rc, _ = precision_recall_curve(yt_, scores[r.Model]["test"])
    ax.plot(rc, p, lw=3 if r.Model == best else 1.4, color=GREEN if r.Model == best else cmap(i), label=f"{r.Model} (PR-AUC {r[tab.columns.get_loc('PR-AUC') + 1]:.3f})")
ax.axhline(yt_.mean(), color="k", ls="--", lw=1, label=f"No-skill (prevalence {yt_.mean():.3f})")
ax.set_xlabel("Recall (delayed class)"); ax.set_ylabel("Precision (delayed class)"); ax.set_title("Precision-Recall Curves — Test Period"); ax.legend(fontsize=7.5, loc="upper right")
save(fig, "clf_07_pr_curves")

# confusion matrices
n = len(tab); ncol = 4; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.3 * nrow))
for ax, r in zip(axes.ravel(), tab.itertuples(index=False)):
    rd = r._asdict() if hasattr(r, "_asdict") else None
    cm = np.array([[tab.loc[tab.Model == r.Model, "Test TN"].iat[0], tab.loc[tab.Model == r.Model, "Test FP"].iat[0]],
                   [tab.loc[tab.Model == r.Model, "Test FN"].iat[0], tab.loc[tab.Model == r.Model, "Test TP"].iat[0]]])
    pr = cm / cm.sum(1, keepdims=True)
    ax.imshow(pr, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({pr[i, j] * 100:.1f}%)", ha="center", va="center", color="white" if pr[i, j] > .5 else "#0f172a", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred On-time", "Pred Delayed"], fontsize=8); ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual On-time", "Actual Delayed"], fontsize=8)
    ax.set_title(r.Model + ("  ★ BEST" if r.Model == best else ""), fontsize=9.5, color=GREEN if r.Model == best else "#0f172a"); ax.grid(False)
for ax in axes.ravel()[n:]: ax.axis("off")
fig.suptitle("Confusion Matrices — Test Period (row-normalised %, validation-optimised threshold)", fontsize=14, fontweight="bold", y=1.0)
save(fig, "clf_08_confusion_matrices")

# feature importance
jb_path = MODELS_DIR / "classification" / f"{best_single_safe}.joblib"
if not jb_path.exists():
    jb_path = MODELS_DIR / "classification" / "XGBoost_tuned.joblib"
bundle = joblib.load(jb_path)
pipe = bundle["pipeline"]; names = [c.split("__", 1)[1] for c in pipe.named_steps["pre"].get_feature_names_out()]
est = pipe.named_steps["clf"]
if hasattr(est, "booster_"):
    imp = est.booster_.feature_importance("gain"); kind = "LightGBM total gain"
elif hasattr(est, "get_booster"):
    sc = est.get_booster().get_score(importance_type="gain"); imp = np.array([sc.get(f"f{i}", 0.0) for i in range(len(names))]); kind = "XGBoost total gain"
elif hasattr(est, "feature_importances_"):
    imp = est.feature_importances_; kind = "impurity-based importance"
else:
    from sklearn.inspection import permutation_importance
    dfx = pd.read_parquet(MODEL_TABLE); 
    raise SystemExit("selected model lacks native importance; implement permutation importance")
imp = np.asarray(imp, dtype=float); imp = imp / imp.sum() * 100
fi = pd.DataFrame({"feature": names, "importance_pct": imp}).sort_values("importance_pct", ascending=False)
if not (ARTIFACTS / "feature_importance_classification.csv").exists():
    fi.to_csv(ARTIFACTS / "feature_importance_classification.csv", index=False)
top = fi.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top.feature, top.importance_pct, color=BLUE)
for i, v in enumerate(top.importance_pct): ax.text(v + .1, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_title(f"Top-20 Feature Importances — {best}\n({kind}, normalised to 100%)"); ax.set_xlabel("Relative importance (%)")
save(fig, "clf_09_feature_importance")

# threshold analysis
sv, st = scores[best]["val"], scores[best]["test"]
grid = np.quantile(sv, np.linspace(0.02, 0.995, 200))
def curves(s, y):
    P, R, F = [], [], []
    for t in grid:
        p = s >= t; tp = np.sum(p & (y == 1)); fp = np.sum(p & (y == 0)); fn = np.sum(~p & (y == 1))
        P.append(tp / max(tp + fp, 1)); R.append(tp / max(tp + fn, 1)); F.append(2 * tp / max(2 * tp + fp + fn, 1))
    return np.array(P), np.array(R), np.array(F)
pv, rv, fv = curves(sv, yv_); pt_, rt_, ft_ = curves(st, yt_)
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
ax[0].plot(grid, pv, color=BLUE, label="Precision"); ax[0].plot(grid, rv, color=RED, label="Recall"); ax[0].plot(grid, fv, color=GREEN, lw=2.5, label="F1")
ax[0].axvline(sel["threshold"], color="k", ls="--", label=f"Chosen threshold = {sel['threshold']:.3f}")
ax[0].set_title("Threshold Analysis — VALIDATION period (used for tuning)"); ax[0].set_xlabel("Decision threshold on predicted delay probability"); ax[0].set_ylabel("Delayed-class metric"); ax[0].legend()
ax[1].plot(grid, pt_, color=BLUE, ls="--", label="Precision"); ax[1].plot(grid, rt_, color=RED, ls="--", label="Recall"); ax[1].plot(grid, ft_, color=GREEN, lw=2.5, ls="--", label="F1")
ax[1].axvline(sel["threshold"], color="k", ls="--", label="Validation-chosen threshold"); ax[1].set_title("Same thresholds on TEST period (descriptive only)")
ax[1].set_xlabel("Decision threshold"); ax[1].legend()
save(fig, "clf_10_threshold_analysis")

# imbalance study & ablation
try:
    imb = load_json(METRICS_DIR / "imbalance_study.json")["variants"]
    d = pd.DataFrame(imb).T.dropna(subset=["f1"])
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(d)); w = 0.27
    ax[0].bar(x - w, d.f1.astype(float), w, label="F1 @ best val threshold", color=BLUE); ax[0].bar(x, d.precision.astype(float), w, label="Precision", color=CYAN)
    ax[0].bar(x + w, d.recall.astype(float), w, label="Recall", color=AMBER)
    ax[0].set_xticks(x); ax[0].set_xticklabels([n.replace(" (", "\n(") for n in d.index], fontsize=8, rotation=0); ax[0].set_title("Imbalance Handling (LightGBM, 100k fit sample)\nValidation period, threshold optimised"); ax[0].legend(fontsize=8)
    ax[1].bar(x - w / 2, d.pr_auc.astype(float), w, label="PR-AUC", color=PURPLE); ax[1].bar(x + w / 2, d["f1_at_0.5"].astype(float), w, label="F1 @ threshold 0.5", color=SLATE)
    ax[1].set_xticks(x); ax[1].set_xticklabels([n.replace(" (", "\n(") for n in d.index], fontsize=8); ax[1].set_title("Ranking quality (PR-AUC) vs. un-tuned threshold"); ax[1].legend(fontsize=8)
    save(fig, "clf_11_imbalance_study")
    ab = load_json(METRICS_DIR / "feature_ablation_classification.json")["results"]
    d = pd.DataFrame(ab).T
    lab = {"CORE": "Original features\n(CORE, 12)", "ENH": "+ calendar features\n(ENH)", "HIST": "+ causal history\n(HIST, 70)"}
    fig, ax = plt.subplots(figsize=(10, 5.5)); x = np.arange(3); w = .2
    for k, (col, c, nm) in enumerate([("f1", GREEN, "F1"), ("pr_auc", PURPLE, "PR-AUC"), ("roc_auc", BLUE, "ROC-AUC"), ("recall", AMBER, "Recall @F1-thr")]):
        ax.bar(x + (k - 1.5) * w, d[col].astype(float), w, label=nm, color=c)
        for i, v in enumerate(d[col].astype(float)): ax.text(i + (k - 1.5) * w, v + .004, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([lab[i] for i in d.index]); ax.set_title("Feature-Set Ablation (LightGBM fixed params, validation period)"); ax.legend()
    save(fig, "clf_12_feature_ablation")
except Exception as e:
    print("imbalance/ablation charts skipped:", e)

# old vs new
cmpd = sel["comparison_vs_old"]
if "old_rf_default_predict_vs_new_best" in cmpd:
    A, B = cmpd["old_rf_default_predict_vs_new_best"], cmpd["old_rf_valthreshold_vs_new_best"]
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy"]
    lbl = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Bal. Acc."]
    fig, ax = plt.subplots(figsize=(13, 5.5)); x = np.arange(len(keys)); w = .27
    ax.bar(x - w, [A[k]["old"] for k in keys], w, color=SLATE, label="Original RF config (default predict)")
    ax.bar(x, [B[k]["old"] for k in keys], w, color=AMBER, label="Original RF config (val-optimised threshold)")
    ax.bar(x + w, [A[k]["new"] for k in keys], w, color=GREEN, label=f"New best: {best}")
    for off, src, kk in [(-w, A, "old"), (0, B, "old"), (w, A, "new")]:
        for i, k in enumerate(keys): ax.text(i + off, src[k][kk] + .01, f"{src[k][kk]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(lbl); ax.set_title("Old vs Improved Classifier — Test Period (same data, same split)"); ax.legend(fontsize=8); ax.set_ylim(0, 1.05)
    save(fig, "clf_13_old_vs_new")
print("classification charts done")
