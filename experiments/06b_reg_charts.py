"""Step 6b: regression presentation charts (all from computed predictions/results)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from common import *
from chartstyle import *
from matplotlib.patches import Patch

tab = pd.read_csv(ARTIFACTS / "model_comparison_regression.csv")
sel = load_json(ARTIFACTS / "regression_selection.json")
tbl = pd.read_parquet(MODEL_TABLE, columns=["DEP_DELAY", "SPLIT"])
yv, yt = tbl.loc[tbl.SPLIT == "val", "DEP_DELAY"].values.astype(float), tbl.loc[tbl.SPLIT == "test", "DEP_DELAY"].values.astype(float)
safe = lambda n: n.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("+", "plus")
preds = {m: np.load(WORK_DIR / f"reg_pred_{safe(m)}.npz") for m in tab.Model}
best = sel["best_overall"]["model"]
best_direct = sel["best_direct"]["model"]
mean_b, med_b = "Training mean baseline", "Training median baseline"

def col(r):
    if r.Model == best: return GREEN
    if r.Group == "baseline": return AMBER
    if r.Group == "original-replica": return SLATE
    if r.Group == "two-stage": return PURPLE
    return BLUE
LEG = [Patch(color=GREEN, label="Selected (lowest validation MAE)"), Patch(color=BLUE, label="Improved models"), Patch(color=SLATE, label="Original-project replicas"),
       Patch(color=PURPLE, label="Two-stage systems"), Patch(color=AMBER, label="Baselines")]

def hbar(metric_val, metric_test, title, fname, xl, fmt="{:.2f}", lower_better=True):
    d = tab.sort_values(metric_test, ascending=not lower_better)
    fig, ax = plt.subplots(figsize=(12, max(7, .36 * len(d))))
    y = np.arange(len(d))
    ax.barh(y + .2, d[metric_test], .4, color=[col(r) for r in d.itertuples()], label="Test")
    ax.barh(y - .2, d[metric_val], .4, color=[col(r) for r in d.itertuples()], alpha=.45, label="Validation")
    ax.set_yticks(y); ax.set_yticklabels(d.Model, fontsize=8)
    for i, (a, b) in enumerate(zip(d[metric_test], d[metric_val])):
        ax.text(a, i + .2, " " + fmt.format(a), va="center", fontsize=7.5); ax.text(b, i - .2, " " + fmt.format(b), va="center", fontsize=7, alpha=.7)
    ax.invert_yaxis(); ax.set_xlabel(xl); ax.set_title(title + "\n(solid = test period, faded = validation period)")
    ax.legend(handles=LEG, fontsize=8, loc="lower right"); save(fig, fname)

hbar("Validation MAE", "Test MAE", "Mean Absolute Error Comparison (minutes, lower is better)", "reg_01_mae_comparison", "MAE (minutes)")
hbar("Val RMSE", "RMSE", "Root Mean Squared Error Comparison (minutes, lower is better)", "reg_02_rmse_comparison", "RMSE (minutes)")
hbar("Val R2", "R2", "R² Comparison (higher is better; negative = worse than predicting the training mean)", "reg_03_r2_comparison", "R²", fmt="{:.3f}", lower_better=False)

pb = preds[best]["test"].astype(float); res = yt - pb
rng = np.random.default_rng(0); idx = rng.choice(len(yt), 60000, replace=False)
# 4 actual vs predicted
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
h = ax[0].hexbin(yt[idx], pb[idx], gridsize=70, bins="log", cmap="Blues", extent=(-40, 200, -40, 100))
ax[0].plot([-40, 100], [-40, 100], color=RED, ls="--", label="Perfect prediction (45°)"); ax[0].set_xlabel("Actual DEP_DELAY (min)"); ax[0].set_ylabel("Predicted DEP_DELAY (min)")
ax[0].set_title(f"Actual vs Predicted — {best}\n(test period, 60k-flight sample, log density)"); ax[0].legend(); fig.colorbar(h, ax=ax[0], label="log10(count)")
ax[1].scatter(yt[idx], pb[idx], s=2, alpha=.15, color=BLUE); ax[1].set_xlim(-40, 1200); ax[1].set_ylim(-40, 120)
ax[1].plot([-40, 120], [-40, 120], color=RED, ls="--"); ax[1].set_title("Full actual range (extreme delays compress to bottom-right)"); ax[1].set_xlabel("Actual DEP_DELAY (min)"); ax[1].set_ylabel("Predicted (min)")
save(fig, "reg_04_actual_vs_predicted")
# 5 residual distribution
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].hist(np.clip(res, -60, 120), bins=120, color=BLUE, alpha=.85); ax[0].axvline(0, color=RED, ls="--"); ax[0].axvline(np.median(res), color=AMBER, label=f"Median residual = {np.median(res):.1f}")
ax[0].axvline(res.mean(), color=GREEN, label=f"Mean residual = {res.mean():.1f}"); ax[0].legend()
ax[0].set_title(f"Residual Distribution (actual − predicted, clipped to [-60,120])\n{best}"); ax[0].set_xlabel("Residual (min)"); ax[0].set_ylabel("Flights")
ax[1].hist(res[res > 15], bins=np.logspace(np.log10(15), np.log10(max(res.max(), 100)), 60), color=PURPLE, alpha=.85); ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_title("Positive Residual Tail (under-predicted delays, log-log)"); ax[1].set_xlabel("Residual (min)"); ax[1].set_ylabel("Flights")
save(fig, "reg_05_residual_distribution")
# 6 residual vs predicted
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(pb[idx], res[idx], s=2, alpha=.15, color=BLUE); ax.axhline(0, color=RED, ls="--"); ax.set_ylim(-80, 250)
bins = np.quantile(pb, np.linspace(0, 1, 21)); mid = []; med = []
for a, b in zip(bins[:-1], bins[1:]):
    m = (pb >= a) & (pb <= b); mid.append(pb[m].mean()); med.append(np.median(res[m]))
ax.plot(mid, med, color=AMBER, lw=2.5, marker="o", label="Median residual per predicted-quantile")
ax.set_xlabel("Predicted DEP_DELAY (min)"); ax.set_ylabel("Residual (min)"); ax.set_title(f"Residuals vs Predicted — {best} (test period)"); ax.legend()
save(fig, "reg_06_residual_vs_predicted")
# 7 error distribution (absolute error ECDF + hist) vs baselines
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
cands = [(best, GREEN), (mean_b, AMBER), (med_b, SLATE)]
old_name = next((m for m in tab.Model if m == "OLD Transformed GBR (signed-log)"), None)
if old_name: cands.append((old_name, PURPLE))
for m, c in cands:
    e = np.abs(yt - preds[m]["test"].astype(float)); xs = np.sort(e)[::50]; ax[0].plot(xs, np.linspace(0, 1, len(xs)), color=c, lw=2, label=f"{m} (MAE {e.mean():.2f})")
    ax[1].hist(np.clip(e, 0, 90), bins=90, histtype="step", color=c, lw=1.8, density=True, label=m)
ax[0].set_xlim(0, 90); ax[0].set_xlabel("Absolute error (min)"); ax[0].set_ylabel("Cumulative share of flights"); ax[0].set_title("Absolute-Error ECDF (test period)"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("Absolute error (min, clipped at 90)"); ax[1].set_ylabel("Density"); ax[1].set_title("Absolute-Error Distribution"); ax[1].legend(fontsize=8)
save(fig, "reg_07_error_distribution")
# 8 performance across delay ranges
bins_ = [-np.inf, 0, 15, 60, 120, np.inf]; labs = ["Early\n(<0)", "On-time/minor\n(0-15)", "Moderate\n(15-60)", "High\n(60-120)", "Severe\n(>120)"]
cat_ = pd.cut(yt, bins_, labels=labs, right=False)
rows = {}
for m in [best, mean_b, med_b] + ([old_name] if old_name else []) + ([best_direct] if best_direct != best else []):
    e = np.abs(yt - preds[m]["test"].astype(float)); rows[m] = pd.Series(e).groupby(cat_, observed=True).mean()
cnt = pd.Series(yt).groupby(cat_, observed=True).size()
fig, ax = plt.subplots(figsize=(13, 5.8)); x = np.arange(len(labs)); w = .8 / len(rows)
for k, (m, s) in enumerate(rows.items()):
    ax.bar(x + (k - (len(rows) - 1) / 2) * w, s.reindex(labs).values, w, label=m, color=[GREEN, AMBER, SLATE, PURPLE, BLUE][k % 5])
ax.set_xticks(x); ax.set_xticklabels([f"{l}\n n={cnt.reindex(labs).iat[i]:,}" for i, l in enumerate(labs)]); ax.set_ylabel("MAE within actual-delay band (min)")
ax.set_title("Prediction Error by Actual Delay Range (test period)"); ax.legend(fontsize=8); ax.set_yscale("log")
save(fig, "reg_08_performance_by_delay_range")
band_tab = pd.DataFrame(rows).reindex(labs); band_tab.index = [l.replace("\n", " ") for l in labs]; band_tab["n_flights"] = cnt.reindex(labs).values
band_tab.to_csv(ARTIFACTS / "regression_error_by_delay_band.csv")
# 9 feature importance
try:
    rb = joblib.load(MODELS_DIR / "regression" / f"{sel['best_direct']['safe_name']}.joblib"); pipe = rb["pipeline"]
    names = [c.split("__", 1)[1] for c in pipe.named_steps["pre"].get_feature_names_out()]
    est = pipe.named_steps["reg"]
    if hasattr(est, "regressor_"): est = est.regressor_
    if hasattr(est, "booster_"): imp = est.booster_.feature_importance("gain"); kind = "LightGBM total gain"
    elif hasattr(est, "get_booster"):
        sc = est.get_booster().get_score(importance_type="gain"); imp = np.array([sc.get(f"f{i}", 0.0) for i in range(len(names))]); kind = "XGBoost total gain"
    elif hasattr(est, "get_feature_importance"): imp = est.get_feature_importance(); kind = "CatBoost PredictionValuesChange"
    elif hasattr(est, "feature_importances_"): imp = est.feature_importances_; kind = "impurity-based importance"
    else: raise AttributeError("no native importance")
    imp = np.asarray(imp, float); imp = imp / imp.sum() * 100
    fi = pd.DataFrame({"feature": names, "importance_pct": imp}).sort_values("importance_pct", ascending=False); fi.to_csv(ARTIFACTS / "feature_importance_regression.csv", index=False)
    top = fi.head(20).iloc[::-1]; fig, ax = plt.subplots(figsize=(10, 8)); ax.barh(top.feature, top.importance_pct, color=PURPLE)
    for i, v in enumerate(top.importance_pct): ax.text(v + .1, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_title(f"Top-20 Feature Importances — {sel['best_direct']['model']}\n({kind}, normalised to 100%)"); ax.set_xlabel("Relative importance (%)"); save(fig, "reg_09_feature_importance")
except Exception as e:
    print("regression importance skipped:", e)
# 10 baseline vs model
order = [mean_b, med_b] + ([old_name] if old_name else []) + [sel["best_direct"]["model"]] + ([best] if best != sel["best_direct"]["model"] else [])
d = tab.set_index("Model").loc[order]
fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
for a, (c, t) in zip(ax, [("Test MAE", "MAE (min)"), ("RMSE", "RMSE (min)"), ("R2", "R²")]):
    a.bar(range(len(d)), d[c], color=[AMBER, SLATE, PURPLE, GREEN, GREEN][:len(d)])
    a.set_xticks(range(len(d))); a.set_xticklabels([n.replace(" (", "\n(").replace("baseline", "\nbaseline")[:34] for n in d.index], fontsize=7.5, rotation=20, ha="right")
    for i, v in enumerate(d[c]): a.text(i, v, f"{v:.3f}" if c == "R2" else f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    a.set_title(t + " — test period"); a.axhline(0, color="k", lw=.6)
fig.suptitle("Baselines vs Original Model vs Selected Models", fontsize=14, fontweight="bold"); save(fig, "reg_10_baseline_vs_model")
# 11 two-stage
ts = tab[tab.Group == "two-stage"]
if len(ts):
    items = [(med_b, AMBER), (sel["best_direct"]["model"], GREEN)] + [(m, PURPLE) for m in ts.Model]
    dd = tab.set_index("Model").loc[[i[0] for i in items]]
    fig, ax = plt.subplots(figsize=(12, 5.5)); ax.bar(range(len(dd)), dd["Test MAE"], color=[i[1] for i in items])
    ax.bar(range(len(dd)), dd["Validation MAE"], width=.45, color="white", alpha=.35)
    for i, (a, b) in enumerate(zip(dd["Test MAE"], dd["Validation MAE"])): ax.text(i, a, f"{a:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(dd))); ax.set_xticklabels([n.replace(" (", "\n(").replace(", ", ",\n") for n in dd.index], fontsize=7.5)
    ax.set_title("Two-Stage System vs Direct Regression vs Median Baseline (Test MAE; white = validation MAE)"); ax.set_ylabel("MAE (min)"); save(fig, "reg_11_two_stage")
print("regression charts done")
