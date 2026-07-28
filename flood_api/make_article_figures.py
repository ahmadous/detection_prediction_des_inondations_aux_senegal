"""Régénère les figures de la composante prévision AVEC DES LABELS EN ANGLAIS,
pour l'article IEEE (le notebook reste en français pour l'usage local).

Produit : en_fig_roc_pr.png, en_fig_confusion_calibration.png,
          en_fig_importances.png, en_fig_forecast_gain.png
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_forecast as base
from rf_forecast_v2 import build_dataset, matrix, rf_factory, hgb_factory, VAL_START, SPLIT_DATE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix,
                             ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 130, "axes.titleweight": "semibold"})
FIG = Path(__file__).resolve().parent / "rf_forecast_results/figures"
H = 3

rows, TH = build_dataset(H)
train = rows[rows.date < VAL_START]
val = rows[(rows.date >= VAL_START) & (rows.date < SPLIT_DATE)]
test = rows[rows.date >= SPLIT_DATE]
Xtr, ytr = matrix(train); Xval, yval = matrix(val); Xte, yte = matrix(test)


def lg_fac():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))


FACT = {"Random Forest": rf_factory(10, 10), "HistGradientBoosting": hgb_factory(5, 0.05),
        "Logistic regression": lg_fac}


def fit_cal(fac):
    est = fac().fit(Xtr, ytr)
    return CalibratedClassifierCV(FrozenEstimator(est), method="isotonic").fit(Xval, yval)


print("Training calibrated models...")
MODELS = {k: fit_cal(f) for k, f in FACT.items()}
PROBA = {k: m.predict_proba(Xte)[:, 1] for k, m in MODELS.items()}
THR = {k: base.pick_threshold_for_recall(yval, m.predict_proba(Xval)[:, 1]) for k, m in MODELS.items()}
p95 = {l: b for l, (a, b) in TH.items()}
PROBA["Persistence"] = (test["precip_3d_sum"].to_numpy() >= test["location"].map(p95).to_numpy()).astype(float)

# ---------------- Fig. ROC + PR (English) ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for k in ["Random Forest", "HistGradientBoosting", "Logistic regression", "Persistence"]:
    fpr, tpr, _ = roc_curve(yte, PROBA[k])
    ax[0].plot(fpr, tpr, lw=2, label=f"{k} (AUC={roc_auc_score(yte, PROBA[k]):.3f})")
    pr, rc, _ = precision_recall_curve(yte, PROBA[k])
    ax[1].plot(rc, pr, lw=2, label=k)
ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
ax[0].set(title="ROC curves (test 2023-2025)", xlabel="False positive rate", ylabel="True positive rate")
ax[0].legend(fontsize=8)
ax[1].axhline(yte.mean(), ls="--", color="grey", lw=1, label=f"Chance ({yte.mean():.2f})")
ax[1].set(title="Precision-Recall curves", xlabel="Recall", ylabel="Precision")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "en_fig_roc_pr.png", bbox_inches="tight"); plt.close()
print("  + en_fig_roc_pr.png")

# ---------------- Fig. Confusion + Calibration (English) ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
pred = (PROBA["Random Forest"] >= THR["Random Forest"]).astype(int)
ConfusionMatrixDisplay(confusion_matrix(yte, pred),
                       display_labels=["No risk", "Flood risk"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set(title=f"Confusion matrix — Random Forest (threshold={THR['Random Forest']:.2f})",
          xlabel="Predicted label", ylabel="True label")
ax[0].grid(False)
for k in ["Random Forest", "HistGradientBoosting", "Logistic regression"]:
    fr, mp = calibration_curve(yte, PROBA[k], n_bins=10, strategy="quantile")
    ax[1].plot(mp, fr, "o-", label=k)
ax[1].plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
ax[1].set(title="Reliability curve", xlabel="Predicted probability", ylabel="Observed frequency")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "en_fig_confusion_calibration.png", bbox_inches="tight"); plt.close()
print("  + en_fig_confusion_calibration.png")

# ---------------- Fig. Importances (English) ----------------
rfm = rf_factory(10, 10)().fit(Xtr, ytr)
F = base.SELECTED_FEATURES
gini = pd.Series(rfm.steps[-1][1].feature_importances_, index=F).sort_values()
perm = permutation_importance(rfm, Xte, yte, n_repeats=8, random_state=42, scoring="roc_auc", n_jobs=-1)
perm_s = pd.Series(perm.importances_mean, index=F).sort_values()
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
gini.tail(15).plot.barh(ax=ax[0], color="#4C72B0")
ax[0].set(title="Gini importance (Random Forest) — top 15", xlabel="Mean decrease in impurity")
perm_s.tail(15).plot.barh(ax=ax[1], color="#C44E52")
ax[1].set(title="Permutation importance (test AUC) — top 15", xlabel="Mean AUC decrease")
plt.tight_layout(); plt.savefig(FIG / "en_fig_importances.png", bbox_inches="tight"); plt.close()
print("  + en_fig_importances.png")

# ---------------- Fig. Forecast contribution (English) ----------------
PR_DIR = Path(__file__).resolve().parent / "donnees_previous_runs"
pr = pd.concat([pd.read_csv(c) for c in sorted(PR_DIR.glob("*_prevruns.csv"))], ignore_index=True)
pr["date"] = pd.to_datetime(pr["date"])
out = []
for loc, g in pr.groupby("location", group_keys=False):
    g = g.sort_values("date").copy()
    g["fc_next1"] = g["fc_lead1"].shift(-1)
    g["fc_next2"] = g["fc_lead2"].shift(-2)
    g["fc_next3"] = g["fc_lead3"].shift(-3)
    g["fc_sum_next3"] = g[["fc_next1", "fc_next2", "fc_next3"]].sum(axis=1)
    g["fc_max_next3"] = g[["fc_next1", "fc_next2", "fc_next3"]].max(axis=1)
    out.append(g)
pr = pd.concat(out, ignore_index=True)
FC = ["fc_next1", "fc_sum_next3", "fc_max_next3"]
df = rows.merge(pr[["location", "date"] + FC], on=["location", "date"], how="inner")
fr_ = df[df.date >= pd.Timestamp("2024-06-01")].dropna(subset=F + FC + ["flood_risk"]).copy()
fr_["flood_risk"] = fr_["flood_risk"].astype(int)
SETS = {"Reanalysis only": F, "Forecast only": FC, "Reanalysis + Forecast": F + FC}
skf = StratifiedKFold(5, shuffle=True, random_state=42)
comp = {}
for name, feats in SETS.items():
    X = fr_[feats].to_numpy(float); y = fr_["flood_risk"].to_numpy(int)
    a, p = [], []
    for tr, te in skf.split(X, y):
        m = rf_factory(10, 10)().fit(X[tr], y[tr]); pb = m.predict_proba(X[te])[:, 1]
        a.append(roc_auc_score(y[te], pb)); p.append(average_precision_score(y[te], pb))
    comp[name] = (np.mean(a), np.mean(p))
cdf = pd.DataFrame(comp, index=["ROC-AUC", "PR-AUC"]).T
fig, ax = plt.subplots(figsize=(8, 4.4))
cdf.plot.bar(ax=ax, color=["#4C72B0", "#C44E52"], rot=8)
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", fontsize=9)
ax.set(title="Contribution of the numerical weather forecast (5-fold CV, 2024-2025)",
       ylabel="Score", ylim=(0, 1))
ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(FIG / "en_fig_forecast_gain.png", bbox_inches="tight"); plt.close()
print("  + en_fig_forecast_gain.png")
print("\nDone. English figures written to", FIG)
