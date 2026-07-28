"""M5 + M1 pour l'article.

M5 — Baseline « prévision seuillée seule » vs modèle ML :
     montre que le ML apporte un gain par rapport à l'usage brut de la prévision.

M1 — Évaluation de la FUSION (système à deux étages) :
     analyse quantitative (à partir des performances mesurées de chaque composante)
     du gain de précision apporté par le filtre visuel, avec analyse de sensibilité.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_forecast as base
from rf_forecast_v2 import build_dataset, rf_factory
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

FIG = Path(__file__).resolve().parent / "rf_forecast_results/figures"
H = 3


def precision_at_recall(y, score, target=0.90):
    pr, rc, _ = precision_recall_curve(y, score)
    m = rc >= target
    return pr[m].max() if m.any() else float("nan")


# ============================================================= M5
print("=" * 64)
print(" M5 — Baseline 'forecast-threshold only' vs ML")
print("=" * 64)
rows = build_dataset(H)[0]
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
fr = df[df.date >= pd.Timestamp("2024-06-01")].dropna(subset=base.SELECTED_FEATURES + FC + ["flood_risk"]).copy()
fr["flood_risk"] = fr["flood_risk"].astype(int)
y = fr["flood_risk"].to_numpy(int)

# (a) raw forecast score (no ML) : 3-day forecast rainfall sum used directly
raw = fr["fc_sum_next3"].to_numpy(float)
auc_raw = roc_auc_score(y, raw); ap_raw = average_precision_score(y, raw)
p90_raw = precision_at_recall(y, raw, 0.90)
# (b) ML model on reanalysis + forecast (out-of-fold)
X = fr[base.SELECTED_FEATURES + FC].to_numpy(float)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
proba = cross_val_predict(rf_factory(10, 10)(), X, y, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
auc_ml = roc_auc_score(y, proba); ap_ml = average_precision_score(y, proba)
p90_ml = precision_at_recall(y, proba, 0.90)
print(f"  {'Raw forecast (threshold only)':32s} AUC={auc_raw:.3f}  PR-AUC={ap_raw:.3f}  P@R0.90={p90_raw:.3f}")
print(f"  {'ML (reanalysis + forecast)':32s} AUC={auc_ml:.3f}  PR-AUC={ap_ml:.3f}  P@R0.90={p90_ml:.3f}")
print(f"  -> ML gain: AUC {auc_ml-auc_raw:+.3f}, PR-AUC {ap_ml-ap_raw:+.3f}, P@R0.90 {p90_ml-p90_raw:+.3f}")

# ============================================================= M1
print("\n" + "=" * 64)
print(" M1 — Two-stage FUSION evaluation (measured component rates)")
print("=" * 64)
# measured operating points
r_f, f_f, p_f = 0.930, 0.527, 0.298   # forecast: recall, FPR, precision (18-zone test)
r_c = 0.955                            # CNN flood recall (ConvNeXt, latest run)
pi = 0.194                             # base rate of flood-risk events (test)


def system(pi, r_f, f_f, r_c, f_c, s=1.0):
    tp = pi * r_f * s * r_c
    fp = (1 - pi) * f_f * s * f_c
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = r_f * s * r_c            # P(alert | true flood)
    return precision, recall


print(f"  Forecast stage alone : precision={p_f:.3f}, recall={r_f:.3f}")
print(f"  Fusion (CNN FPR=0, all flagged events imaged): "
      f"precision={system(pi,r_f,f_f,r_c,0.0)[0]:.3f}, recall={system(pi,r_f,f_f,r_c,0.0)[1]:.3f}")
print("\n  Sensitivity to CNN false-positive rate (submission rate s=1):")
print(f"  {'CNN FPR':>8s} {'System precision':>18s} {'System recall':>14s}")
fprs = [0.0, 0.02, 0.05, 0.10]
sys_prec = []
for fc in fprs:
    pcz, rcz = system(pi, r_f, f_f, r_c, fc)
    sys_prec.append(pcz)
    print(f"  {fc:8.2f} {pcz:18.3f} {rcz:14.3f}")

print("\n  Sensitivity to citizen image-submission rate (CNN FPR=0.02):")
print(f"  {'submit s':>8s} {'System precision':>18s} {'System recall':>14s}")
for s in [1.0, 0.7, 0.5, 0.3]:
    pcz, rcz = system(pi, r_f, f_f, r_c, 0.02, s)
    print(f"  {s:8.2f} {pcz:18.3f} {rcz:14.3f}")

# ---- figure : precision vs CNN FPR ----
xx = np.linspace(0, 0.12, 60)
yy = [system(pi, r_f, f_f, r_c, fc)[0] for fc in xx]
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.plot(xx * 100, yy, lw=2.5, color="#2ecc71", label="Two-stage system (fusion)")
ax.axhline(p_f, ls="--", color="#C44E52", lw=2, label=f"Forecasting stage alone ({p_f:.2f})")
ax.scatter([0], [system(pi, r_f, f_f, r_c, 0.0)[0]], color="#2ecc71", zorder=5)
ax.annotate("CNN measured\n(FPR=0)", (0.2, 0.96), fontsize=9)
ax.set(xlabel="Visual-stage false-positive rate (%)", ylabel="System alert precision",
       title="Fusion lifts alert precision far above the forecasting stage", ylim=(0, 1.05))
ax.legend(loc="lower left"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "en_fig_fusion.png", dpi=150, bbox_inches="tight")
print("\n✓ figure: en_fig_fusion.png")
