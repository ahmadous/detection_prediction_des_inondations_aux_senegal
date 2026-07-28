"""Démonstration : augmenter la précision SANS faire chuter le rappel.
On fixe le rappel (0,85 / 0,90 / 0,95) et on lit la précision atteignable
pour deux jeux de variables : réanalyse seule vs réanalyse + prévision.
Probabilités hors-échantillon (5-fold stratifié) -> lecture sur la courbe PR."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_forecast as base
from rf_forecast_v2 import build_dataset, rf_factory
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

H = 3
PR_DIR = Path(__file__).resolve().parent / "donnees_previous_runs"

# --- dataset réanalyse (label + features) + prévision (features futures, sans fuite) ---
rows = build_dataset(H)[0]
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
print(f"Échantillon : {len(fr)} jours-zones, taux de positifs = {y.mean()*100:.1f}%\n")

SETS = {"Réanalyse seule": base.SELECTED_FEATURES,
        "Réanalyse + Prévision": base.SELECTED_FEATURES + FC}
skf = StratifiedKFold(5, shuffle=True, random_state=42)
TARGETS = [0.95, 0.90, 0.85]


def precision_at_recall(y, proba, target):
    prec, rec, _ = precision_recall_curve(y, proba)
    mask = rec >= target
    return prec[mask].max() if mask.any() else float("nan")


print(f"{'Jeu de variables':26s} {'AUC':>6s} {'PR-AUC':>7s}   " +
      "  ".join(f"P@R={t:.2f}" for t in TARGETS))
print("-" * 70)
res = {}
for name, feats in SETS.items():
    X = fr[feats].to_numpy(float)
    proba = cross_val_predict(rf_factory(10, 10)(), X, y, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
    auc = roc_auc_score(y, proba); ap = average_precision_score(y, proba)
    ps = [precision_at_recall(y, proba, t) for t in TARGETS]
    res[name] = ps
    print(f"{name:26s} {auc:6.3f} {ap:7.3f}   " + "   ".join(f"{p:6.3f}" for p in ps))

print("\nGain de précision à rappel FIXE (réanalyse+prévision − réanalyse) :")
for i, t in enumerate(TARGETS):
    d = res["Réanalyse + Prévision"][i] - res["Réanalyse seule"][i]
    print(f"  à rappel = {t:.2f} :  précision {res['Réanalyse seule'][i]:.3f} -> "
          f"{res['Réanalyse + Prévision'][i]:.3f}   ({d:+.3f})")
