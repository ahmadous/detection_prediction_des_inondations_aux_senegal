"""Apport de la VRAIE prévision météo (Previous Runs) — démonstration de principe.

Compare 3 jeux de variables pour prédire l'inondation réelle à t+1…t+3 :
  1. RÉANALYSE seule        (précurseurs connus à t — notre modèle actuel)
  2. PRÉVISION seule        (pluie prévue pour t+1…t+3, émise à t)
  3. RÉANALYSE + PRÉVISION  (les deux)

Données : saisons des pluies 2024 + 2025 (période où la prévision archivée existe).
Petit échantillon -> démonstration de principe, validée par CV stratifiée + holdout.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_forecast as base
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

H = 3
PR_DIR = Path(__file__).resolve().parent / "donnees_previous_runs"

# 1) réanalyse : label + features
daily = base.add_features(base.aggregate_daily(base.load_hourly(base.DATA_DIR)))
TH = base.calibrate_thresholds(daily[daily.date < base.SPLIT_DATE], H)
daily = base.add_future_target(daily, H, TH)

# 2) prévision : features de la fenêtre future, émises à t
pr = pd.concat([pd.read_csv(c) for c in sorted(PR_DIR.glob("*_prevruns.csv"))], ignore_index=True)
pr["date"] = pd.to_datetime(pr["date"])
out = []
for loc, g in pr.groupby("location", group_keys=False):
    g = g.sort_values("date").copy()
    g["fc_next1"] = g["fc_lead1"].shift(-1)                 # prévu pour t+1, émis à t
    g["fc_next2"] = g["fc_lead2"].shift(-2)                 # prévu pour t+2, émis à t
    g["fc_next3"] = g["fc_lead3"].shift(-3)                 # prévu pour t+3, émis à t
    g["fc_sum_next3"] = g[["fc_next1", "fc_next2", "fc_next3"]].sum(axis=1)
    g["fc_max_next3"] = g[["fc_next1", "fc_next2", "fc_next3"]].max(axis=1)
    out.append(g)
pr = pd.concat(out, ignore_index=True)
FC = ["fc_next1", "fc_sum_next3", "fc_max_next3"]

# 3) fusion + saison des pluies + période à prévision authentique (>= 2024-06)
df = daily.merge(pr[["location", "date"] + FC], on=["location", "date"], how="inner")
df["actual_next1"] = df.sort_values(["location", "date"]).groupby("location")["precipitation_mm_sum"].shift(-1)
rows = df[(df.month.isin(base.RAINY_MONTHS)) & (df.date >= pd.Timestamp("2024-06-01"))].copy()
rows = rows.dropna(subset=base.SELECTED_FEATURES + FC + ["flood_risk"])
rows["flood_risk"] = rows["flood_risk"].astype(int)

# 4) contrôle skill : la prévision doit différer du réel (corr < 1)
skill = rows["fc_next1"].corr(rows["actual_next1"])
print(f"Contrôle skill prévision J+1 : corr(prévu, réel) = {skill:.3f}  "
      f"(<1 = vraie prévision avec erreur ✓)")
print(f"Échantillon : {len(rows)} jours-zones, {rows.flood_risk.mean()*100:.1f}% à risque "
      f"(2024: {int((rows.date.dt.year==2024).sum())}, 2025: {int((rows.date.dt.year==2025).sum())})")

FEATURE_SETS = {
    "1. Réanalyse seule": base.SELECTED_FEATURES,
    "2. Prévision seule": FC,
    "3. Réanalyse + Prévision": base.SELECTED_FEATURES + FC,
}
def rf(): return make_pipeline(StandardScaler(), RandomForestClassifier(
    n_estimators=400, max_depth=10, min_samples_leaf=10, max_features="sqrt",
    class_weight="balanced_subsample", random_state=42, n_jobs=-1))

def f1max(y, p):
    pr_, rc_, _ = precision_recall_curve(y, p)
    f1 = 2 * pr_[:-1] * rc_[:-1] / (pr_[:-1] + rc_[:-1] + 1e-9)
    i = f1.argmax()
    return pr_[i], rc_[i], f1[i]

# --- A) validation croisée stratifiée 5 folds (robustesse sur petit échantillon) ---
print("\n--- Validation croisée stratifiée (5 folds) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, feats in FEATURE_SETS.items():
    X = rows[feats].to_numpy(float); y = rows["flood_risk"].to_numpy(int)
    aucs, aps = [], []
    for tr, te in skf.split(X, y):
        m = rf().fit(X[tr], y[tr]); p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p)); aps.append(average_precision_score(y[te], p))
    print(f"  {name:28s} AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}   "
          f"PR-AUC = {np.mean(aps):.3f}")

# --- B) holdout temporel : train 2024 -> test 2025 ---
tr = rows[rows.date.dt.year == 2024]; te = rows[rows.date.dt.year == 2025]
print(f"\n--- Holdout temporel : train 2024 ({len(tr)}) -> test 2025 ({len(te)}, "
      f"{te.flood_risk.mean()*100:.0f}% à risque) ---")
for name, feats in FEATURE_SETS.items():
    m = rf().fit(tr[feats].to_numpy(float), tr["flood_risk"].to_numpy(int))
    p = m.predict_proba(te[feats].to_numpy(float))[:, 1]
    y = te["flood_risk"].to_numpy(int)
    pcz, rcz, fz = f1max(y, p)
    print(f"  {name:28s} AUC = {roc_auc_score(y,p):.3f}   PR-AUC = {average_precision_score(y,p):.3f}   "
          f"| point F1-max : P={pcz:.2f} R={rcz:.2f} F1={fz:.2f}")
