"""Expérience : apport des features de PRÉVISION météo (Open-Meteo Forecast).

Compare, sur la même période (train 2022-2023, test 2024-2025) :
  - BASE          : features de réanalyse (précurseurs connus à t)
  - BASE+PRÉVISION: idem + prévision de pluie pour la fenêtre future t+1…t+3

La prévision, émise à t, est disponible à t -> légitime, pas de fuite.
Le label reste l'inondation RÉELLE (réanalyse) sur t+1…t+3.
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
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, precision_score, recall_score, f1_score)

H = 3
FC_DIR = Path(__file__).resolve().parent / "donnees_forecast"

# ---------- 1) réanalyse : features + label (comme le notebook) ----------
daily = base.add_features(base.aggregate_daily(base.load_hourly(base.DATA_DIR)))
TH = base.calibrate_thresholds(daily[daily.date < base.SPLIT_DATE], H)
daily = base.add_future_target(daily, H, TH)

# ---------- 2) prévisions : features connues à t pour la fenêtre t+1…t+3 ----------
def load_forecast():
    frames = []
    for csv in sorted(FC_DIR.glob("*_forecast.csv")):
        loc = csv.stem.replace("_forecast", "")
        f = pd.read_csv(csv)
        f["date"] = pd.to_datetime(f["time"]).dt.floor("D")
        f["location"] = loc
        f = f.sort_values("date")
        p = pd.to_numeric(f["precipitation_sum"], errors="coerce")
        # prévision pour la fenêtre FUTURE (émise à t)
        f["fc_precip_next1"] = p.shift(-1)
        f["fc_precip_sum_next3"] = sum(p.shift(-k) for k in range(1, H + 1))
        f["fc_precip_max_next3"] = pd.concat([p.shift(-k) for k in range(1, H + 1)], axis=1).max(axis=1)
        ph = pd.to_numeric(f["precipitation_hours"], errors="coerce")
        f["fc_precip_hours_next3"] = sum(ph.shift(-k) for k in range(1, H + 1))
        pp = pd.to_numeric(f.get("precipitation_probability_max"), errors="coerce")
        f["fc_precip_prob_max_next3"] = pd.concat([pp.shift(-k) for k in range(1, H + 1)], axis=1).max(axis=1)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)

fc = load_forecast()
FC_FEATURES = ["fc_precip_next1", "fc_precip_sum_next3", "fc_precip_max_next3",
               "fc_precip_hours_next3", "fc_precip_prob_max_next3"]

# ---------- 3) contrôle : skill réel de la prévision (corr fcst vs actuel) ----------
actual_next1 = (daily.sort_values(["location", "date"])
                .groupby("location")["precipitation_mm_sum"].shift(-1))
chk = daily[["location", "date"]].copy()
chk["actual_next1"] = actual_next1.values
chk = chk.merge(fc[["location", "date", "fc_precip_next1"]], on=["location", "date"], how="inner").dropna()
corr = chk["actual_next1"].corr(chk["fc_precip_next1"])
print(f"Contrôle skill prévision J+1 : corrélation(prévu, réel) = {corr:.3f}  "
      f"(≈0,6-0,8 = vraie prévision ; ≈1,0 = réanalyse déguisée)")

# ---------- 4) fusion + fenêtre 2022+ (couverture des prévisions) ----------
df = daily.merge(fc[["location", "date"] + FC_FEATURES], on=["location", "date"], how="inner")
rows = df[df.month.isin(base.RAINY_MONTHS)].dropna(subset=["flood_risk"]).copy()
rows["flood_risk"] = rows["flood_risk"].astype(int)
rows = rows.dropna(subset=base.SELECTED_FEATURES + FC_FEATURES)

train = rows[rows.date < pd.Timestamp("2024-01-01")]
test = rows[rows.date >= pd.Timestamp("2024-01-01")]
print(f"\ntrain 2022-2023 = {len(train)} ({train.flood_risk.mean()*100:.1f}% à risque) | "
      f"test 2024-2025 = {len(test)} ({test.flood_risk.mean()*100:.1f}% à risque)")


def evaluate(feature_set, name):
    Xtr = train[feature_set].to_numpy(float); ytr = train["flood_risk"].to_numpy(int)
    Xte = test[feature_set].to_numpy(float); yte = test["flood_risk"].to_numpy(int)
    m = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=10, max_features="sqrt",
        class_weight="balanced_subsample", random_state=42, n_jobs=-1)).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p); ap = average_precision_score(yte, p)
    pr, rc, th = precision_recall_curve(yte, p)
    f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-9)
    i = f1.argmax()
    print(f"\n### {name}")
    print(f"  ROC-AUC = {auc:.3f}   PR-AUC = {ap:.3f}")
    print(f"  Point F1-max : Précision = {pr[i]:.2f}  Rappel = {rc[i]:.2f}  F1 = {f1[i]:.2f}")
    return dict(auc=auc, ap=ap, precision=float(pr[i]), recall=float(rc[i]), f1=float(f1[i]))


r_base = evaluate(base.SELECTED_FEATURES, "BASE — réanalyse seule")
r_full = evaluate(base.SELECTED_FEATURES + FC_FEATURES, "BASE + PRÉVISION météo")

print("\n" + "=" * 60)
print(f"GAIN apporté par la prévision :")
print(f"  ROC-AUC   : {r_base['auc']:.3f} -> {r_full['auc']:.3f}  ({r_full['auc']-r_base['auc']:+.3f})")
print(f"  PR-AUC    : {r_base['ap']:.3f} -> {r_full['ap']:.3f}  ({r_full['ap']-r_base['ap']:+.3f})")
print(f"  Précision : {r_base['precision']:.2f} -> {r_full['precision']:.2f}  (au point F1-max)")
print(f"  Rappel    : {r_base['recall']:.2f} -> {r_full['recall']:.2f}")
