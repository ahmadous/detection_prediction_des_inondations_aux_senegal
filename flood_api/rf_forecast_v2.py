"""
Chantier 1 — Renforcement du modèle de prévision d'inondation (J+1…J+h).

Améliore rf_forecast.py sur 4 axes :
  (a) Ajout d'un Gradient Boosting (HistGradientBoosting = équivalent XGBoost,
      intégré à scikit-learn, aucune dépendance externe).
  (b) Calibration des probabilités (CalibratedClassifierCV, isotonic) pour un
      meilleur Brier score et un seuil qui transfère mieux.
  (c) Réglage des hyperparamètres RF/HGB par recherche sur validation temporelle.
  (d) Split train / val(2021-2022) / test(>=2023) : le seuil opérationnel est
      choisi sur la tranche RÉCENTE (val), plus proche des conditions de test,
      pour absorber la dérive climatique (test 2023-2025 plus humide).

Réutilise le chargement + feature engineering (sans fuite) de rf_forecast.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_forecast as base  # noqa: E402  (chargement + features + métriques)

VAL_START = pd.Timestamp("2021-01-01")
SPLIT_DATE = base.SPLIT_DATE  # 2023-01-01
FEATURES = base.SELECTED_FEATURES
RS = base.RANDOM_STATE


def build_dataset(horizon: int):
    hourly = base.load_hourly(base.DATA_DIR)
    daily = base.add_features(base.aggregate_daily(hourly))
    thresholds = base.calibrate_thresholds(daily[daily["date"] < SPLIT_DATE], horizon)
    daily = base.add_future_target(daily, horizon, thresholds)
    rows = daily[daily["month"].isin(base.RAINY_MONTHS)].dropna(subset=["flood_risk"]).copy()
    rows["flood_risk"] = rows["flood_risk"].astype(int)
    base.assert_no_leakage(FEATURES)
    return rows, thresholds


def matrix(df):
    X = df[FEATURES].ffill().bfill().fillna(0).to_numpy(float)
    return X, df["flood_risk"].to_numpy(int)


def cv_auc(estimator_fn, X, y, n_splits=4):
    """AUC moyen en validation croisée temporelle (estimator_fn() -> pipeline neuf)."""
    scores = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
        est = estimator_fn()
        est.fit(X[tr], y[tr])
        p = est.predict_proba(X[va])[:, 1]
        scores.append(roc_auc_score(y[va], p))
    return np.array(scores)


# --------------------------------------------------------------------------- #
# Grilles d'hyperparamètres (compactes, réglées par CV temporelle)
# --------------------------------------------------------------------------- #
def rf_factory(md, msl):
    return lambda: make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=400, max_depth=md, min_samples_leaf=msl,
                               max_features="sqrt", class_weight="balanced_subsample",
                               random_state=RS, n_jobs=-1),
    )


def hgb_factory(md, lr):
    return lambda: make_pipeline(
        StandardScaler(),
        HistGradientBoostingClassifier(max_depth=md, learning_rate=lr, max_iter=400,
                                       l2_regularization=1.0, early_stopping=True,
                                       validation_fraction=0.15, class_weight="balanced",
                                       random_state=RS),
    )


def search(name, factories, X, y):
    print(f"\n[{name}] recherche d'hyperparamètres (CV temporelle 4 folds)…")
    best, best_auc, best_desc = None, -1, None
    for desc, fac in factories:
        s = cv_auc(fac, X, y)
        print(f"  {desc:32s} AUC = {s.mean():.3f} ± {s.std():.3f}")
        if s.mean() > best_auc:
            best, best_auc, best_desc = fac, s.mean(), desc
    print(f"  -> meilleur : {best_desc} (AUC={best_auc:.3f})")
    return best, best_auc, best_desc


def main(horizon: int):
    print(f"\n{'='*70}\n RENFORCEMENT — Prévision inondation J+1…J+{horizon}\n{'='*70}")
    rows, thresholds = build_dataset(horizon)

    train = rows[rows["date"] < VAL_START]                                   # < 2021
    val = rows[(rows["date"] >= VAL_START) & (rows["date"] < SPLIT_DATE)]     # 2021-2022
    test = rows[rows["date"] >= SPLIT_DATE]                                   # >= 2023
    trainval = rows[rows["date"] < SPLIT_DATE]

    Xtr, ytr = matrix(train)
    Xval, yval = matrix(val)
    Xte, yte = matrix(test)
    Xtv, ytv = matrix(trainval)
    print(f"\ntrain<2021={len(ytr)} ({ytr.mean()*100:.1f}%) | "
          f"val 21-22={len(yval)} ({yval.mean()*100:.1f}%) | "
          f"test>=2023={len(yte)} ({yte.mean()*100:.1f}%)")

    # --- réglage sur train (CV temporelle) ---
    rf_grid = [(f"RF depth={md} leaf={msl}", rf_factory(md, msl))
               for md in (6, 8, 10) for msl in (10, 30)]
    hgb_grid = [(f"HGB depth={md} lr={lr}", hgb_factory(md, lr))
                for md in (3, 5) for lr in (0.05, 0.1)]

    rf_fac, rf_cv, rf_desc = search("RandomForest", rf_grid, Xtr, ytr)
    hgb_fac, hgb_cv, hgb_desc = search("HistGradientBoosting", hgb_grid, Xtr, ytr)

    def logit_fac():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, class_weight="balanced"))
    lg_cv = cv_auc(logit_fac, Xtr, ytr)
    print(f"\n[LogReg] CV AUC = {lg_cv.mean():.3f} ± {lg_cv.std():.3f}")

    # --- modèles finaux : entraînés sur train, CALIBRÉS sur val (isotonic) ---
    def fit_calibrated(fac):
        base_est = fac().fit(Xtr, ytr)
        cal = CalibratedClassifierCV(base_est, method="isotonic", cv="prefit")
        cal.fit(Xval, yval)  # calibration sur tranche récente
        return base_est, cal

    models = {}
    for key, fac in (("RandomForest", rf_fac), ("HistGradientBoosting", hgb_fac),
                     ("LogReg", logit_fac)):
        raw, cal = fit_calibrated(fac)
        models[key] = (raw, cal)

    # --- seuil choisi sur VAL (conditions récentes) pour rappel≈0,90 ---
    print(f"\n--- TEST 2023–2025 (modèles calibrés, seuil réglé sur val 2021-2022) ---")
    report = {}
    for key, (raw, cal) in models.items():
        proba_val = cal.predict_proba(Xval)[:, 1]
        proba_te = cal.predict_proba(Xte)[:, 1]
        thr = base.pick_threshold_for_recall(yval, proba_val, target_recall=0.90)
        m = base.metrics_at_threshold(yte, proba_te, thr)
        report[key] = m
        print(f"  {key:22s} AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} "
              f"Brier={m['brier']:.3f} | seuil={thr:.2f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"[TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}]")

    # baseline persistance (référence basse)
    p95 = {loc: b for loc, (a, b) in thresholds.items()}
    persist = (test["precip_3d_sum"].to_numpy() >= test["location"].map(p95).to_numpy()).astype(float)
    pm = base.metrics_at_threshold(yte, persist, 0.5)
    report["Persistence"] = pm
    print(f"  {'Persistance (baseline)':22s} AUC={pm['roc_auc']:.3f} PR-AUC={pm['pr_auc']:.3f} "
          f"| R={pm['recall']:.3f} F1={pm['f1']:.3f}")

    summary = dict(
        horizon=horizon,
        cv_auc={"RandomForest": round(rf_cv, 4), "HistGradientBoosting": round(hgb_cv, 4),
                "LogReg": round(float(lg_cv.mean()), 4)},
        best_hyperparams={"RandomForest": rf_desc, "HistGradientBoosting": hgb_desc},
        test=report,
        n=dict(train=len(ytr), val=len(yval), test=len(yte)),
        rates=dict(train=float(ytr.mean()), val=float(yval.mean()), test=float(yte.mean())),
    )
    out = base.OUT_DIR / f"results_v2_h{horizon}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n✓ Résultats sauvegardés : {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=3)
    main(ap.parse_args().horizon)
