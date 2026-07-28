"""Montre le compromis précision/rappel : la précision basse vient du POINT
DE FONCTIONNEMENT (seuil haut-rappel), pas d'une incapacité du modèle.
Compare 3 points de fonctionnement, seuils choisis sur val (2021-22)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_forecast as base
from rf_forecast_v2 import build_dataset, matrix, rf_factory, hgb_factory, VAL_START, SPLIT_DATE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

H = 3
rows, TH = build_dataset(H)
train = rows[rows.date < VAL_START]; val = rows[(rows.date >= VAL_START) & (rows.date < SPLIT_DATE)]
test = rows[rows.date >= SPLIT_DATE]
Xtr, ytr = matrix(train); Xval, yval = matrix(val); Xte, yte = matrix(test)


def fit(fac):
    est = fac().fit(Xtr, ytr)
    return CalibratedClassifierCV(FrozenEstimator(est), method="isotonic").fit(Xval, yval)


def thr_recall(y, p, target=0.90):
    o = np.argsort(-p); rec = np.cumsum(y[o]) / y.sum()
    ok = np.where(rec >= target)[0]
    return float(p[o][ok[0]]) if len(ok) else 0.5


def thr_f1(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-9)
    return float(th[f1.argmax()])


def thr_precision(y, p, target=0.55):
    pr, rc, th = precision_recall_curve(y, p)
    ok = np.where(pr[:-1] >= target)[0]
    return float(th[ok[0]]) if len(ok) else 0.99


def line(name, y, p, thr):
    pred = (p >= thr).astype(int)
    return (f"  {name:34s} seuil={thr:.2f}  "
            f"Précision={precision_score(y,pred,zero_division=0):.2f}  "
            f"Rappel={recall_score(y,pred,zero_division=0):.2f}  "
            f"F1={f1_score(y,pred,zero_division=0):.2f}  "
            f"alertes/j-réel={pred.sum()}/{int(y.sum())}")


for mname, fac in [("HistGradientBoosting", hgb_factory(5, 0.1)),
                   ("Random Forest", rf_factory(10, 10))]:
    m = fit(fac)
    pv = m.predict_proba(Xval)[:, 1]; pt = m.predict_proba(Xte)[:, 1]
    print(f"\n### {mname} — test 2023-2025 (base rate = {yte.mean():.1%} de jours à risque)")
    print(line("Point A — priorité rappel (0,90)", yte, pt, thr_recall(yval, pv, 0.90)))
    print(line("Point B — F1 maximal (équilibré)", yte, pt, thr_f1(yval, pv)))
    print(line("Point C — priorité précision (≥0,55)", yte, pt, thr_precision(yval, pv, 0.55)))
