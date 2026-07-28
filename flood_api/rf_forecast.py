"""
Prévision d'inondation à horizon J+1 … J+h par Random Forest.

REFONTE de model.ipynb pour corriger la CIRCULARITÉ (target leakage) :

    Ancien pipeline (circulaire) :
        LABEL   = f(precipitation_mm_sum_t, precip_3d_sum_t)
        FEATURES contiennent precipitation_mm_sum_t, precip_3d_sum_t
        => AUC = 1,000 tautologique (le modèle recopie sa propre règle)

    Nouveau pipeline (vraie prévision) :
        LABEL   = forte pluie DANS LE FUTUR (t+1 … t+h)
        FEATURES = uniquement l'état connu au jour t (pluie antécédente,
                   humidité du sol, pression + tendance, VPD, saison…)
        => aucune donnée de la fenêtre-cible (t+1…t+h) n'entre dans X.

On respecte la sémantique hydrologique du mémoire (double seuil P97/P95
calibré par zone sur le TRAIN uniquement), mais décalée vers le futur.

Usage :
    python rf_forecast.py --horizon 3
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).resolve().parent / "donnees"
EXTRA_DIR = Path(__file__).resolve().parent / "donnees_extra"  # zones ajoutées via API
OUT_DIR = Path(__file__).resolve().parent / "rf_forecast_results"
OUT_DIR.mkdir(exist_ok=True)

RAINY_MONTHS = {6, 7, 8, 9, 10, 11}  # saison des pluies juin -> novembre
SPLIT_DATE = pd.Timestamp("2023-01-01")
RANDOM_STATE = 42

SUBSTITUTIONS = {" ": "_", "/": "_per_", "%": "pct", "°": "deg",
                 "²": "2", "³": "3", "(": "", ")": "", "-": "_"}

AGGREGATION_MAP = {
    "precipitation_mm": ("sum", "max"),
    "rain_mm": ("sum",),
    "relative_humidity_2m_pct": ("mean", "min", "max"),
    "temperature_2m_degc": ("mean", "min", "max"),
    "dew_point_2m_degc": ("mean",),
    "vapour_pressure_deficit_kpa": ("mean", "max"),
    "pressure_msl_hpa": ("mean",),
    "surface_pressure_hpa": ("mean",),
    "wind_speed_10m_km_per_h": ("mean",),
    "wind_gusts_10m_km_per_h": ("mean", "max"),
    "soil_moisture_0_to_7cm_m3_per_m3": ("mean", "max"),
    "soil_moisture_7_to_28cm_m3_per_m3": ("mean",),
    "soil_moisture_28_to_100cm_m3_per_m3": ("mean",),
    "soil_moisture_100_to_255cm_m3_per_m3": ("mean",),
    "cloud_cover_pct": ("mean",),
    "et0_fao_evapotranspiration_mm": ("sum",),
    "shortwave_radiation_w_per_m2": ("mean",),
}
GEO_COLUMNS = ("latitude", "longitude", "elevation")

# Features connues au jour t (AUCUNE ne provient de la fenêtre-cible t+1…t+h)
SELECTED_FEATURES = [
    # --- pluie antécédente (fenêtres glissantes se terminant à t) ---
    "precipitation_mm_sum", "precipitation_mm_max",
    "precip_3d_sum", "precip_7d_sum", "precip_15d_sum",
    # --- état de saturation du sol (précurseur clé, absent du label) ---
    "soil_moisture_0_to_7cm_m3_per_m3_mean",
    "soil_moisture_7_to_28cm_m3_per_m3_mean",
    "soil_moisture_28_to_100cm_m3_per_m3_mean",
    "soil_moisture_100_to_255cm_m3_per_m3_mean",
    "soil_moisture_gradient",
    # --- précurseurs atmosphériques ---
    "relative_humidity_2m_pct_mean", "relative_humidity_2m_pct_max",
    "dew_point_2m_degc_mean",
    "vapour_pressure_deficit_kpa_mean",
    "pressure_msl_hpa_mean", "pressure_tendency_1d", "pressure_tendency_3d",
    "cloud_cover_pct_mean",
    "temperature_2m_degc_mean", "temperature_2m_degc_max",
    "wind_speed_10m_km_per_h_mean", "wind_gusts_10m_km_per_h_max",
    "et0_fao_evapotranspiration_mm_sum",
    # --- saisonnalité + géographie ---
    "dayofyear_sin", "dayofyear_cos", "is_rainy_season",
    "latitude", "longitude", "elevation",
]

RF_PARAMS = dict(
    n_estimators=400, max_depth=12, min_samples_leaf=5,
    class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
)


# --------------------------------------------------------------------------- #
# Chargement + agrégation journalière (repris de model.ipynb, corrigé)
# --------------------------------------------------------------------------- #
def normalise_column(label: str) -> str:
    label = label.strip().lower()
    for src, dst in SUBSTITUTIONS.items():
        label = label.replace(src, dst)
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_")


def coerce_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def location_from_path(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 3 and all(p.isdigit() for p in parts[-2:]):
        return "_".join(parts[:-2])
    return path.stem


def load_hourly(data_dir: Path) -> pd.DataFrame:
    frames = []
    csv_files = list(data_dir.glob("*.csv"))
    if EXTRA_DIR.exists():                       # zones supplémentaires (même format)
        csv_files += list(EXTRA_DIR.glob("*.csv"))
    for csv in sorted(csv_files):
        meta = pd.read_csv(csv, nrows=1).iloc[0].to_dict()
        meta = {normalise_column(k): v for k, v in meta.items()}
        frame = pd.read_csv(csv, skiprows=3)
        frame.columns = [normalise_column(c) for c in frame.columns]
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        frame = frame.dropna(subset=["time"])
        num = [c for c in frame.columns if c != "time"]
        frame[num] = frame[num].apply(pd.to_numeric, errors="coerce")
        frame["location"] = location_from_path(csv)
        for g in ("latitude", "longitude", "elevation"):
            frame[g] = coerce_float(meta.get(g))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["location", "time"])


def aggregate_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    df = hourly.copy()
    df["date"] = df["time"].dt.floor("D")
    agg = {f: list(s) for f, s in AGGREGATION_MAP.items() if f in df.columns}
    base = {c: "first" for c in GEO_COLUMNS if c in df.columns}
    g = df.groupby(["location", "date"]).agg({**base, **agg}).dropna(how="all")
    g.columns = [b if (not s or s == "first") else f"{b}_{s}" for b, s in g.columns]
    for idx in g.index.names:
        if idx in g.columns:
            g = g.drop(columns=[idx])
    return g.reset_index()


def add_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Features connues à t : fenêtres glissantes TRAILING + saison + tendance."""
    daily = daily.copy()
    daily["month"] = daily["date"].dt.month
    doy = daily["date"].dt.dayofyear
    daily["dayofyear_sin"] = np.sin(2 * math.pi * doy / 365.25)
    daily["dayofyear_cos"] = np.cos(2 * math.pi * doy / 365.25)
    daily["is_rainy_season"] = daily["month"].isin(RAINY_MONTHS).astype(int)

    out = []
    for _, grp in daily.groupby("location", group_keys=False):
        g = grp.sort_values("date").copy()
        p = g["precipitation_mm_sum"]
        # fenêtres TRAILING (se terminent à t -> antécédentes, pas de fuite)
        g["precip_3d_sum"] = p.rolling(3, min_periods=1).sum()
        g["precip_7d_sum"] = p.rolling(7, min_periods=1).sum()
        g["precip_15d_sum"] = p.rolling(15, min_periods=1).sum()
        st = g.get("soil_moisture_0_to_7cm_m3_per_m3_mean")
        ss = g.get("soil_moisture_7_to_28cm_m3_per_m3_mean")
        if st is not None and ss is not None:
            g["soil_moisture_gradient"] = st - ss
        pr = g.get("pressure_msl_hpa_mean")
        if pr is not None:
            # chute de pression = précurseur de perturbation (connu à t)
            g["pressure_tendency_1d"] = pr - pr.shift(1)
            g["pressure_tendency_3d"] = pr - pr.shift(3)
        out.append(g)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# Cible FUTURE (t+1 … t+h) — la clé de la refonte
# --------------------------------------------------------------------------- #
def add_future_target(daily: pd.DataFrame, horizon: int,
                      thresholds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    y_t = 1 si, sur la fenêtre FUTURE (t+1 … t+h) :
              pluie journalière max > P97(zone)   OU
              cumul futur           > P95_3j(zone)
    Les seuils sont calibrés sur le TRAIN uniquement (cf. calibrate_thresholds).
    Aucune valeur future n'est ajoutée aux features : precip_future_* sert
    seulement à construire y puis est supprimée.
    """
    daily = daily.copy()
    out = []
    for loc, grp in daily.groupby("location", group_keys=False):
        g = grp.sort_values("date").copy()
        p = g["precipitation_mm_sum"]
        # somme de pluie sur les h jours FUTURS (t+1 … t+h)
        fut_sum = sum(p.shift(-k) for k in range(1, horizon + 1))
        # pluie journalière FUTURE maximale sur la fenêtre
        fut_max = pd.concat([p.shift(-k) for k in range(1, horizon + 1)], axis=1).max(axis=1)
        p97, p95 = thresholds[loc]
        g["_precip_future_sum"] = fut_sum
        g["_precip_future_max"] = fut_max
        g["flood_risk"] = ((fut_max >= p97) | (fut_sum >= p95)).astype("float")
        # invalider les t dont la fenêtre future est incomplète (fin de série)
        g.loc[p.shift(-horizon).isna(), "flood_risk"] = np.nan
        out.append(g)
    return pd.concat(out, ignore_index=True)


def calibrate_thresholds(train: pd.DataFrame, horizon: int) -> Dict[str, Tuple[float, float]]:
    """Seuils par zone calculés sur le TRAIN (P97 pluie/jour, P95 cumul h-jours)."""
    th = {}
    for loc, g in train.groupby("location"):
        g = g.sort_values("date")
        p = g["precipitation_mm_sum"]
        cum_h = p.rolling(horizon, min_periods=1).sum()  # distribution des cumuls h-jours
        th[loc] = (float(p.quantile(0.97)), float(cum_h.quantile(0.95)))
    return th


# --------------------------------------------------------------------------- #
# Anti-fuite : vérification explicite
# --------------------------------------------------------------------------- #
def assert_no_leakage(features: List[str]) -> None:
    banned = ["_precip_future", "flood_risk"]
    leaks = [f for f in features if any(b in f for b in banned)]
    assert not leaks, f"FUITE DÉTECTÉE : {leaks} dérivent du futur / du label"
    print("✓ Anti-fuite : aucune feature ne provient de la fenêtre-cible.")


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #
def metrics_at_threshold(y, proba, thr) -> Dict[str, float]:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        roc_auc=roc_auc_score(y, proba),
        pr_auc=average_precision_score(y, proba),
        brier=brier_score_loss(y, proba),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        accuracy=accuracy_score(y, pred),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=float(thr),
    )


def pick_threshold_for_recall(y, proba, target_recall=0.90) -> float:
    """Seuil le plus élevé (max précision) atteignant target_recall — choisi sur TRAIN."""
    order = np.argsort(-proba)
    ys = y[order]
    psorted = proba[order]
    pos = y.sum()
    if pos == 0:
        return 0.5
    tp = np.cumsum(ys)
    recall = tp / pos
    ok = np.where(recall >= target_recall)[0]
    if len(ok) == 0:
        return 0.5
    return float(psorted[ok[0]])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(horizon: int) -> None:
    print(f"\n{'='*68}\n PRÉVISION D'INONDATION À J+1…J+{horizon} — Random Forest\n{'='*68}")

    hourly = load_hourly(DATA_DIR)
    daily = aggregate_daily(hourly)
    daily = add_features(daily)

    # 1) split temporel AVANT calibration des seuils (pas de fuite)
    train_raw = daily[daily["date"] < SPLIT_DATE]
    thresholds = calibrate_thresholds(train_raw, horizon)
    print("\nSeuils calibrés par zone (train) — (P97 pluie/j, P95 cumul {}j) :".format(horizon))
    for loc, (a, b) in thresholds.items():
        print(f"  {loc:16s}  P97={a:5.1f} mm   P95_{horizon}j={b:6.1f} mm")

    # 2) cible future construite sur toute la série (features à t, label à t+1…t+h)
    daily = add_future_target(daily, horizon, thresholds)

    # 3) on modélise la saison à risque (juin–octobre), label déjà défini sur série complète
    model_rows = daily[daily["month"].isin(RAINY_MONTHS)].dropna(subset=["flood_risk"]).copy()
    model_rows["flood_risk"] = model_rows["flood_risk"].astype(int)

    train = model_rows[model_rows["date"] < SPLIT_DATE].copy()
    test = model_rows[model_rows["date"] >= SPLIT_DATE].copy()

    assert_no_leakage(SELECTED_FEATURES)

    def matrix(df):
        X = df[SELECTED_FEATURES].ffill().bfill().fillna(0).to_numpy(float)
        return X, df["flood_risk"].to_numpy(int)

    Xtr, ytr = matrix(train)
    Xte, yte = matrix(test)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    print(f"\nJeux : train={len(ytr)} obs ({ytr.mean()*100:.1f}% à risque) | "
          f"test={len(yte)} obs ({yte.mean()*100:.1f}% à risque)")

    # ------------------------------------------------------------------ #
    # 4) Modèle + baselines (pour prouver un vrai gain prédictif)
    # ------------------------------------------------------------------ #
    rf = RandomForestClassifier(**RF_PARAMS).fit(Xtr_s, ytr)
    rf_tr = rf.predict_proba(Xtr_s)[:, 1]
    rf_te = rf.predict_proba(Xte_s)[:, 1]

    logit = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr_s, ytr)
    lg_te = logit.predict_proba(Xte_s)[:, 1]

    # baseline persistance : "aujourd'hui est déjà un jour de fort cumul antécédent"
    p95_map = {loc: b for loc, (a, b) in thresholds.items()}
    persist_te = (test["precip_3d_sum"].to_numpy()
                  >= test["location"].map(p95_map).to_numpy()).astype(float)

    thr = pick_threshold_for_recall(ytr, rf_tr, target_recall=0.90)

    rf_m = metrics_at_threshold(yte, rf_te, thr)
    lg_m = metrics_at_threshold(yte, lg_te, 0.5)
    ps_m = metrics_at_threshold(yte, persist_te, 0.5)

    # ------------------------------------------------------------------ #
    # 5) Validation croisée temporelle (ROC-AUC, sans seuil)
    # ------------------------------------------------------------------ #
    tscv = TimeSeriesSplit(n_splits=5)
    order = train.sort_values("date").index
    Xo, yo = matrix(train.loc[order])
    cv_auc = []
    for tr_i, va_i in tscv.split(Xo):
        sc = StandardScaler().fit(Xo[tr_i])
        m = RandomForestClassifier(**RF_PARAMS).fit(sc.transform(Xo[tr_i]), yo[tr_i])
        pv = m.predict_proba(sc.transform(Xo[va_i]))[:, 1]
        cv_auc.append(roc_auc_score(yo[va_i], pv))
    cv_auc = np.array(cv_auc)

    # ------------------------------------------------------------------ #
    # 6) Importances (Gini + permutation, plus honnête)
    # ------------------------------------------------------------------ #
    gini = pd.Series(rf.feature_importances_, index=SELECTED_FEATURES).sort_values(ascending=False)
    perm = permutation_importance(rf, Xte_s, yte, n_repeats=10,
                                  random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
    perm_s = pd.Series(perm.importances_mean, index=SELECTED_FEATURES).sort_values(ascending=False)

    # ------------------------------------------------------------------ #
    # Rapport
    # ------------------------------------------------------------------ #
    def show(name, m):
        print(f"  {name:22s} AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  "
              f"[TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}]")

    print(f"\n--- TEST 2023–2025 (seuil RF={thr:.3f} calibré pour rappel≈0,90 sur train) ---")
    show("Random Forest", rf_m)
    show("Régression logistique", lg_m)
    show("Persistance (baseline)", ps_m)
    print(f"\nValidation croisée temporelle (5 folds) ROC-AUC : "
          f"{cv_auc.mean():.3f} ± {cv_auc.std():.3f}  {np.round(cv_auc,3).tolist()}")
    print(f"Brier score RF (test) : {rf_m['brier']:.4f}  (0=parfait)")

    print("\nTop-12 importances Gini :")
    for k, v in gini.head(12).items():
        print(f"  {k:42s} {v*100:5.1f}%")
    print("\nTop-12 importances par PERMUTATION (chute d'AUC, test) :")
    for k, v in perm_s.head(12).items():
        print(f"  {k:42s} {v:+.4f}")

    results = dict(
        horizon=horizon, thresholds=thresholds,
        n_train=len(ytr), n_test=len(yte),
        rate_train=float(ytr.mean()), rate_test=float(yte.mean()),
        random_forest=rf_m, logistic=lg_m, persistence=ps_m,
        cv_auc_mean=float(cv_auc.mean()), cv_auc_std=float(cv_auc.std()),
        cv_auc_folds=np.round(cv_auc, 4).tolist(),
        gini_importance=gini.round(4).to_dict(),
        permutation_importance=perm_s.round(4).to_dict(),
    )
    out = OUT_DIR / f"results_h{horizon}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n✓ Résultats sauvegardés : {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=3, help="horizon de prévision en jours")
    main(ap.parse_args().horizon)
