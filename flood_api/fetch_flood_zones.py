"""Ajoute/corrige des zones INONDABLES (réanalyse 2005-2025 + prévisions 2024-2025).
Espacement + backoff pour éviter les 429 d'Open-Meteo.
Corrige keur_massar & kaolack_leona (anciens points ruraux archivés)."""
import time
from pathlib import Path
import requests
import pandas as pd

BASE = Path(__file__).resolve().parent
EXTRA = BASE / "donnees_extra"; EXTRA.mkdir(exist_ok=True)
PREV = BASE / "donnees_previous_runs"; PREV.mkdir(exist_ok=True)
ARCH = BASE / "donnees" / "_archive_mislabeled"; ARCH.mkdir(exist_ok=True)

# nom -> (lat, lon, mécanisme d'inondation)
ZONES = {
    "keur_massar": (14.782, -17.316, "pluviale urbaine (banlieue Dakar)"),      # corrige
    "kaolack_leona": (14.152, -16.072, "remontée de nappe (bassin arachidier)"),  # corrige
    "parcelles_assainies": (14.766, -17.427, "pluviale urbaine (banlieue Dakar)"),
    "pikine": (14.755, -17.396, "pluviale urbaine (banlieue Dakar)"),
    "guediawaye": (14.778, -17.410, "pluviale urbaine (banlieue Dakar)"),
    "mbacke": (14.790, -15.909, "pluviale urbaine (Touba/Mbacké)"),
    "podor": (16.652, -14.959, "crue fluviale (vallée du fleuve)"),
    "richard_toll": (16.462, -15.700, "crue fluviale (vallée du fleuve)"),
}

REANALYSIS_HOURLY = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "pressure_msl", "cloud_cover", "vapour_pressure_deficit",
    "et0_fao_evapotranspiration", "wind_speed_10m", "wind_gusts_10m",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm"]
PREV_HOURLY = ["precipitation", "precipitation_previous_day1",
               "precipitation_previous_day2", "precipitation_previous_day3"]


def get(url, params, timeout):
    for attempt in range(6):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                raise RuntimeError("429")
            r.raise_for_status()
            return r
        except Exception as e:
            wait = min(15 * (attempt + 1), 70)
            print(f"    retry {attempt+1} ({str(e)[:40]}) — attente {wait}s")
            time.sleep(wait)
    return None


# archiver les anciens fichiers mal placés (réversible)
for z in ("keur_massar", "kaolack_leona"):
    old = BASE / "donnees" / f"{z}_2005_2025.csv"
    if old.exists():
        old.rename(ARCH / old.name)
        print(f"↪ archivé (ancien point rural) : {old.name}")

for zone, (lat, lon, mech) in ZONES.items():
    # --- réanalyse (2005-2025) ---
    dest = EXTRA / f"{zone}_2005_2025.csv"
    if dest.exists() and dest.stat().st_size > 10000:
        print(f"• {zone} réanalyse déjà présente")
    else:
        r = get("https://archive-api.open-meteo.com/v1/archive",
                dict(latitude=lat, longitude=lon, start_date="2005-04-07",
                     end_date="2025-07-31", hourly=",".join(REANALYSIS_HOURLY),
                     format="csv", timezone="GMT"), 180)
        if r and not r.text.lstrip().startswith("{"):
            dest.write_text(r.text)
            print(f"✓ réanalyse  {zone:20s} [{mech}]")
        else:
            print(f"✗ ÉCHEC réanalyse {zone}")
        time.sleep(20)

    # --- prévisions (2024-2025) ---
    pdest = PREV / f"{zone}_prevruns.csv"
    r = get("https://previous-runs-api.open-meteo.com/v1/forecast",
            dict(latitude=lat, longitude=lon, start_date="2024-01-01",
                 end_date="2025-07-31", hourly=",".join(PREV_HOURLY), timezone="GMT"), 90)
    if r:
        h = r.json()["hourly"]; df = pd.DataFrame(h)
        df["date"] = pd.to_datetime(df["time"]).dt.floor("D")
        daily = df.groupby("date").agg(actual=("precipitation", "sum"),
            fc_lead1=("precipitation_previous_day1", "sum"),
            fc_lead2=("precipitation_previous_day2", "sum"),
            fc_lead3=("precipitation_previous_day3", "sum")).reset_index()
        daily["location"] = zone
        daily.to_csv(pdest, index=False)
        print(f"✓ prévision  {zone}")
    else:
        print(f"✗ ÉCHEC prévision {zone}")
    time.sleep(8)

print("Terminé.")
