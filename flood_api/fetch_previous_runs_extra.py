"""Prévisions à échéance (Previous Runs API) pour les 12 zones ajoutées, 2024-2025.
Sauvegarde dans donnees_previous_runs/ (mêmes fichiers que les 6 zones initiales)."""
import time
from pathlib import Path
import requests
import pandas as pd

OUT = Path(__file__).resolve().parent / "donnees_previous_runs"
OUT.mkdir(exist_ok=True)
ZONES = {
    "dakar": (14.6928, -17.4467), "saint_louis": (16.0326, -16.4818),
    "ziguinchor": (12.5641, -16.2719), "diourbel": (14.6561, -16.2314),
    "thies": (14.7910, -16.9256), "fatick": (14.3390, -16.4110),
    "louga": (15.6194, -16.2264), "kaffrine": (14.1057, -15.5416),
    "sedhiou": (12.7081, -15.5569), "kedougou": (12.5556, -12.1747),
    "bakel": (14.9010, -12.4665), "podor": (16.6518, -14.9592),
}
HOURLY = ["precipitation", "precipitation_previous_day1",
          "precipitation_previous_day2", "precipitation_previous_day3"]
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

for zone, (lat, lon) in ZONES.items():
    dest = OUT / f"{zone}_prevruns.csv"
    if dest.exists() and dest.stat().st_size > 5000:
        print(f"• {zone} déjà présent"); continue
    params = dict(latitude=lat, longitude=lon, start_date="2024-01-01",
                  end_date="2025-07-31", hourly=",".join(HOURLY), timezone="GMT")
    for attempt in range(4):
        try:
            r = requests.get(URL, params=params, timeout=90); r.raise_for_status()
            h = r.json()["hourly"]; df = pd.DataFrame(h)
            df["date"] = pd.to_datetime(df["time"]).dt.floor("D")
            daily = df.groupby("date").agg(
                actual=("precipitation", "sum"),
                fc_lead1=("precipitation_previous_day1", "sum"),
                fc_lead2=("precipitation_previous_day2", "sum"),
                fc_lead3=("precipitation_previous_day3", "sum")).reset_index()
            daily["location"] = zone
            daily.to_csv(dest, index=False)
            print(f"✓ {zone:14s} {len(daily)} jours")
            break
        except Exception as e:
            print(f"  {zone} tentative {attempt+1}: {str(e)[:100]}"); time.sleep(4)
    time.sleep(1)
print("Terminé.")
