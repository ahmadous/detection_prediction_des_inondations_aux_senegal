"""Télécharge les VRAIES prévisions à échéance (Open-Meteo Previous Runs API).

Pour chaque jour valide D, `precipitation_previous_dayK` = pluie prévue pour D
par le run émis K jours plus tôt (à D-K). Donc pour un jour d'émission t, la
prévision de la fenêtre future t+1…t+3 s'obtient par :
    fc(t+1) = lead1 valide en t+1 ; fc(t+2) = lead2 valide en t+2 ; fc(t+3) = lead3 valide en t+3
=> émise à t, disponible à t : pas de fuite.

Archive disponible ~2024-03 -> 2025-07 (démonstration de principe).
"""
import time
from pathlib import Path
import requests
import pandas as pd

OUT = Path(__file__).resolve().parent / "donnees_previous_runs"
OUT.mkdir(exist_ok=True)
ZONES = {
    "kaolack_leona": (13.813708, -15.551483), "keur_massar": (15.43058, -15.887329),
    "kolda": (12.899824, -14.959137), "matam": (15.641477, -13.220337),
    "tambacounda": (13.743409, -13.636383), "touba": (14.86819, -15.852753),
}
HOURLY = ["precipitation", "precipitation_previous_day1",
          "precipitation_previous_day2", "precipitation_previous_day3"]
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

for zone, (lat, lon) in ZONES.items():
    params = dict(latitude=lat, longitude=lon, start_date="2024-01-01",
                  end_date="2025-07-31", hourly=",".join(HOURLY), timezone="GMT")
    for attempt in range(4):
        try:
            r = requests.get(URL, params=params, timeout=90)
            r.raise_for_status()
            h = r.json()["hourly"]
            df = pd.DataFrame(h)
            df["date"] = pd.to_datetime(df["time"]).dt.floor("D")
            # somme journalière de la pluie réelle et des prévisions par échéance
            daily = df.groupby("date").agg(
                actual=("precipitation", "sum"),
                fc_lead1=("precipitation_previous_day1", "sum"),
                fc_lead2=("precipitation_previous_day2", "sum"),
                fc_lead3=("precipitation_previous_day3", "sum"),
            ).reset_index()
            daily["location"] = zone
            daily.to_csv(OUT / f"{zone}_prevruns.csv", index=False)
            nn = daily["fc_lead3"].notna().sum()
            print(f"✓ {zone:16s} {len(daily)} jours ({nn} avec prév J-3) -> {zone}_prevruns.csv")
            break
        except Exception as e:
            print(f"  {zone} tentative {attempt+1}: {e}")
            time.sleep(4)
    time.sleep(1)
print("Terminé.")
