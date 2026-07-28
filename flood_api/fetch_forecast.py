"""Télécharge les PRÉVISIONS météo archivées (Open-Meteo Historical Forecast API)
pour les 6 zones, 2022-2025. Ces prévisions, émises AVANT l'échéance, sont
disponibles au jour t -> légitimes comme variables d'entrée (pas de fuite).
"""
import time
from pathlib import Path
import requests

OUT = Path(__file__).resolve().parent / "donnees_forecast"
OUT.mkdir(exist_ok=True)

ZONES = {
    "kaolack_leona": (13.813708, -15.551483),
    "keur_massar": (15.43058, -15.887329),
    "kolda": (12.899824, -14.959137),
    "matam": (15.641477, -13.220337),
    "tambacounda": (13.743409, -13.636383),
    "touba": (14.86819, -15.852753),
}
DAILY = ["precipitation_sum", "precipitation_hours", "precipitation_probability_max",
         "temperature_2m_max", "wind_speed_10m_max", "shortwave_radiation_sum"]
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

for zone, (lat, lon) in ZONES.items():
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": "2022-01-01", "end_date": "2025-07-31",
        "daily": ",".join(DAILY), "timezone": "GMT",
    }
    for attempt in range(4):
        try:
            r = requests.get(URL, params=params, timeout=60)
            r.raise_for_status()
            d = r.json()["daily"]
            import csv
            path = OUT / f"{zone}_forecast.csv"
            cols = ["time"] + DAILY
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                for i in range(len(d["time"])):
                    w.writerow([d["time"][i]] + [d[c][i] for c in DAILY])
            print(f"✓ {zone:16s} {len(d['time'])} jours -> {path.name}")
            break
        except Exception as e:
            print(f"  {zone} tentative {attempt+1} échec: {e}")
            time.sleep(3)
    time.sleep(1)
print("Terminé.")
