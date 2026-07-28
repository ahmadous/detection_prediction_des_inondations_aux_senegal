"""Agrandit le jeu de données : réanalyse ERA5 (archive API) pour de nouvelles
zones du Sénégal, 2005-2025, au format Open-Meteo CSV (compatible avec le
pipeline existant). Sauvegarde dans donnees_extra/."""
import time
from pathlib import Path
import requests

OUT = Path(__file__).resolve().parent / "donnees_extra"
OUT.mkdir(exist_ok=True)

# nouvelles zones (villes/points exposés, réparties sur le territoire)
ZONES = {
    "dakar": (14.6928, -17.4467), "saint_louis": (16.0326, -16.4818),
    "ziguinchor": (12.5641, -16.2719), "diourbel": (14.6561, -16.2314),
    "thies": (14.7910, -16.9256), "fatick": (14.3390, -16.4110),
    "louga": (15.6194, -16.2264), "kaffrine": (14.1057, -15.5416),
    "sedhiou": (12.7081, -15.5569), "kedougou": (12.5556, -12.1747),
    "bakel": (14.9010, -12.4665), "podor": (16.6518, -14.9592),
}
HOURLY = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
          "pressure_msl", "cloud_cover", "vapour_pressure_deficit",
          "et0_fao_evapotranspiration", "wind_speed_10m", "wind_gusts_10m",
          "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
          "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm"]
URL = "https://archive-api.open-meteo.com/v1/archive"

for zone, (lat, lon) in ZONES.items():
    dest = OUT / f"{zone}_2005_2025.csv"
    if dest.exists() and dest.stat().st_size > 10000:
        print(f"• {zone:14s} déjà présent, saut"); continue
    params = dict(latitude=lat, longitude=lon, start_date="2005-04-07",
                  end_date="2025-07-31", hourly=",".join(HOURLY),
                  format="csv", timezone="GMT")
    for attempt in range(4):
        try:
            r = requests.get(URL, params=params, timeout=180)
            r.raise_for_status()
            if r.text.lstrip().startswith("{"):
                raise RuntimeError(r.text[:200])  # erreur JSON
            dest.write_text(r.text)
            print(f"✓ {zone:14s} {dest.stat().st_size//1024} Ko -> {dest.name}")
            break
        except Exception as e:
            print(f"  {zone} tentative {attempt+1}: {str(e)[:120]}")
            time.sleep(6)
    time.sleep(2)
print("Terminé.")
