"""Géocodage inverse (Nominatim/OSM) des zones -> vrais noms pour citation.
Respecte la limite de 1 requête/seconde."""
import time
from pathlib import Path
import requests
import pandas as pd

BASE = Path(__file__).resolve().parent
UA = {"User-Agent": "flood-research/1.0 (paseydou.sow@univ-thies.sn)"}


def coords(csv):
    line = csv.read_text().splitlines()[1].split(",")
    return float(line[0]), float(line[1])


def reverse(lat, lon):
    try:
        r = requests.get("https://nominatim.openstreetmap.org/reverse",
                         params=dict(lat=lat, lon=lon, format="json", zoom=12,
                                     **{"accept-language": "fr"}), headers=UA, timeout=25)
        a = r.json().get("address", {})
        place = (a.get("city") or a.get("town") or a.get("village") or
                 a.get("suburb") or a.get("municipality") or a.get("county") or "?")
        dept = a.get("county") or a.get("state_district") or ""
        region = a.get("state") or ""
        return place, dept, region
    except Exception as e:
        return f"erreur:{e}", "", ""


rows = []
files = sorted(BASE.glob("donnees/*.csv")) + sorted(BASE.glob("donnees_extra/*.csv"))
for csv in files:
    label = csv.stem.replace("_2005_2025", "")
    origin = "mémoire" if "donnees/" in str(csv) else "ajoutée"
    lat, lon = coords(csv)
    place, dept, region = reverse(lat, lon)
    rows.append(dict(zone=label, origine=origin, lat=lat, lon=lon,
                     lieu_reel=place, departement=dept, region=region))
    print(f"{label:16s} ({origin:7s}) {lat:.3f},{lon:.3f} -> {place} / {dept} / {region}")
    time.sleep(1.1)

pd.DataFrame(rows).to_csv(BASE / "rf_forecast_results/zones_verification.csv", index=False)
print("\n✓ Sauvé : rf_forecast_results/zones_verification.csv")
