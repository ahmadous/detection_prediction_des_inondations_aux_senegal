"""Figure 1 : carte des 18 zones d'étude, colorées par mécanisme d'inondation."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIG = Path(__file__).resolve().parent / "rf_forecast_results/figures"
FIG.mkdir(parents=True, exist_ok=True)

# zone : (lat, lon, label, mécanisme)
ZONES = [
    (14.798, -17.346, "Dakar periphery", "Urban pluvial"),
    (14.868, -15.853, "Touba", "Urban pluvial"),
    (14.798, -15.922, "Mbacké", "Urban pluvial"),
    (14.657, -16.227, "Diourbel", "Urban pluvial"),
    (14.798, -16.927, "Thiès", "Urban pluvial"),
    (14.165, -16.039, "Kaolack", "Groundwater rise"),
    (14.095, -15.526, "Kaffrine", "Groundwater rise"),
    (14.306, -16.401, "Fatick", "Groundwater rise"),
    (15.641, -13.220, "Matam", "Fluvial"),
    (16.626, -14.943, "Podor", "Fluvial"),
    (14.868, -12.498, "Bakel", "Fluvial"),
    (16.063, -16.449, "Saint-Louis", "Fluvial + marine"),
    (12.548, -16.275, "Ziguinchor", "Intense rainfall"),
    (12.900, -14.959, "Kolda", "Intense rainfall"),
    (12.689, -15.571, "Sédhiou", "Intense rainfall"),
    (13.743, -13.636, "Tambacounda", "Intense rainfall"),
    (12.548, -12.206, "Kédougou", "Intense rainfall"),
    (15.641, -16.186, "Louga", "Semi-arid"),
]
COLORS = {
    "Urban pluvial": "#C44E52", "Groundwater rise": "#4C72B0",
    "Fluvial": "#55A868", "Fluvial + marine": "#8172B3",
    "Intense rainfall": "#CCB974", "Semi-arid": "#937860",
}

fig, ax = plt.subplots(figsize=(9, 7))

# frontière du Sénégal
gj = json.load(open("/tmp/senegal.geojson"))
for feat in gj["features"]:
    geom = feat["geometry"]
    polys = geom["coordinates"] if geom["type"] == "Polygon" else [p[0] for p in geom["coordinates"]]
    for ring in polys:
        xs = [c[0] for c in ring]; ys = [c[1] for c in ring]
        ax.plot(xs, ys, color="#333333", lw=1.2)
        ax.fill(xs, ys, color="#f2f2f2", zorder=0)

# zones
for lat, lon, label, mech in ZONES:
    ax.scatter(lon, lat, s=90, color=COLORS[mech], edgecolor="white",
               linewidth=0.8, zorder=3)
    ax.annotate(label, (lon, lat), textcoords="offset points", xytext=(5, 4),
                fontsize=8, color="#222222")

handles = [Patch(color=c, label=m) for m, c in COLORS.items()]
ax.legend(handles=handles, title="Flood mechanism", loc="lower left", fontsize=9, title_fontsize=9)
ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
ax.set_title("Fig. 1. The eighteen flood-prone study zones in Senegal", fontsize=12, weight="bold")
ax.set_aspect(1.0); ax.grid(True, ls=":", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG / "fig1_zones_map.png", dpi=150, bbox_inches="tight")
print("✓ Carte enregistrée :", FIG / "fig1_zones_map.png")
