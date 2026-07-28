"""Régénère la figure des matrices de confusion CNN AVEC LABELS ANGLAIS,
à partir des valeurs exactes du run de flood_comparaison.ipynb.
(Les matrices sont connues ; aucun réentraînement nécessaire.)"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FIG = Path(__file__).resolve().parent / "rf_forecast_results/figures"
sns.set_theme(style="white")

# Valeurs exactes du DERNIER run (rows/cols = [Flood, No flood])
RESULTS = {
    "ResNet-18":       dict(cm=[[48, 19], [0, 46]], acc=83.19, rec=71.64, color="#3498db"),
    "EfficientNet-B3": dict(cm=[[58, 9],  [2, 44]], acc=90.27, rec=86.57, color="#e74c3c"),
    "ConvNeXt-Tiny":   dict(cm=[[64, 3],  [0, 46]], acc=97.35, rec=95.52, color="#2ecc71"),
    "Swin-Tiny":       dict(cm=[[64, 3],  [0, 46]], acc=97.35, rec=95.52, color="#f39c12"),
}
CLASSES = ["Flood", "No flood"]

fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
for ax, (name, r) in zip(axes, RESULTS.items()):
    sns.heatmap(np.array(r["cm"]), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.5, cbar=False, annot_kws={"size": 14})
    ax.set_title(f'{name}\nAcc = {r["acc"]:.2f}%  |  Flood recall = {r["rec"]:.2f}%',
                 fontsize=10, fontweight="bold", color=r["color"])
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)

plt.suptitle("Confusion matrices of the four architectures (113 validation images)",
             fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(FIG / "en_cnn_confusion.png", dpi=200, bbox_inches="tight")
print("✓ en_cnn_confusion.png written to", FIG)
