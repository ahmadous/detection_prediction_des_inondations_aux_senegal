"""Régénère en FRANÇAIS PROPRE (sans tiret cadratin) les figures locales de
l'article, avec le suffixe _fr : carte, ROC/PR, confusion+calibration,
importances, apport prévision, fusion. (Les figures CNN viennent du Colab.)"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_forecast as base
from rf_forecast_v2 import build_dataset, matrix, rf_factory, hgb_factory, VAL_START, SPLIT_DATE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 130, "axes.titleweight": "semibold"})
FIG = Path(__file__).resolve().parent / "rf_forecast_results/figures"
H = 3

# ================= CARTE (fig1_zones_map_fr) =================
ZONES = [
    (14.798,-17.346,"Banlieue de Dakar","Pluviale urbaine"),(14.868,-15.853,"Touba","Pluviale urbaine"),
    (14.798,-15.922,"Mbacké","Pluviale urbaine"),(14.657,-16.227,"Diourbel","Pluviale urbaine"),
    (14.798,-16.927,"Thiès","Pluviale urbaine"),(14.165,-16.039,"Kaolack","Remontée de nappe"),
    (14.095,-15.526,"Kaffrine","Remontée de nappe"),(14.306,-16.401,"Fatick","Remontée de nappe"),
    (15.641,-13.220,"Matam","Crue fluviale"),(16.626,-14.943,"Podor","Crue fluviale"),
    (14.868,-12.498,"Bakel","Crue fluviale"),(16.063,-16.449,"Saint-Louis","Crue et intrusion marine"),
    (12.548,-16.275,"Ziguinchor","Pluies intenses"),(12.900,-14.959,"Kolda","Pluies intenses"),
    (12.689,-15.571,"Sédhiou","Pluies intenses"),(13.743,-13.636,"Tambacounda","Pluies intenses"),
    (12.548,-12.206,"Kédougou","Pluies intenses"),(15.641,-16.186,"Louga","Semi-aride")]
COLORS={"Pluviale urbaine":"#C44E52","Remontée de nappe":"#4C72B0","Crue fluviale":"#55A868",
        "Crue et intrusion marine":"#8172B3","Pluies intenses":"#CCB974","Semi-aride":"#937860"}
fig,ax=plt.subplots(figsize=(9,7))
for feat in json.load(open("/tmp/senegal.geojson"))["features"]:
    g=feat["geometry"]; polys=g["coordinates"] if g["type"]=="Polygon" else [p[0] for p in g["coordinates"]]
    for ring in polys:
        xs=[c[0] for c in ring]; ys=[c[1] for c in ring]
        ax.plot(xs,ys,color="#333",lw=1.2); ax.fill(xs,ys,color="#f2f2f2",zorder=0)
for lat,lon,label,mech in ZONES:
    ax.scatter(lon,lat,s=90,color=COLORS[mech],edgecolor="white",linewidth=0.8,zorder=3)
    ax.annotate(label,(lon,lat),textcoords="offset points",xytext=(5,4),fontsize=8,color="#222")
ax.legend(handles=[Patch(color=c,label=m) for m,c in COLORS.items()],title="Mécanisme d'inondation",
          loc="lower left",fontsize=9,title_fontsize=9)
ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
ax.set_title("Les dix-huit zones d'étude exposées aux inondations au Sénégal",fontsize=12,weight="bold")
ax.set_aspect(1.0); ax.grid(True,ls=":",alpha=0.4); plt.tight_layout()
plt.savefig(FIG/"fig1_zones_map_fr.png",dpi=150,bbox_inches="tight"); plt.close()
print("+ fig1_zones_map_fr.png")

# ================= MODÈLES (pour les figures de prévision) =================
rows,TH=build_dataset(H)
train=rows[rows.date<VAL_START]; val=rows[(rows.date>=VAL_START)&(rows.date<SPLIT_DATE)]; test=rows[rows.date>=SPLIT_DATE]
Xtr,ytr=matrix(train); Xval,yval=matrix(val); Xte,yte=matrix(test)
def lg(): return make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced"))
FACT={"Forêt aléatoire":rf_factory(10,10),"HistGradientBoosting":hgb_factory(5,0.05),"Régression logistique":lg}
def fitcal(f):
    e=f().fit(Xtr,ytr); return CalibratedClassifierCV(FrozenEstimator(e),method="isotonic").fit(Xval,yval)
M={k:fitcal(f) for k,f in FACT.items()}
P={k:m.predict_proba(Xte)[:,1] for k,m in M.items()}
THR={k:base.pick_threshold_for_recall(yval,m.predict_proba(Xval)[:,1]) for k,m in M.items()}
p95={l:b for l,(a,b) in TH.items()}
P["Persistance"]=(test["precip_3d_sum"].to_numpy()>=test["location"].map(p95).to_numpy()).astype(float)

# --- ROC + PR (fig2_roc_pr_fr) ---
fig,ax=plt.subplots(1,2,figsize=(11,4.4))
for k in ["Forêt aléatoire","HistGradientBoosting","Régression logistique","Persistance"]:
    fpr,tpr,_=roc_curve(yte,P[k]); ax[0].plot(fpr,tpr,lw=2,label=f"{k} (AUC={roc_auc_score(yte,P[k]):.3f})")
    pr,rc,_=precision_recall_curve(yte,P[k]); ax[1].plot(rc,pr,lw=2,label=k)
ax[0].plot([0,1],[0,1],"--",color="grey",lw=1)
ax[0].set(title="Courbes ROC (test 2023-2025)",xlabel="Taux de faux positifs",ylabel="Taux de vrais positifs"); ax[0].legend(fontsize=8)
ax[1].axhline(yte.mean(),ls="--",color="grey",lw=1,label=f"Hasard ({yte.mean():.2f})")
ax[1].set(title="Courbes précision-rappel",xlabel="Rappel",ylabel="Précision"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG/"fig2_roc_pr_fr.png",bbox_inches="tight"); plt.close(); print("+ fig2_roc_pr_fr.png")

# --- Confusion + calibration (fig3_confusion_calibration_fr) ---
fig,ax=plt.subplots(1,2,figsize=(11,4.4))
pred=(P["Forêt aléatoire"]>=THR["Forêt aléatoire"]).astype(int)
ConfusionMatrixDisplay(confusion_matrix(yte,pred),display_labels=["Sans risque","Risque"]).plot(ax=ax[0],cmap="Blues",colorbar=False)
ax[0].set(title=f"Matrice de confusion, forêt aléatoire (seuil {THR['Forêt aléatoire']:.2f})",xlabel="Étiquette prédite",ylabel="Étiquette réelle"); ax[0].grid(False)
for k in ["Forêt aléatoire","HistGradientBoosting","Régression logistique"]:
    fr,mp=calibration_curve(yte,P[k],n_bins=10,strategy="quantile"); ax[1].plot(mp,fr,"o-",label=k)
ax[1].plot([0,1],[0,1],"--",color="grey",lw=1,label="Calibration parfaite")
ax[1].set(title="Courbe de fiabilité",xlabel="Probabilité prédite",ylabel="Fréquence observée"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG/"fig3_confusion_calibration_fr.png",bbox_inches="tight"); plt.close(); print("+ fig3_confusion_calibration_fr.png")

# --- Importances (fig4_importances_fr) ---
rfm=rf_factory(10,10)().fit(Xtr,ytr); F=base.SELECTED_FEATURES
gini=pd.Series(rfm.steps[-1][1].feature_importances_,index=F).sort_values()
perm=permutation_importance(rfm,Xte,yte,n_repeats=8,random_state=42,scoring="roc_auc",n_jobs=-1)
perms=pd.Series(perm.importances_mean,index=F).sort_values()
fig,ax=plt.subplots(1,2,figsize=(13,6))
gini.tail(15).plot.barh(ax=ax[0],color="#4C72B0"); ax[0].set(title="Importance de Gini (forêt aléatoire)",xlabel="Diminution moyenne de l'impureté")
perms.tail(15).plot.barh(ax=ax[1],color="#C44E52"); ax[1].set(title="Importance par permutation (AUC de test)",xlabel="Baisse moyenne de l'AUC")
plt.tight_layout(); plt.savefig(FIG/"fig4_importances_fr.png",bbox_inches="tight"); plt.close(); print("+ fig4_importances_fr.png")

# --- Apport prévision (fig5_apport_prevision_fr) ---
PRD=Path(__file__).resolve().parent/"donnees_previous_runs"
pr=pd.concat([pd.read_csv(c) for c in sorted(PRD.glob("*_prevruns.csv"))],ignore_index=True); pr["date"]=pd.to_datetime(pr["date"])
out=[]
for loc,g in pr.groupby("location",group_keys=False):
    g=g.sort_values("date").copy()
    g["fc_next1"]=g["fc_lead1"].shift(-1); g["fc_next2"]=g["fc_lead2"].shift(-2); g["fc_next3"]=g["fc_lead3"].shift(-3)
    g["fc_sum_next3"]=g[["fc_next1","fc_next2","fc_next3"]].sum(axis=1); g["fc_max_next3"]=g[["fc_next1","fc_next2","fc_next3"]].max(axis=1); out.append(g)
pr=pd.concat(out,ignore_index=True); FC=["fc_next1","fc_sum_next3","fc_max_next3"]
df=rows.merge(pr[["location","date"]+FC],on=["location","date"],how="inner")
frr=df[df.date>=pd.Timestamp("2024-06-01")].dropna(subset=F+FC+["flood_risk"]).copy(); frr["flood_risk"]=frr["flood_risk"].astype(int)
SETS={"Réanalyse seule":F,"Prévision seule":FC,"Réanalyse + Prévision":F+FC}; skf=StratifiedKFold(5,shuffle=True,random_state=42); comp={}
for name,feats in SETS.items():
    X=frr[feats].to_numpy(float); y=frr["flood_risk"].to_numpy(int); a,p=[],[]
    for tr,te in skf.split(X,y):
        m=rf_factory(10,10)().fit(X[tr],y[tr]); pb=m.predict_proba(X[te])[:,1]; a.append(roc_auc_score(y[te],pb)); p.append(average_precision_score(y[te],pb))
    comp[name]=(np.mean(a),np.mean(p))
cdf=pd.DataFrame(comp,index=["ROC-AUC","PR-AUC"]).T
fig,ax=plt.subplots(figsize=(8,4.4)); cdf.plot.bar(ax=ax,color=["#4C72B0","#C44E52"],rot=8)
for c in ax.containers: ax.bar_label(c,fmt="%.3f",fontsize=9)
ax.set(title="Apport de la prévision météorologique (validation croisée 5 plis, 2024-2025)",ylabel="Score",ylim=(0,1)); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(FIG/"fig5_apport_prevision_fr.png",bbox_inches="tight"); plt.close(); print("+ fig5_apport_prevision_fr.png")

# --- Fusion (fig_fusion_fr) ---
r_f,f_f,p_f,r_c,pi=0.930,0.527,0.298,0.955,0.194
def system(fc):
    tp=pi*r_f*r_c; fp=(1-pi)*f_f*fc; return tp/(tp+fp) if tp+fp>0 else 1.0
xx=np.linspace(0,0.12,60); yy=[system(fc) for fc in xx]
fig,ax=plt.subplots(figsize=(7,4.4))
ax.plot(xx*100,yy,lw=2.5,color="#2ecc71",label="Système à deux étages (fusion)")
ax.axhline(p_f,ls="--",color="#C44E52",lw=2,label=f"Étage de prévision seul ({p_f:.2f})")
ax.scatter([0],[system(0.0)],color="#2ecc71",zorder=5); ax.annotate("CNN mesuré\n(TFP=0)",(0.2,0.96),fontsize=9)
ax.set(xlabel="Taux de faux positifs de l'étage visuel (%)",ylabel="Précision de l'alerte système",
       title="La fusion relève la précision de l'alerte au-dessus de l'étage de prévision",ylim=(0,1.05))
ax.legend(loc="lower left"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG/"fig_fusion_fr.png",dpi=150,bbox_inches="tight"); plt.close(); print("+ fig_fusion_fr.png")
print("Terminé.")
