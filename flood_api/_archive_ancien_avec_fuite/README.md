# Archive : ancien notebook (fuite de cible)

`model.ipynb` est la version initiale du modèle de prévision. Elle souffrait
d'une **fuite de cible** : la variable à prédire (`flood_risk`) était définie à
partir des mêmes variables de pluie fournies en entrée, d'où un AUC = 1,000
tautologique. Conservée à titre historique.

La version corrigée (vraie prévision J+1 à J+3, sans fuite) se trouve dans
`flood_api/flood_forecast.ipynb` et les scripts `rf_forecast.py` / `rf_forecast_v2.py`.
