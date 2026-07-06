# Porto Taxi — exploration des données

Projet : à terme, un modèle **JEPA** pour l'apprentissage de représentations de
trajectoires. Étape actuelle : **exploration du dataset** Porto Taxi
(ECML/PKDD 2015) pour caractériser les données avant toute décision de modèle.

## Données

`train.csv` (~2 Go) placé à la racine du projet. Colonnes principales :
`TRIP_ID, TAXI_ID, TIMESTAMP, CALL_TYPE, ORIGIN_STAND, POLYLINE`. La colonne
`POLYLINE` est une liste ordonnée de points `[longitude, latitude]` échantillonnés
toutes les 15 secondes.

## Prérequis

- Python 3.10+
- `train.csv` à la racine (à côté de `eda.py`)

## Exploration (EDA)

Analyse sur **l'intégralité** du dataset (~1,71M lignes), en une seule passe
streaming (le fichier n'est jamais chargé entièrement en RAM).

```bash
pip install numpy matplotlib nbconvert
python eda.py            # passe complète (~90 s) → eda_cache.npz + eda_cache.json
python build_notebook.py # génère exploration.ipynb à partir du cache
```

- [eda.py](eda.py) — parcourt `train.csv` ligne par ligne et met en cache les
  agrégats de 4 axes : couverture spatiale & densité, géométrie des
  trajectoires, qualité des données, temporel & métadonnées.
  `python eda.py 50000` limite à un échantillon pour un test rapide.
- [exploration.ipynb](exploration.ipynb) — charge le cache et rend les
  graphiques + observations. Ne relit pas les 2 Go.

Repères sur le run complet : **1 674 160 trajectoires valides** (rétention
97,9 %), 448 taxis, couverture sur 12 mois ; durée médiane ≈ 10 min, distance
médiane ≈ 4 km ; ~10 % des trajectoires contiennent au moins un saut GPS.

### Réglages EDA

Dans [eda.py](eda.py) :

- `MIN_POINTS` — seuil minimal de points par trajectoire (défaut : 2)
- `GRID_LON` / `GRID_LAT` / `GRID_N` — fenêtre et résolution de la grille de densité
- `JUMP_SPEED_KMH` — seuil de vitesse implicite au-delà duquel un segment est
  compté comme saut GPS (défaut : 150 km/h)

## JEPA — apprentissage de représentations

Modèle **JEPA** (Joint Embedding Predictive Architecture) auto-supervisé,
inspiré d'I-JEPA (masquage par blocs) avec anti-collapse **VICReg** (pas d'EMA).
Le code est **device-agnostic** (`mps → cuda → cpu`).

### Pipeline

```bash
pip install torch scikit-learn numpy matplotlib

# 1) Prétraitement : train.csv -> data/trips.npz (streaming, interpolation/rejet)
python prepare_data.py --limit 200000      # ou sans --limit pour tout

# 2) Sanity check local (rapide) : la loss descend, pas de collapse
python -m jepa.train --overfit

# 3) Entraînement (local ou Colab). Les vrais runs se font sur GPU (voir Colab).
python -m jepa.train --subset 50000 --epochs 20

# 4) Évaluation : 4 protocoles + figures dans eval_out/
python -m jepa.eval --ckpt checkpoints/jepa.pt
```

### Modules

- [prepare_data.py](prepare_data.py) — passe streaming ; **interpole** un point
  manquant isolé, **rejette** les trajets aberrants (trou > 1 point, trop de
  sauts). Écrit un cache CSR `data/trips.npz`.
- [jepa/features.py](jepa/features.py) — 6 features/point : coords locales
  (x, y km relatifs au départ) + cinématique (vitesse, sin/cos cap, virage).
- [jepa/data.py](jepa/data.py) — Dataset variable-length + collate (padding +
  masque d'attention).
- [jepa/model.py](jepa/model.py) — encodeur Transformer, masquage I-JEPA,
  predictor, pertes VICReg (invariance + variance + covariance).
- [jepa/train.py](jepa/train.py) — boucle device-agnostic, mode `--overfit`.
- [jepa/eval.py](jepa/eval.py) — embeddings gelés → sonde linéaire
  (CALL_TYPE / heure / jour), régression durée & distance, clustering + carte,
  prédiction du point suivant.
- [jepa/config.py](jepa/config.py) — **tous** les hyperparamètres (dims, blocs
  cibles, poids VICReg, `max_len`, sous-ensemble…).

### Entraînement sur GPU (Colab, via git clone + Drive)

Les vrais runs se font sur GPU (sur Apple Silicon / MPS l'attention masquée est
lente). Le repo ne contient que le **code** ; les données passent par **Google
Drive**. Workflow :

1. En local : `python prepare_data.py --limit 200000` (rapide).
2. Déposer `data/trips.npz` + `data/feat_stats.npz` dans `MyDrive/jepa-taxi/data/`.
3. Sur Colab : ouvrir [colab_train.ipynb](colab_train.ipynb) — il `git clone` le
   repo, monte le Drive, puis entraîne/évalue sur GPU. `train.csv` (2 Go) n'est
   **pas** requis sur Colab.

Les chemins de données sont surchargeables en CLI (`--data-path`, `--feat-stats`)
pour pointer vers le Drive. Régénérer le notebook : `python build_colab.py`
(éditer `REPO_URL` dedans).
