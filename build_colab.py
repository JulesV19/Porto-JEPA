"""Génère colab_train.ipynb : notebook prêt à l'emploi pour entraîner le JEPA
sur GPU Colab (CUDA), via **git clone** du repo + **Google Drive** pour le cache.

Non exécuté ici — il tourne sur Colab.

Workflow :
  - le repo GitHub ne contient que le CODE (pas de données) ;
  - le cache prétraité (data/trips.npz + feat_stats.npz) est généré en local
    (prepare_data.py) puis déposé sur Google Drive ;
  - sur Colab : on clone le code, on monte le Drive, on entraîne sur GPU.
"""

import nbformat as nbf

REPO_URL = "https://github.com/JulesV19/Porto-JEPA.git"
REPO_DIR = "Porto-JEPA"
# Le cache est cherché automatiquement dans le Drive (voir cellule 3) —
# pas besoin de connaître le nom exact du dossier.

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(f"""# JEPA trajectoires Porto — entraînement GPU (Colab)

Code cloné depuis [github.com/JulesV19/Porto-JEPA](https://github.com/JulesV19/Porto-JEPA),
données lues depuis Google Drive. Le code est **device-agnostic** : le GPU CUDA
de Colab est détecté automatiquement.

**Prérequis (une fois, en local puis sur Drive) :**
1. Générer le cache : `python prepare_data.py --limit 200000`
2. Déposer `trips.npz` et `feat_stats.npz` n'importe où dans ton Google Drive
   (la cellule 3 les retrouve automatiquement).

**Runtime :** *Exécution → Modifier le type d'exécution → GPU*.""")

code("!nvidia-smi -L\n"
     "import torch\n"
     "print('CUDA:', torch.cuda.is_available(),\n"
     "      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')")

code(f"""# 1) Cloner le code depuis GitHub
REPO_URL = "{REPO_URL}"   # <-- mets l'URL de TON repo
!git clone $REPO_URL {REPO_DIR}
%cd {REPO_DIR}
!git log --oneline -1""")

code("# 2) Dépendances (torch déjà présent sur Colab)\n"
     "!pip -q install scikit-learn matplotlib numpy")

code("""# 3) Monter le Drive et localiser automatiquement le cache
from google.colab import drive
import glob, os
drive.mount('/content/drive')

hits = glob.glob('/content/drive/MyDrive/**/trips.npz', recursive=True)
assert hits, "trips.npz introuvable dans le Drive — dépose-le puis relance."
DATA = os.path.dirname(hits[0])
STATS = os.path.join(DATA, 'feat_stats.npz')
assert os.path.exists(STATS), f"feat_stats.npz manquant à côté de trips.npz ({DATA})"
print('Cache trouvé dans :', DATA)""")

code("""# 4) Sanity overfit (rapide) — vérifie que ça apprend sans collapse
!python -m jepa.train --overfit --data-path {DATA}/trips.npz --feat-stats {STATS}""")

code("""# 5) Entraînement sur GPU (ajuste subset/epochs/batch selon le GPU)
!python -m jepa.train --subset 200000 --epochs 20 --batch-size 256 \\
    --data-path {DATA}/trips.npz --feat-stats {STATS}""")

code("""# 6) Évaluation (4 protocoles) + figures
!python -m jepa.eval --ckpt checkpoints/jepa.pt --n-eval 8000 \\
    --data-path {DATA}/trips.npz --feat-stats {STATS}
from IPython.display import Image
Image('eval_out/clusters_map.png')""")

code(f"""# 7) Sauvegarder le checkpoint sur le Drive (persistant entre sessions)
import shutil, os
os.makedirs('/content/drive/MyDrive/jepa-taxi/checkpoints', exist_ok=True)
shutil.copy('checkpoints/jepa.pt', '/content/drive/MyDrive/jepa-taxi/checkpoints/')
print('checkpoint copié sur le Drive')""")

md("""### Notes
- Tous les hyperparamètres sont dans `jepa/config.py` ; sur GPU on peut monter
  `d_model`, `n_layers`, `batch_size`, `subset_size` (voire tout le dataset).
- Pour re-tirer les dernières modifs du code : `!git pull` dans le dossier cloné.
- `train.csv` (2 Go) n'est **pas** nécessaire sur Colab.""")

nb["cells"] = cells
with open("colab_train.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("colab_train.ipynb écrit :", len(cells), "cellules")
