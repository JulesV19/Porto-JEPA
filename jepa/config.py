"""Configuration partagée du JEPA (données, modèle, entraînement).

Tous les hyperparamètres « knobs » du plan sont ici. Les valeurs par défaut
visent un modèle compact tenant sur MPS ; on passe à l'échelle (Colab) en
changeant surtout `d_model`, `batch_size` et `subset_size`.
"""

from dataclasses import dataclass, field, asdict

import torch


def pick_device():
    """mps → cuda → cpu (device-agnostic)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Config:
    # --- Données ---
    data_path: str = "data/trips.npz"
    feat_stats_path: str = "data/feat_stats.npz"
    max_len: int = 256           # cap de longueur (p99 des trajets = 191)
    n_features: int = 6          # x, y, speed, sin θ, cos θ, Δθ  (cf. features.py)
    subset_size: int | None = None   # None = tout le cache ; sinon N trajets
    val_frac: float = 0.1

    # --- Modèle : encodeur Transformer ---
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dim_ff: int = 256
    dropout: float = 0.1

    # --- Predictor ---
    pred_layers: int = 2
    pred_heads: int = 4

    # --- Masquage I-JEPA ---
    n_target_blocks: int = 3     # nb de blocs cibles échantillonnés
    target_frac: tuple = (0.10, 0.25)   # taille d'un bloc cible (frac. de la seq)
    min_context: int = 4         # nb min de tokens de contexte

    # --- Pertes VICReg ---
    lambda_inv: float = 25.0     # invariance (prédiction MSE en latent)
    lambda_var: float = 25.0     # variance (anti-collapse)
    lambda_cov: float = 1.0      # covariance (décorrélation)
    vic_gamma: float = 1.0       # cible d'écart-type pour le terme variance

    # --- Entraînement ---
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    num_workers: int = 0         # 0 = sûr sur macOS/MPS
    grad_clip: float = 1.0
    seed: int = 0

    device: str = field(default_factory=pick_device)

    def to_dict(self):
        return asdict(self)
