"""Dataset et collation pour trajectoires de longueur variable.

Lit le cache CSR produit par prepare_data.py, calcule les features à la volée
(features.compute_features), standardise, tronque à max_len, et assemble des
batches paddés avec masque d'attention.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import compute_features, Normalizer


def load_cache(path):
    """Charge trips.npz -> dict d'arrays (points, offsets, méta)."""
    d = np.load(path)
    return {k: d[k] for k in d.files}


class TrajDataset(Dataset):
    def __init__(self, cache, normalizer, max_len, indices):
        self.points = cache["points"]
        self.offsets = cache["offsets"]
        self.norm = normalizer
        self.max_len = max_len
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        seg = self.points[self.offsets[i]:self.offsets[i + 1]]
        if seg.shape[0] > self.max_len:
            seg = seg[:self.max_len]
        feats = self.norm.apply(compute_features(seg))
        return torch.from_numpy(feats), feats.shape[0]


def collate_fn(batch):
    """batch de (feats (L,F), L) -> (X (B,Lmax,F), pad_mask (B,Lmax), lengths)."""
    feats, lengths = zip(*batch)
    b = len(feats)
    lmax = max(lengths)
    fdim = feats[0].shape[1]
    x = torch.zeros(b, lmax, fdim, dtype=torch.float32)
    pad_mask = torch.ones(b, lmax, dtype=torch.bool)   # True = padding
    for j, (f, n) in enumerate(zip(feats, lengths)):
        x[j, :n] = f
        pad_mask[j, :n] = False
    return x, pad_mask, torch.tensor(lengths, dtype=torch.long)


def make_datasets(cfg):
    """Prépare (train_ds, val_ds, cache, normalizer) à partir de la config."""
    cache = load_cache(cfg.data_path)
    n_trips = len(cache["offsets"]) - 1

    # Normaliseur : chargé si présent, sinon ajusté et sauvegardé.
    if os.path.exists(cfg.feat_stats_path):
        norm = Normalizer.load(cfg.feat_stats_path)
    else:
        norm = Normalizer.fit(cache["points"], cache["offsets"])
        os.makedirs(os.path.dirname(cfg.feat_stats_path) or ".", exist_ok=True)
        norm.save(cfg.feat_stats_path)

    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(n_trips)
    if cfg.subset_size:
        idx = idx[:cfg.subset_size]

    n_val = max(1, int(len(idx) * cfg.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = TrajDataset(cache, norm, cfg.max_len, train_idx)
    val_ds = TrajDataset(cache, norm, cfg.max_len, val_idx)
    return train_ds, val_ds, cache, norm
