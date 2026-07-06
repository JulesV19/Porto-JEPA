"""Évaluation des représentations JEPA (encodeur gelé).

Quatre protocoles :
  1. Sonde linéaire (LogisticRegression) : CALL_TYPE, heure, jour.
  2. Régression (Ridge) : durée & distance du trajet.
  3. Clustering (KMeans) + carte des trajets représentatifs par cluster.
  4. Prédiction du point suivant depuis les reps par-point gelées.

  python -m jepa.eval --ckpt checkpoints/jepa.pt
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score

from .config import Config
from .data import load_cache, collate_fn
from .features import compute_features, Normalizer
from .model import JEPA
from eda import haversine_km

OUT_DIR = "eval_out"


def load_model(ckpt_path):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(**blob["cfg"])
    cfg.device = Config().device          # device courant
    model = JEPA(cfg).to(cfg.device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def extract(model, cfg, cache, indices, max_pts_nextstep=50000):
    """Retourne embeddings (n,D), méta, et paires (rep_point, delta_suivant)."""
    points, offsets, norm = cache["points"], cache["offsets"], cache["_norm"]
    E, meta = [], {"call_type": [], "hour": [], "dow": [],
                   "duration_min": [], "distance_km": []}
    npt_reps, npt_targets = [], []

    bs = 256
    for s in range(0, len(indices), bs):
        chunk = indices[s:s + bs]
        feats_list, lengths = [], []
        raw_segs = []
        for i in chunk:
            seg = points[offsets[i]:offsets[i + 1]]
            if seg.shape[0] > cfg.max_len:
                seg = seg[:cfg.max_len]
            raw_segs.append(seg)
            f = norm.apply(compute_features(seg))
            feats_list.append(torch.from_numpy(f))
            lengths.append(f.shape[0])
        x, pad_mask, _ = collate_fn(list(zip(feats_list, lengths)))
        x, pad_mask = x.to(cfg.device), pad_mask.to(cfg.device)

        emb = model.embed_pooled(x, pad_mask).cpu().numpy()
        reps = model.encode(x, pad_mask).cpu().numpy()          # (B,L,D)
        E.append(emb)

        for b, i in enumerate(chunk):
            seg = raw_segs[b]
            n = len(seg)
            meta["call_type"].append(int(cache["call_type"][i]))
            ts = int(cache["timestamp"][i])
            lt = time.localtime(ts) if ts > 0 else None
            meta["hour"].append(lt.tm_hour if lt else -1)
            meta["dow"].append(lt.tm_wday if lt else -1)
            meta["duration_min"].append((n - 1) * 15 / 60.0)
            d = haversine_km(seg[:-1, 0], seg[:-1, 1], seg[1:, 0], seg[1:, 1])
            meta["distance_km"].append(float(d.sum()))

            # Paires point-suivant (deltas en coords locales km)
            if n >= 2 and len(npt_reps) * 1 < max_pts_nextstep:
                lon0, lat0 = seg[0]
                coslat = np.cos(np.radians(lat0))
                xk = (seg[:, 0] - lon0) * 111.32 * coslat
                yk = (seg[:, 1] - lat0) * 110.57
                dx = np.diff(xk); dy = np.diff(yk)
                npt_reps.append(reps[b, :n - 1])
                npt_targets.append(np.stack([dx, dy], axis=1))

    E = np.concatenate(E, axis=0)
    meta = {k: np.asarray(v) for k, v in meta.items()}
    npt_reps = np.concatenate(npt_reps, axis=0)[:max_pts_nextstep]
    npt_targets = np.concatenate(npt_targets, axis=0)[:max_pts_nextstep]
    return E, meta, npt_reps, npt_targets


def probe_classif(E, y, name):
    ok = y >= 0
    E, y = E[ok], y[ok]
    if len(np.unique(y)) < 2:
        print(f"  [{name}] ignoré (une seule classe)")
        return
    Xtr, Xte, ytr, yte = train_test_split(E, y, test_size=0.25, random_state=0,
                                          stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(sc.transform(Xtr), ytr)
    acc = accuracy_score(yte, clf.predict(sc.transform(Xte)))
    # Base : classe majoritaire
    base = np.bincount(ytr).max() / len(ytr)
    print(f"  [{name}] accuracy={acc:.3f}  (baseline maj.={base:.3f})")


def probe_regress(E, y, name):
    Xtr, Xte, ytr, yte = train_test_split(E, y, test_size=0.25, random_state=0)
    sc = StandardScaler().fit(Xtr)
    reg = Ridge(alpha=10.0).fit(sc.transform(Xtr), ytr)
    r2 = r2_score(yte, reg.predict(sc.transform(Xte)))
    print(f"  [{name}] R²={r2:.3f}")


def next_point(reps, targets):
    Xtr, Xte, ytr, yte = train_test_split(reps, targets, test_size=0.25,
                                          random_state=0)
    sc = StandardScaler().fit(Xtr)
    reg = Ridge(alpha=10.0).fit(sc.transform(Xtr), ytr)
    pred = reg.predict(sc.transform(Xte))
    mse = np.mean((pred - yte) ** 2)
    base = np.mean((yte - ytr.mean(0)) ** 2)     # prédire le delta moyen
    print(f"  [point suivant] MSE={mse:.4f}  (baseline moy.={base:.4f}, "
          f"R²={1 - mse / base:.3f})")


def cluster_map(E, cache, indices, k=6):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(E)
    labels = km.labels_
    points, offsets = cache["points"], cache["offsets"]
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    fig, ax = plt.subplots(figsize=(9, 9))
    rng = np.random.default_rng(0)
    for c in range(k):
        members = np.where(labels == c)[0]
        pick = rng.choice(members, size=min(15, len(members)), replace=False)
        for m in pick:
            i = indices[m]
            seg = points[offsets[i]:offsets[i + 1]]
            ax.plot(seg[:, 0], seg[:, 1], color=colors[c], lw=0.6, alpha=0.6)
    ax.set_title(f"Clusters d'embeddings (k={k}) — trajets représentatifs")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_xlim(-8.75, -8.5); ax.set_ylim(41.08, 41.25)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "clusters_map.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  carte clusters : {out}")


def main(ckpt, n_eval, data_path=None, feat_stats=None):
    model, cfg = load_model(ckpt)
    if data_path:
        cfg.data_path = data_path
    if feat_stats:
        cfg.feat_stats_path = feat_stats
    print(f"Device : {cfg.device} | modèle chargé depuis {ckpt}")

    cache = load_cache(cfg.data_path)
    cache["_norm"] = Normalizer.load(cfg.feat_stats_path)
    n_trips = len(cache["offsets"]) - 1
    rng = np.random.default_rng(123)
    indices = rng.choice(n_trips, size=min(n_eval, n_trips), replace=False)
    indices.sort()

    print(f"Extraction des embeddings sur {len(indices):,} trajets...")
    E, meta, npt_reps, npt_targets = extract(model, cfg, cache, indices)
    print(f"Embeddings : {E.shape}")

    print("\n1) Sonde linéaire (métadonnées)")
    probe_classif(E, meta["call_type"], "CALL_TYPE")
    probe_classif(E, meta["hour"], "heure")
    probe_classif(E, meta["dow"], "jour")

    print("\n2) Régression durée / distance")
    probe_regress(E, meta["duration_min"], "durée (min)")
    probe_regress(E, meta["distance_km"], "distance (km)")

    print("\n3) Clustering + carte")
    cluster_map(E, cache, indices)

    print("\n4) Prédiction du point suivant")
    next_point(npt_reps, npt_targets)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/jepa.pt")
    ap.add_argument("--n-eval", type=int, default=8000)
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--feat-stats", default=None)
    args = ap.parse_args()
    main(args.ckpt, args.n_eval, args.data_path, args.feat_stats)
