"""Entraînement du JEPA — device-agnostic (mps/cuda/cpu).

  python -m jepa.train --overfit          # sanity : loss chute, pas de collapse
  python -m jepa.train --subset 50000     # run sur sous-ensemble
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import make_datasets, collate_fn, TrajDataset, load_cache
from .model import JEPA, sample_target_mask

CKPT_DIR = "checkpoints"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_batch(batch, device):
    x, pad_mask, lengths = batch
    return x.to(device), pad_mask.to(device), lengths


def run_epoch(model, loader, opt, cfg, train=True):
    model.train(train)
    agg = {"loss": 0.0, "inv": 0.0, "var": 0.0, "cov": 0.0, "emb_std": 0.0}
    nb = 0
    for batch in loader:
        x, pad_mask, lengths = move_batch(batch, cfg.device)
        tmask = sample_target_mask(lengths.tolist(), x.size(1), cfg).to(cfg.device)
        with torch.set_grad_enabled(train):
            loss, logs = model(x, pad_mask, tmask)
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        for k in agg:
            agg[k] += logs[k]
        nb += 1
    return {k: v / max(nb, 1) for k, v in agg.items()}


def train(cfg, overfit=False):
    set_seed(cfg.seed)
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"Device : {cfg.device}")

    if overfit:
        cache = load_cache(cfg.data_path)
        from .features import Normalizer
        norm = (Normalizer.load(cfg.feat_stats_path)
                if os.path.exists(cfg.feat_stats_path)
                else Normalizer.fit(cache["points"], cache["offsets"]))
        ds = TrajDataset(cache, norm, cfg.max_len, np.arange(8))
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)
        val_loader = None
        cfg.epochs = 200
    else:
        train_ds, val_ds, _, _ = make_datasets(cfg)
        loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=cfg.num_workers)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=cfg.num_workers)
        print(f"Train: {len(train_ds):,} trajets | Val: {len(val_ds):,}")

    model = JEPA(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres : {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    history = []
    for ep in range(cfg.epochs):
        tr = run_epoch(model, loader, opt, cfg, train=True)
        line = (f"ep {ep+1:3d}/{cfg.epochs}  loss={tr['loss']:.3f}  "
                f"inv={tr['inv']:.4f}  var={tr['var']:.3f}  cov={tr['cov']:.3f}  "
                f"emb_std={tr['emb_std']:.3f}")
        if val_loader is not None and (ep + 1) % 2 == 0:
            va = run_epoch(model, val_loader, opt, cfg, train=False)
            line += f"  | val_loss={va['loss']:.3f} val_inv={va['inv']:.4f}"
        if overfit and (ep + 1) % 20 != 0 and ep != 0:
            pass  # log allégé en overfit
        else:
            print(line, flush=True)
        history.append(tr)

    # Courbe d'apprentissage
    _plot_history(history, overfit)

    if not overfit:
        ckpt = os.path.join(CKPT_DIR, "jepa.pt")
        torch.save({"model": model.state_dict(), "cfg": cfg.to_dict()}, ckpt)
        print(f"Checkpoint : {ckpt}")

    # Bilan sanity (overfit)
    if overfit:
        first, last = history[0], history[-1]
        print("\n--- Sanity overfit ---")
        print(f"inv     : {first['inv']:.4f} -> {last['inv']:.4f} "
              f"({'OK descend' if last['inv'] < first['inv'] else 'PROBLEME'})")
        print(f"emb_std : {last['emb_std']:.3f} "
              f"({'OK > 0 (pas de collapse)' if last['emb_std'] > 0.1 else 'RISQUE collapse'})")
    return model, history


def _plot_history(history, overfit):
    ep = range(1, len(history) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["inv"] for h in history], label="inv (prédiction)")
    ax[0].plot(ep, [h["var"] for h in history], label="var")
    ax[0].plot(ep, [h["cov"] for h in history], label="cov")
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[0].set_title("Termes de perte")
    ax[1].plot(ep, [h["emb_std"] for h in history], color="#e76f51")
    ax[1].axhline(0, color="gray", lw=0.5)
    ax[1].set_xlabel("epoch"); ax[1].set_title("Écart-type des embeddings (anti-collapse)")
    fig.tight_layout()
    os.makedirs(CKPT_DIR, exist_ok=True)
    out = os.path.join(CKPT_DIR, "overfit_curve.png" if overfit else "train_curve.png")
    fig.savefig(out, dpi=110)
    print(f"Courbe : {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--data-path", default=None,
                    help="cache trips.npz (ex. chemin Drive sur Colab)")
    ap.add_argument("--feat-stats", default=None, help="cache feat_stats.npz")
    args = ap.parse_args()

    cfg = Config()
    if args.subset is not None:
        cfg.subset_size = args.subset
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.data_path is not None:
        cfg.data_path = args.data_path
    if args.feat_stats is not None:
        cfg.feat_stats_path = args.feat_stats
    train(cfg, overfit=args.overfit)
