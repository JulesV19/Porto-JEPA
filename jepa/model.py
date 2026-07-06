"""Modèle JEPA pour trajectoires : encodeur Transformer + predictor + VICReg.

Schéma (façon I-JEPA, adapté au 1D séquentiel, anti-collapse VICReg sans EMA) :

  x (B,L,F) --projection+posenc--> tokens
  masquage  : on tire des blocs CIBLES contigus ; le reste = CONTEXTE.

  context_rep = Enc(tokens | attention limitée au contexte)     [gradient]
  target_rep  = stopgrad( Enc(tokens | séquence complète) )     [pas de gradient]
  pred_rep    = Predictor(context_rep + mask-tokens aux positions cibles)

  perte = λ_inv · MSE(pred_rep, target_rep)          (invariance / prédiction)
        + λ_var · variance-hinge(context_rep)         (anti-collapse)
        + λ_cov · covariance(context_rep)             (décorrélation)

L'encodeur cible partage les poids de l'encodeur de contexte (stop-grad sur la
cible). VICReg, appliqué aux sorties de l'encodeur, empêche le collapse — pas
besoin de moyenne mobile (EMA).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)          # (max_len, d_model)

    def forward(self, length):
        return self.pe[:length]                 # (L, d_model)


class TrajectoryEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(cfg.n_features, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout, batch_first=True, activation="gelu")
        # enable_nested_tensor=False : le fast-path nested n'est pas implémenté
        # sur MPS et casse en mode eval.
        self.enc = nn.TransformerEncoder(layer, cfg.n_layers,
                                         enable_nested_tensor=False)

    def forward(self, x, key_padding_mask):
        """x (B,L,F), key_padding_mask (B,L) True=ignoré -> reps (B,L,D)."""
        h = self.proj(x) + self.pos(x.size(1)).unsqueeze(0)
        return self.enc(h, src_key_padding_mask=key_padding_mask)


class Predictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pos = PositionalEncoding(cfg.d_model)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.pred_heads, dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout, batch_first=True, activation="gelu")
        self.net = nn.TransformerEncoder(layer, cfg.pred_layers,
                                         enable_nested_tensor=False)
        self.head = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, context_rep, target_mask, pad_mask):
        """context_rep (B,L,D) : reps du contexte (valides hors cibles/padding).
        target_mask (B,L) True=cible. Retourne pred_rep (B,L,D)."""
        b, l, d = context_rep.shape
        pe = self.pos(l).unsqueeze(0)                    # (1,L,D)
        mask_tok = self.mask_token.view(1, 1, d) + pe    # (1,L,D)
        tm = target_mask.unsqueeze(-1)                   # (B,L,1)
        # Positions cibles -> mask-token+posenc ; sinon reps de contexte.
        inp = torch.where(tm, mask_tok.expand(b, l, d), context_rep)
        out = self.net(inp, src_key_padding_mask=pad_mask)
        return self.head(out)


class JEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = TrajectoryEncoder(cfg)
        self.predictor = Predictor(cfg)

    # --- encodage utilisé aussi par l'évaluation (representation par trajet) ---
    def encode(self, x, pad_mask):
        return self.encoder(x, pad_mask)

    def embed_pooled(self, x, pad_mask):
        """Embedding par trajet = moyenne masquée des tokens (encodeur complet)."""
        rep = self.encoder(x, pad_mask)                  # (B,L,D)
        keep = (~pad_mask).unsqueeze(-1).float()
        return (rep * keep).sum(1) / keep.sum(1).clamp(min=1.0)

    def forward(self, x, pad_mask, target_mask):
        # Contexte : l'encodeur ignore padding ET cibles.
        ctx_ignore = pad_mask | target_mask
        context_rep = self.encoder(x, ctx_ignore)

        # Cible : encodeur complet (voit toute la séquence), sans gradient ni
        # dropout (cible déterministe -> prédiction stable).
        with torch.no_grad():
            was_training = self.encoder.training
            self.encoder.eval()
            target_rep = self.encoder(x, pad_mask)
            self.encoder.train(was_training)

        pred_rep = self.predictor(context_rep, target_mask, pad_mask)

        return self._loss(context_rep, target_rep, pred_rep,
                          pad_mask, target_mask)

    def _loss(self, context_rep, target_rep, pred_rep, pad_mask, target_mask):
        cfg = self.cfg

        # --- Invariance : prédiction des reps cibles (en latent) ---
        tsel = target_mask
        if tsel.any():
            preds = pred_rep[tsel]
            tgts = target_rep[tsel].detach()
            inv = F.mse_loss(preds, tgts)
        else:
            inv = pred_rep.sum() * 0.0

        # --- VICReg sur les sorties d'encodeur (positions de contexte) ---
        ctx_valid = (~pad_mask) & (~target_mask)
        z = context_rep[ctx_valid]                       # (M, D)
        var, cov = self._vicreg(z)

        total = cfg.lambda_inv * inv + cfg.lambda_var * var + cfg.lambda_cov * cov
        return total, {
            "loss": float(total.detach()),
            "inv": float(inv.detach()),
            "var": float(var.detach()),
            "cov": float(cov.detach()),
            "emb_std": float(z.detach().std(0).mean()) if z.numel() else 0.0,
        }

    def _vicreg(self, z):
        """Termes variance (anti-collapse) et covariance (décorrélation)."""
        if z.shape[0] < 2:
            zero = z.sum() * 0.0
            return zero, zero
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        var = torch.mean(F.relu(self.cfg.vic_gamma - std))
        zc = z - z.mean(dim=0)
        cov = (zc.T @ zc) / (z.shape[0] - 1)
        d = z.shape[1]
        off = cov - torch.diag(torch.diag(cov))
        cov_loss = (off ** 2).sum() / d
        return var, cov_loss


def sample_target_mask(lengths, max_len, cfg, generator=None):
    """Tire un masque cible (B, max_len) True=cible, façon blocs I-JEPA.

    Par trajet : n_target_blocks blocs contigus de taille frac. de la longueur,
    en garantissant qu'il reste du contexte (>= min_context si possible).
    """
    b = len(lengths)
    mask = torch.zeros(b, max_len, dtype=torch.bool)
    rng = np.random.default_rng(None if generator is None else generator)
    lo, hi = cfg.target_frac
    for j, n in enumerate(lengths):
        n = int(n)
        if n < 2:
            continue
        budget = max(1, n - cfg.min_context)     # tokens cibles max
        placed = 0
        for _ in range(cfg.n_target_blocks):
            if placed >= budget:
                break
            size = int(round(rng.uniform(lo, hi) * n))
            size = max(1, min(size, budget - placed))
            start = int(rng.integers(0, n - size + 1))
            mask[j, start:start + size] = True
            placed = int(mask[j, :n].sum())
        # Filet de sécurité : au moins 1 cible et 1 contexte.
        if mask[j, :n].sum() == 0:
            mask[j, n - 1] = True
        if mask[j, :n].sum() == n:
            mask[j, 0] = False
    return mask
