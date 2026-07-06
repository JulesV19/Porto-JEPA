"""Prétraitement streaming du dataset Porto Taxi pour le JEPA.

Une passe sur train.csv (même approche que eda.py) qui :
  - ne garde que les trajectoires valides (>= MIN_POINTS, non MISSING_DATA) ;
  - **interpole** un point manquant isolé (petit trou temporel = 1 point) ;
  - **rejette** les trajectoires franchement aberrantes (trou de >1 point,
    trop de sauts, ou fraction de sauts trop élevée).

Résultat écrit dans data/trips.npz sous forme CSR (points concaténés + offsets)
pour éviter un objet Python par trajet.

Usage :
    python prepare_data.py                 # tout le fichier
    python prepare_data.py --limit 50000   # sous-ensemble (dev MPS)
"""

import argparse
import csv
import json
import os
import time

import numpy as np

from eda import CSV_PATH, SAMPLE_INTERVAL_S, JUMP_SPEED_KMH, haversine_km
from jepa.features import Normalizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MIN_POINTS = 2

# Seuils de gestion des trous (ajustables — cf. plan).
SEG_MAX_KM = JUMP_SPEED_KMH * 1000 / 3600 * SAMPLE_INTERVAL_S / 1000  # 0.625 km
MAX_JUMPS = 2            # au-delà -> trajet rejeté
MAX_JUMP_FRAC = 0.05     # fraction de segments-sauts au-delà -> rejeté

CALL_TYPE_MAP = {"A": 0, "B": 1, "C": 2}

csv.field_size_limit(10 * 1024 * 1024)


def clean_trip(arr):
    """Applique interpolation / rejet à une trajectoire (N,2) [lon,lat].

    Retourne (points_nettoyés, n_interp) ou (None, raison_rejet).
    """
    seg_km = haversine_km(arr[:-1, 0], arr[:-1, 1], arr[1:, 0], arr[1:, 1])
    seg_jump = seg_km > SEG_MAX_KM
    n_jumps = int(seg_jump.sum())

    if n_jumps == 0:
        return arr, 0

    # Rejet : trou de plus d'un point (Δt réel inconnu).
    if np.any(seg_km > 2 * SEG_MAX_KM):
        return None, "gap>1pt"
    # Rejet : trop de sauts, ou trop denses.
    if n_jumps > MAX_JUMPS:
        return None, "trop_de_sauts"
    if n_jumps / seg_km.size > MAX_JUMP_FRAC:
        return None, "frac_sauts"

    # Interpolation : insérer un point médian sur chaque segment-saut.
    # (chaque saut ici vérifie SEG_MAX_KM < seg <= 2*SEG_MAX_KM => 1 point manquant)
    out = [arr[0]]
    for i in range(seg_km.size):
        if seg_jump[i]:
            out.append((arr[i] + arr[i + 1]) / 2.0)
        out.append(arr[i + 1])
    return np.asarray(out, dtype=np.float64), n_jumps


def run(csv_path, limit, out_path, stats_path, log_every=100_000):
    t0 = time.time()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    points_chunks = []          # arrays (Ni, 2)
    lengths = []                # Ni par trajet retenu
    taxi_ids, timestamps = [], []
    call_types, origin_stands = [], []

    n_rows = n_valid = n_kept = 0
    n_interp_pts = 0
    reject = {"gap>1pt": 0, "trop_de_sauts": 0, "frac_sauts": 0}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            if limit and n_rows > limit:
                n_rows -= 1
                break
            if log_every and n_rows % log_every == 0:
                print(f"  {n_rows:,} lignes... ({time.time()-t0:.0f}s)", flush=True)

            if row.get("MISSING_DATA") == "True":
                continue
            try:
                pts = json.loads(row.get("POLYLINE", ""))
            except (ValueError, TypeError):
                continue
            if not isinstance(pts, list) or len(pts) < MIN_POINTS:
                continue
            n_valid += 1

            arr = np.asarray(pts, dtype=np.float64)
            cleaned, info = clean_trip(arr)
            if cleaned is None:
                reject[info] += 1
                continue

            n_interp_pts += info
            n_kept += 1
            points_chunks.append(cleaned.astype(np.float64))
            lengths.append(len(cleaned))
            taxi_ids.append(int(row.get("TAXI_ID", 0) or 0))
            ts = row.get("TIMESTAMP", "")
            timestamps.append(int(ts) if ts.isdigit() else 0)
            call_types.append(CALL_TYPE_MAP.get(row.get("CALL_TYPE", ""), -1))
            stand = row.get("ORIGIN_STAND", "")
            origin_stands.append(int(stand) if stand.isdigit() else -1)

    # Assemblage CSR
    lengths = np.asarray(lengths, dtype=np.int64)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    points = (np.concatenate(points_chunks, axis=0) if points_chunks
              else np.zeros((0, 2), dtype=np.float64))

    np.savez_compressed(
        out_path,
        points=points,
        offsets=offsets,
        taxi_id=np.asarray(taxi_ids, dtype=np.int64),
        timestamp=np.asarray(timestamps, dtype=np.int64),
        call_type=np.asarray(call_types, dtype=np.int8),
        origin_stand=np.asarray(origin_stands, dtype=np.int16),
    )

    # Statistiques de features (mean/std) -> feat_stats.npz, prêtes pour Colab.
    if points.shape[0] > 0:
        norm = Normalizer.fit(points, offsets)
        norm.save(stats_path)

    elapsed = time.time() - t0
    summary = {
        "elapsed_s": round(elapsed, 1),
        "n_rows": n_rows,
        "n_valid": n_valid,
        "n_kept": n_kept,
        "n_interpolated_points": n_interp_pts,
        "n_rejected": int(sum(reject.values())),
        "reject_reasons": reject,
        "total_points": int(points.shape[0]),
        "out": out_path,
        "stats_out": stats_path,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="nombre de lignes CSV à lire (défaut : tout)")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "trips.npz"))
    ap.add_argument("--stats-out", default=os.path.join(DATA_DIR, "feat_stats.npz"))
    args = ap.parse_args()
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"train.csv introuvable à {CSV_PATH}")
    run(CSV_PATH, args.limit, args.out, args.stats_out)
