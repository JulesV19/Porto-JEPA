"""Analyse exploratoire (EDA) du dataset Porto Taxi — une seule passe streaming.

On parcourt train.csv ligne par ligne (jamais tout en RAM) et on accumule des
agrégats bornés couvrant 4 axes :

  1. Couverture spatiale & densité   (bounding box + grille de densité)
  2. Géométrie des trajectoires        (nb points, durée, longueur, régularité)
  3. Qualité des données               (missing, vides, doublons, sauts GPS)
  4. Temporel & métadonnées            (heure/jour/mois, call type, stands, taxis)

Les résultats sont mis en cache dans eda_cache.npz + eda_cache.json pour que le
notebook puisse re-tracer sans re-scanner les 2 Go.

Usage :
    python eda.py            # passe complète sur tout le fichier
    python eda.py 50000      # limite aux 50 000 premières lignes (test rapide)
"""

import csv
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "train.csv")
CACHE_NPZ = os.path.join(BASE_DIR, "eda_cache.npz")
CACHE_JSON = os.path.join(BASE_DIR, "eda_cache.json")

SAMPLE_INTERVAL_S = 15          # intervalle d'échantillonnage GPS (donnée fixe)
MIN_POINTS = 2                  # seuil de validité d'une trajectoire

# Grille de densité spatiale : région de Porto élargie.
GRID_LON = (-8.75, -8.45)
GRID_LAT = (41.05, 41.25)
GRID_N = 300                    # 300x300 cellules

# Seuil de "saut GPS" : vitesse implicite entre 2 points > 150 km/h est
# physiquement improbable pour un taxi urbain (15 s ⇒ >625 m par pas).
JUMP_SPEED_KMH = 150.0

# --- Bins des histogrammes de dynamique (Axe 5) ---
SPEED_EDGES = np.arange(0, 162, 2.0)      # vitesse instantanée, km/h (0..160)
ACC_EDGES = np.arange(-40, 41, 1.0)       # variation de vitesse, km/h par pas 15 s
ANGLE_EDGES = np.arange(0, 185, 5.0)      # amplitude de virage, degrés (0..180)

# --- Bins pour la caractérisation des sauts GPS (Axe 6) ---
JUMPDIST_EDGES = np.arange(0, 52, 1.0)    # longueur d'un segment-saut, km
JUMPPOS_EDGES = np.arange(0, 1.0001, 0.05)  # position relative du saut dans le trajet

STATIONARY_KM = 0.005                     # segment < 5 m : véhicule ~à l'arrêt

csv.field_size_limit(10 * 1024 * 1024)


def haversine_km(lon1, lat1, lon2, lat2):
    """Distances haversine (km) entre suites de points, vectorisé numpy."""
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def run_analysis(csv_path=CSV_PATH, limit=None, log_every=100_000):
    t0 = time.time()

    # --- Compteurs de qualité (sur toutes les lignes) ---
    n_rows = 0
    n_missing_flag = 0          # MISSING_DATA == True
    n_parse_error = 0           # POLYLINE non parsable
    n_empty = 0                 # 0 point
    n_too_short = 0             # 1 point (< MIN_POINTS)
    n_valid = 0                 # trajectoires retenues (>= 2 points)

    # --- Agrégats par trajectoire valide (arrays 1D, ~1.7M floats : OK RAM) ---
    n_points_list = []          # nb de points
    length_km_list = []         # longueur parcourue (km)
    max_seg_km_list = []        # plus long segment (km) -> détecte les sauts
    dup_frac_list = []          # fraction de points dupliqués consécutifs
    n_jumps_list = []           # nb de segments > seuil de vitesse

    # --- Dynamique (Axe 5) : histogrammes accumulés (bornés en RAM) ---
    speed_hist = np.zeros(len(SPEED_EDGES) - 1, dtype=np.int64)
    acc_hist = np.zeros(len(ACC_EDGES) - 1, dtype=np.int64)
    angle_hist = np.zeros(len(ANGLE_EDGES) - 1, dtype=np.int64)
    n_seg_total = 0             # nb total de segments
    n_seg_over = 0             # segments > SPEED_EDGES max (hors échelle vitesse)
    n_seg_stationary = 0       # segments < STATIONARY_KM (véhicule à l'arrêt)
    med_speed_list = []        # vitesse médiane par trajectoire (segments réalistes)

    # --- Caractérisation des sauts GPS (Axe 6) ---
    jumpdist_hist = np.zeros(len(JUMPDIST_EDGES) - 1, dtype=np.int64)
    jumppos_hist = np.zeros(len(JUMPPOS_EDGES) - 1, dtype=np.int64)
    n_trips_jump = 0           # trajectoires avec >= 1 saut
    n_trips_spike_only = 0     # sauts = uniquement des pics isolés (aller-retour)
    n_trips_discont = 0        # >= 1 saut non expliqué par un pic (discontinuité)
    n_spike_points = 0         # nb total de points aberrants isolés
    n_jump_seg_total = 0       # nb total de segments-sauts
    n_jump_unexplained = 0     # segments-sauts sans pic associé (téléportations)

    # --- Grille de densité spatiale ---
    grid = np.zeros((GRID_N, GRID_N), dtype=np.int64)

    # --- Bounding box réel (sur points plausibles) ---
    bbox = {"lon_min": math.inf, "lon_max": -math.inf,
            "lat_min": math.inf, "lat_max": -math.inf}

    # --- Temporel & métadonnées ---
    hour_c = Counter()
    dow_c = Counter()           # 0 = lundi
    month_c = Counter()
    calltype_c = Counter()
    stand_c = Counter()
    taxi_c = Counter()

    lon_lo, lon_hi = GRID_LON
    lat_lo, lat_hi = GRID_LAT
    lon_span = lon_hi - lon_lo
    lat_span = lat_hi - lat_lo
    seg_max_m = JUMP_SPEED_KMH * 1000 / 3600 * SAMPLE_INTERVAL_S / 1000  # km/pas

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            if limit and n_rows > limit:
                n_rows -= 1
                break
            if log_every and n_rows % log_every == 0:
                print(f"  {n_rows:,} lignes... ({time.time()-t0:.0f}s)", flush=True)

            # Métadonnées (indépendantes de la validité de la polyline)
            calltype_c[row.get("CALL_TYPE", "")] += 1
            taxi_c[row.get("TAXI_ID", "")] += 1
            ts = row.get("TIMESTAMP", "")
            if ts.isdigit():
                lt = time.localtime(int(ts))
                hour_c[lt.tm_hour] += 1
                dow_c[(lt.tm_wday)] += 1
                month_c[lt.tm_mon] += 1

            if row.get("MISSING_DATA") == "True":
                n_missing_flag += 1

            raw = row.get("POLYLINE", "")
            try:
                pts = json.loads(raw)
            except (ValueError, TypeError):
                n_parse_error += 1
                continue
            if not isinstance(pts, list):
                n_parse_error += 1
                continue
            if len(pts) == 0:
                n_empty += 1
                continue
            if len(pts) < MIN_POINTS:
                n_too_short += 1
                continue

            # --- Trajectoire valide ---
            n_valid += 1
            arr = np.asarray(pts, dtype=np.float64)  # (N, 2) : [lon, lat]
            lon = arr[:, 0]
            lat = arr[:, 1]

            n_points_list.append(len(pts))

            # ORIGIN_STAND (seulement pour trajectoires valides)
            stand = row.get("ORIGIN_STAND", "")
            if stand:
                stand_c[stand] += 1

            # Distances segment par segment
            seg_km = haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:])
            length_km_list.append(float(seg_km.sum()))
            max_seg_km_list.append(float(seg_km.max()) if seg_km.size else 0.0)
            seg_jump = seg_km > seg_max_m          # segments-sauts (booléen)
            n_jumps_list.append(int(seg_jump.sum()))

            # Doublons consécutifs (point identique au précédent)
            same = np.all(arr[1:] == arr[:-1], axis=1)
            dup_frac_list.append(float(same.mean()) if same.size else 0.0)

            # --- Axe 5 : dynamique ---
            # Vitesse instantanée par segment (km/h) : distance / 15 s.
            speed = seg_km * (3600.0 / SAMPLE_INTERVAL_S)
            n_seg_total += seg_km.size
            n_seg_over += int(np.count_nonzero(speed > SPEED_EDGES[-1]))
            n_seg_stationary += int(np.count_nonzero(seg_km < STATIONARY_KM))
            speed_hist += np.histogram(speed, bins=SPEED_EDGES)[0]

            realistic = ~seg_jump                   # on écarte les sauts GPS
            if realistic.any():
                med_speed_list.append(float(np.median(speed[realistic])))
                # Accélération : variation de vitesse entre segments réalistes voisins
                if seg_km.size >= 2:
                    acc = np.diff(speed)
                    acc_valid = realistic[:-1] & realistic[1:]
                    if acc_valid.any():
                        acc_hist += np.histogram(acc[acc_valid], bins=ACC_EDGES)[0]

            # Angles de virage entre segments consécutifs (lon corrigé par cos(lat))
            if arr.shape[0] >= 3:
                coslat = math.cos(math.radians(float(lat.mean())))
                vx = np.diff(lon) * coslat
                vy = np.diff(lat)
                dot = vx[:-1] * vx[1:] + vy[:-1] * vy[1:]
                cross = vx[:-1] * vy[1:] - vy[:-1] * vx[1:]
                ang = np.degrees(np.arctan2(np.abs(cross), dot))  # 0..180
                moving = (seg_km[:-1] > STATIONARY_KM) & (seg_km[1:] > STATIONARY_KM)
                if moving.any():
                    angle_hist += np.histogram(ang[moving], bins=ANGLE_EDGES)[0]

            # --- Axe 6 : caractérisation des sauts GPS ---
            if seg_jump.any():
                n_trips_jump += 1
                njs = int(seg_jump.sum())
                n_jump_seg_total += njs
                jumpdist_hist += np.histogram(seg_km[seg_jump],
                                              bins=JUMPDIST_EDGES)[0]
                if seg_km.size > 1:
                    pos = np.nonzero(seg_jump)[0] / (seg_km.size - 1)
                    jumppos_hist += np.histogram(pos, bins=JUMPPOS_EDGES)[0]

                # Point aberrant isolé ("pic") : point interne dont les deux
                # segments adjacents sont des sauts (aller puis retour).
                explained = np.zeros(seg_km.size, dtype=bool)
                spike_pts = 0
                if seg_km.size >= 2:
                    spike = seg_jump[:-1] & seg_jump[1:]  # point i (1..N-2)
                    spike_pts = int(spike.sum())
                    idx = np.nonzero(spike)[0]
                    explained[idx] = True          # segment entrant (i-1 -> i)
                    explained[idx + 1] = True      # segment sortant (i -> i+1)
                n_spike_points += spike_pts
                unexplained = int(np.count_nonzero(seg_jump & ~explained))
                n_jump_unexplained += unexplained
                if unexplained == 0:
                    n_trips_spike_only += 1        # sauts = pics récupérables
                else:
                    n_trips_discont += 1           # >= 1 téléportation franche

            # Bounding box (points dans une plage plausible pour éviter les
            # coordonnées 0,0 aberrantes)
            plausible = (lon > -9.5) & (lon < -8.0) & (lat > 40.5) & (lat < 41.8)
            if plausible.any():
                bbox["lon_min"] = min(bbox["lon_min"], float(lon[plausible].min()))
                bbox["lon_max"] = max(bbox["lon_max"], float(lon[plausible].max()))
                bbox["lat_min"] = min(bbox["lat_min"], float(lat[plausible].min()))
                bbox["lat_max"] = max(bbox["lat_max"], float(lat[plausible].max()))

            # Grille de densité (points dans la fenêtre GRID_*)
            ix = ((lon - lon_lo) / lon_span * GRID_N).astype(np.int64)
            iy = ((lat - lat_lo) / lat_span * GRID_N).astype(np.int64)
            inb = (ix >= 0) & (ix < GRID_N) & (iy >= 0) & (iy < GRID_N)
            if inb.any():
                np.add.at(grid, (iy[inb], ix[inb]), 1)

    elapsed = time.time() - t0
    print(f"Termine : {n_rows:,} lignes en {elapsed:.0f}s "
          f"({n_valid:,} trajectoires valides).", flush=True)

    # Conversion en arrays
    n_points = np.asarray(n_points_list, dtype=np.int32)
    length_km = np.asarray(length_km_list, dtype=np.float64)
    max_seg_km = np.asarray(max_seg_km_list, dtype=np.float64)
    dup_frac = np.asarray(dup_frac_list, dtype=np.float64)
    n_jumps = np.asarray(n_jumps_list, dtype=np.int32)
    duration_min = (n_points - 1) * SAMPLE_INTERVAL_S / 60.0

    # Distribution du nb de trajectoires par taxi
    trips_per_taxi = np.asarray(list(taxi_c.values()), dtype=np.int32)
    med_speed = np.asarray(med_speed_list, dtype=np.float64)

    # --- Sauvegarde ---
    np.savez_compressed(
        CACHE_NPZ,
        n_points=n_points,
        length_km=length_km,
        max_seg_km=max_seg_km,
        dup_frac=dup_frac,
        n_jumps=n_jumps,
        duration_min=duration_min,
        grid=grid,
        trips_per_taxi=trips_per_taxi,
        # Axe 5 : dynamique
        speed_hist=speed_hist, speed_edges=SPEED_EDGES,
        acc_hist=acc_hist, acc_edges=ACC_EDGES,
        angle_hist=angle_hist, angle_edges=ANGLE_EDGES,
        med_speed=med_speed,
        # Axe 6 : sauts GPS
        jumpdist_hist=jumpdist_hist, jumpdist_edges=JUMPDIST_EDGES,
        jumppos_hist=jumppos_hist, jumppos_edges=JUMPPOS_EDGES,
    )

    def counter_to_dict(c):
        return {str(k): int(v) for k, v in c.items()}

    summary = {
        "elapsed_s": round(elapsed, 1),
        "n_rows": n_rows,
        "n_missing_flag": n_missing_flag,
        "n_parse_error": n_parse_error,
        "n_empty": n_empty,
        "n_too_short": n_too_short,
        "n_valid": n_valid,
        "n_taxis": len(taxi_c),
        "bbox": {k: (None if math.isinf(v) else round(v, 6))
                 for k, v in bbox.items()},
        "grid_extent": {"lon": list(GRID_LON), "lat": list(GRID_LAT),
                        "n": GRID_N},
        "jump_speed_kmh": JUMP_SPEED_KMH,
        # Axe 5 : dynamique
        "n_seg_total": n_seg_total,
        "n_seg_over": n_seg_over,
        "n_seg_stationary": n_seg_stationary,
        "speed_edges_max": float(SPEED_EDGES[-1]),
        # Axe 6 : sauts GPS
        "n_trips_jump": n_trips_jump,
        "n_trips_spike_only": n_trips_spike_only,
        "n_trips_discont": n_trips_discont,
        "n_spike_points": n_spike_points,
        "n_jump_seg_total": n_jump_seg_total,
        "n_jump_unexplained": n_jump_unexplained,
        "hour": counter_to_dict(hour_c),
        "dow": counter_to_dict(dow_c),
        "month": counter_to_dict(month_c),
        "call_type": counter_to_dict(calltype_c),
        "top_stands": dict(stand_c.most_common(20)),
    }
    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Cache écrit : {CACHE_NPZ} + {CACHE_JSON}", flush=True)
    return summary


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"train.csv introuvable à {CSV_PATH}")
    run_analysis(limit=limit)
