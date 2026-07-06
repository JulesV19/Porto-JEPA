"""Features par point : coords locales (relatives au départ) + cinématique.

Vecteur de 6 dimensions par point :
    [x, y, speed, sin θ, cos θ, Δθ]
  - x, y     : position en km relative au 1er point du trajet (cadre local métrique)
  - speed    : vitesse instantanée (km/h) du segment arrivant sur le point
  - sin/cos θ: direction du déplacement (cap)
  - Δθ       : virage (variation de cap) par rapport au point précédent

Le 1er point a une cinématique nulle (pas de segment entrant).
La direction est mise à zéro sur les points ~immobiles (cap mal défini).
"""

import numpy as np

N_FEATURES = 6

# Conversion degrés -> km autour de Porto (lat ~41°).
KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON = 111.32          # à multiplier par cos(lat0)
STATIONARY_KM = 0.005            # < 5 m : cap non défini


def compute_features(pts):
    """pts : (N,2) [lon,lat] -> features (N, 6) float32."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    lon0, lat0 = pts[0]
    coslat = np.cos(np.radians(lat0))

    x = (pts[:, 0] - lon0) * KM_PER_DEG_LON * coslat
    y = (pts[:, 1] - lat0) * KM_PER_DEG_LAT

    feats = np.zeros((n, N_FEATURES), dtype=np.float64)
    feats[:, 0] = x
    feats[:, 1] = y
    if n < 2:
        return feats.astype(np.float32)

    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx * dx + dy * dy)                 # km par segment (N-1,)
    speed = dist * (3600.0 / 15.0)                    # km/h
    theta = np.arctan2(dy, dx)                        # cap du segment
    moving = dist > STATIONARY_KM
    sin_t = np.where(moving, np.sin(theta), 0.0)
    cos_t = np.where(moving, np.cos(theta), 0.0)

    # Alignés sur le point d'arrivée (indices 1..N-1)
    feats[1:, 2] = speed
    feats[1:, 3] = sin_t
    feats[1:, 4] = cos_t

    # Virage : différence de cap entre segments consécutifs (arrivée >= 2)
    dtheta = np.diff(theta)
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))   # wrap dans [-π, π]
    valid = moving[:-1] & moving[1:]
    feats[2:, 5] = np.where(valid, dtheta, 0.0)

    return feats.astype(np.float32)


class Normalizer:
    """Standardisation z-score par dimension."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std[self.std < 1e-6] = 1.0

    def apply(self, feats):
        return (feats - self.mean) / self.std

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["mean"], d["std"])

    @classmethod
    def fit(cls, points, offsets, max_trips=20000):
        """Calcule mean/std sur (un échantillon de) trajets du cache CSR."""
        n_trips = len(offsets) - 1
        idx = np.arange(n_trips)
        if n_trips > max_trips:
            rng = np.random.default_rng(0)
            idx = rng.choice(n_trips, size=max_trips, replace=False)

        s = np.zeros(N_FEATURES, dtype=np.float64)
        ss = np.zeros(N_FEATURES, dtype=np.float64)
        count = 0
        for i in idx:
            seg = points[offsets[i]:offsets[i + 1]]
            f = compute_features(seg).astype(np.float64)
            s += f.sum(axis=0)
            ss += (f * f).sum(axis=0)
            count += f.shape[0]
        mean = s / count
        var = ss / count - mean * mean
        std = np.sqrt(np.maximum(var, 1e-12))
        return cls(mean.astype(np.float32), std.astype(np.float32))
