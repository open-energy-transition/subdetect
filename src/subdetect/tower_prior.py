"""Rasterized S1-derived power-tower proximity prior — extra model input channel.

Mirrors the validated point-detection formula in `scripts/isolated_tower_prior.py`
(contrast_db=5.0 VH local-contrast maxima, isolation filter <=3 neighbors/250 m —
see docs/issues/s1-tower-chain-corridor-evidence.md, memory subdetect-tower-s1-priors):
individually detectable towers are S1-only (S2 is blind to sub-pixel towers), so this
never touches S2. Unlike that script, which scores pre-existing candidate polygons,
this rasterizes a continuous exp-decay-to-nearest-point field over the whole cell so
every chip cut from it carries the same "is there a tower chain nearby" information —
deployed as an extra TerraMind input modality with a freshly-initialized patch embed
(see terramind.py's dict-modality path), not through OSM, so it generalizes to
unmapped corridors exactly like the validated `iso_towers_1km` ranking feature does.

Distances are computed in pixel space of the composite's native grid (~10 m/px) rather
than reprojecting to an equal-area CRS first, unlike the candidate-ranking script — an
approximation acceptable here since composite tiles are small (0.1 deg) local UTM-ish
grids where pixel spacing is close to metric, and this is a per-chip model input, not a
precision measurement.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

CONTRAST_DB = 5.0
MAX_NEIGHBORS = 3
NEIGHBOR_RADIUS_PX = 25.0  # ~250 m at 10 m/px, matches the validated ranking feature
DECAY_PX = 200.0  # ~2 km at 10 m/px, matches the 0.5+0.5*exp(-d/2000) ranking prior shape


def bright_point_mask(s1_arr: np.ndarray, contrast_db: float = CONTRAST_DB,
                      max_neighbors: int = MAX_NEIGHBORS) -> np.ndarray:
    """(H, W) bool mask of isolated VH local-contrast maxima (tower signature).

    `s1_arr` is the raw (2, H, W) S1 DN array (VV, VH order per config.S1_BANDS).
    """
    vh = s1_arr[1].astype("float32")
    valid = vh > 0
    db = vh / 500.0 - 50.0  # config.S1_SCALE / S1_OFFSET_DB
    band = np.where(valid, db, np.nan)
    local = ndimage.uniform_filter(np.nan_to_num(band), 31)
    norm = ndimage.uniform_filter(valid.astype("float32"), 31)
    local = np.where(norm > 0.3, local / np.maximum(norm, 1e-6), np.nan)
    contrast = np.nan_to_num(band - local, nan=-99.0)
    is_max = contrast == ndimage.maximum_filter(contrast, 3)
    mask = is_max & (contrast >= contrast_db) & valid
    rows, cols = np.where(mask)
    if len(rows) > 1 and max_neighbors >= 0:
        pts = np.column_stack([cols, rows]).astype(float)
        tree = cKDTree(pts)
        n_nb = np.array([len(nb) - 1 for nb in tree.query_ball_point(pts, NEIGHBOR_RADIUS_PX)])
        keep = n_nb <= max_neighbors
        out = np.zeros_like(mask)
        out[rows[keep], cols[keep]] = True
        return out
    return mask


def tower_prior_raster(s1_arr: np.ndarray, decay_px: float = DECAY_PX) -> tuple[np.ndarray, np.ndarray]:
    """Returns (prior, valid): prior is (H, W) float32 in [0, 1] = exp(-dist_to_nearest_
    isolated_bright_point / decay_px); valid is (H, W) bool, True where the source S1
    pixel itself has data (both VV and VH > 0) — callers should treat ~valid as nodata,
    NOT as prior==0 (a genuinely tower-free area also scores near 0)."""
    valid = (s1_arr[0] > 0) & (s1_arr[1] > 0)
    mask = bright_point_mask(s1_arr)
    if not mask.any():
        return np.zeros(s1_arr.shape[-2:], dtype="float32"), valid
    dist = ndimage.distance_transform_edt(~mask)
    prior = np.exp(-dist / decay_px).astype("float32")
    return prior, valid


# uint16 DN encoding for the composite COG: 0 reserved for nodata (no S1 coverage),
# 1..65535 linearly spans prior in [0, 1] — mirrors config.S1_OFFSET_DB/S1_SCALE's
# reserved-zero convention so downstream nodata handling stays consistent.
DN_SCALE = 65534.0


def tower_prior_dn(s1_arr: np.ndarray) -> np.ndarray:
    """(1, H, W) uint16 DN array ready to write as a single-band composite layer."""
    prior, valid = tower_prior_raster(s1_arr)
    dn = np.zeros(prior.shape, dtype="uint16")
    dn[valid] = 1 + np.round(np.clip(prior[valid], 0.0, 1.0) * DN_SCALE).astype("uint16")
    return dn[None]
