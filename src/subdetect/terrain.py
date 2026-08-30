"""Copernicus DEM GLO-30 -> co-registered terrain channels per composite cell.

Added 2026-08-29 for the small run-of-river hydro feasibility study. Until then the repo
had no elevation data of any kind: `sentinel-1-rtc` is terrain-corrected upstream by the
data provider, but nothing here knew about slope, aspect, layover or valley position.

That is tolerable in Sindh and not tolerable in the Karakoram, for two separate reasons.
As a *model input*, a run-of-river plant is defined by topography — it sits on a valley
floor just below a steep drop, beside a stream — so terrain is the strongest available
prior. As an *analysis* input, every S1 feature in `scripts/` models the background with a
31-px uniform filter, which assumes stationarity; on a mountainside it is the terrain, not
the infrastructure, that dominates any contrast measurement. Matched controls need terrain
covariates or the comparison measures "valley vs mountainside" and nothing else.

Four bands are stored (see config.DEM_BANDS); only the last three are fed to a model.

  elev_m         raw elevation. Matching covariate, aspect source, snow-line QC.
  slope_deg      Horn 3x3 on the metric grid. A bench-cut powerhouse pad is a flat pixel.
  rel_elev_500m  (z - min51) / (max51 - min51): valley-floor-ness, dimensionless, ~0 on
                 the floor and ~1 on the ridge.
  relief_1km     max101 - min101: gorge-ness, and a proxy for available hydraulic head.

Two choices worth defending, because the obvious alternatives are wrong:

*Normalised* relative elevation rather than a raw `z - min` HAND proxy. The raw form and
`relief` are algebraically nested — both grow monotonically with local ruggedness — so as
two model channels they are largely redundant, and as two matching covariates they
double-weight one axis inside any distance metric. Normalising decorrelates them.

Window scales of 510 m and 1010 m rather than the 1-2 km that "height above drainage"
usually implies. Chitral/Swat/Gilgit sites sit in *tributary* side-valleys 200-800 m wide;
a 2 km minimum filter centred on a tributary floor reaches over the intervening spur into
the trunk valley and measures height above the wrong river entirely.

Known limitation, documented rather than fixed: on an alluvial fan or river terrace the
window minimum sits at the fan toe, so a genuine valley-floor pixel scores spuriously
high. Acceptable for a model prior; it is why distance-to-stream must also be a matching
covariate rather than relying on rel_elev alone.

No flow-routing library is used or needed (none is installed). Pure numpy + scipy.ndimage,
~0.2 s per cell — the STAC read is the only real cost.
"""

from __future__ import annotations

import logging

import numpy as np
import odc.stac
from scipy import ndimage

from subdetect.config import DEM_BANDS, DEM_OFFSET, DEM_SCALE
from subdetect.imagery import _SEARCH_LOCK, _catalog

log = logging.getLogger(__name__)

DEM_COLLECTION = "cop-dem-glo-30"
DEM_ASSET = "data"

# Halo carried into the window filters and cropped off afterwards. Must exceed half the
# widest window (101 px); 128 px = 1.28 km leaves generous margin. Without it every cell
# edge gets a ~500 m band of wrong values AND adjacent cells disagree along their seam.
DEM_PAD_PX = 128

WIN_VALLEY_PX = 51    # 510 m: tributary-valley position
WIN_RELIEF_PX = 101   # 1010 m: gorge relief
PRESMOOTH_PX = 1.5    # ~15 m, half the 30 m source posting
FLAT_RELIEF_M = 5.0   # below this the normalised ratio is noise, not signal


def fetch_elevation(bbox, geobox, pad_px: int = DEM_PAD_PX):
    """(z, valid) float32/bool on `geobox` padded by `pad_px`, or None if no tiles.

    `bbox` is the cell's WGS84 bbox for the STAC search — the same contract as
    `imagery.s1_composite`: search by bbox, land on the geobox.
    """
    gpad = geobox.pad(pad_px)
    # Pad the SEARCH bbox too, or a cell near a 1-deg tile boundary silently misses the
    # neighbouring tile and the halo comes back nodata.
    mx = pad_px * 10.0 / 111_320.0
    sbox = (bbox[0] - mx * 2, bbox[1] - mx, bbox[2] + mx * 2, bbox[3] + mx)
    with _SEARCH_LOCK:  # pystac-client is not thread-safe; reuse imagery.py's lock
        items = list(_catalog().search(collections=[DEM_COLLECTION], bbox=sbox).items())
    if not items:
        log.warning("no %s tiles for bbox %s", DEM_COLLECTION, bbox)
        return None
    # groupby="time" is correct here *because* every COP30 item carries the same datetime,
    # so the tiles collapse into one mosaicked slice. groupby=None would return one slice
    # per tile and produce a NaN-striped array.
    ds = odc.stac.load(
        items, bands=[DEM_ASSET], geobox=gpad, resampling="bilinear",
        groupby="time", chunks={"x": 2048, "y": 2048}, fail_on_error=False,
    )
    z = ds[DEM_ASSET]
    if "time" in z.dims:
        z = z.isel(time=0, drop=True)
    z = z.compute().values.astype("float32")
    valid = np.isfinite(z) & (z > -1000.0)
    if valid.mean() < 0.01:
        log.warning("DEM for bbox %s is %.1f%% valid -- skipping", bbox, 100 * valid.mean())
        return None
    return z, valid


def _nearest_fill(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid pixels with their nearest valid neighbour's value.

    min/max/gaussian filters have undefined behaviour on NaN — every NaN comparison is
    False, so one NaN silently poisons an entire window. Filling first and re-masking
    afterwards is both cheaper and more correct than any NaN-aware filter.
    """
    if valid.all():
        return z
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return z[tuple(idx)]


def slope_aspect(z: np.ndarray, px: float = 10.0, presmooth_px: float = PRESMOOTH_PX):
    """Horn 3x3 slope (degrees) and aspect (degrees clockwise from north), metric grid.

    The presmooth is not cosmetic: the source is 30 m posting resampled to 10 m, so a raw
    3x3 stencil differentiates over a 20 m baseline — two thirds of one source post — and
    reports the interpolant's own ripple as terrain texture.
    """
    zs = ndimage.gaussian_filter(z, presmooth_px, mode="nearest")
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], "float32") / (8.0 * px)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], "float32") / (8.0 * px)
    dzdx = ndimage.correlate(zs, kx, mode="nearest")
    # Rows increase SOUTHWARD in a north-up raster, so the row gradient must be negated to
    # get d(z)/d(north). Getting this wrong mirrors aspect N<->S while leaving slope
    # magnitude perfectly correct -- i.e. it fails silently, and would invert every
    # illumination-matching decision downstream.
    dzdy = -ndimage.correlate(zs, ky, mode="nearest")
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype("float32")
    # Aspect is the DOWNHILL bearing -- the direction the slope faces, standard GIS
    # convention -- hence the negated gradients: arctan2(dzdx, dzdy) would give the uphill
    # bearing, i.e. every value rotated 180 deg. Verified: a slope facing north returns 0
    # and one facing east returns 90, so cos(aspect) is +1 for north-facing terrain.
    aspect = (np.degrees(np.arctan2(-dzdx, -dzdy)) % 360.0).astype("float32")
    return slope, aspect


def terrain_channels(z: np.ndarray, px: float = 10.0) -> dict[str, np.ndarray]:
    """The three derived channels; elevation itself is passed through by the caller."""
    slope, _ = slope_aspect(z, px)
    zmin_v = ndimage.minimum_filter(z, size=WIN_VALLEY_PX, mode="nearest")
    zmax_v = ndimage.maximum_filter(z, size=WIN_VALLEY_PX, mode="nearest")
    rng = zmax_v - zmin_v
    # On a flat plain `rng` is just the DEM's own vertical noise, and (z-min)/rng would
    # amplify that noise across the full [0,1] range. Return the neutral 0.5 instead.
    rel = np.where(rng < FLAT_RELIEF_M, 0.5, (z - zmin_v) / np.maximum(rng, FLAT_RELIEF_M))
    relief = (ndimage.maximum_filter(z, size=WIN_RELIEF_PX, mode="nearest")
              - ndimage.minimum_filter(z, size=WIN_RELIEF_PX, mode="nearest"))
    return {
        "slope_deg": slope,
        "rel_elev_500m": np.clip(rel, 0.0, 1.0).astype("float32"),
        "relief_1km": relief.astype("float32"),
    }


def dem_dn(z: np.ndarray, valid: np.ndarray, channels: dict[str, np.ndarray]) -> np.ndarray:
    """(4, H, W) uint16 DN, 0 = nodata, per config.DEM_OFFSET / DEM_SCALE."""
    vals = [z] + [channels[b] for b in DEM_BANDS[1:]]
    out = np.zeros((len(DEM_BANDS),) + z.shape, dtype="uint16")
    for i, (v, off, sc) in enumerate(zip(vals, DEM_OFFSET, DEM_SCALE)):
        x = np.clip(np.asarray(v, "float64") + off, 0.0, 65534.0 / sc)
        out[i][valid] = (1 + np.round(x[valid] * sc)).astype("uint16")
    return out


def dem_composite(bbox, geobox, pad_px: int = DEM_PAD_PX):
    """(array[4,H,W] uint16, transform, crs) on `geobox` exactly, or None.

    Same return contract as imagery.annual_composite / s1_composite, so compose.py's
    writer needs no special case.
    """
    got = fetch_elevation(bbox, geobox, pad_px)
    if got is None:
        return None
    zp, validp = got
    h, w = geobox.shape
    exp = (h + 2 * pad_px, w + 2 * pad_px)
    if zp.shape != exp:
        # GeoBox.pad(n) expands by exactly n px on each side on the same lattice, so this
        # should never fire; if it does, the crop below would be misaligned by a pixel and
        # every terrain value would be attributed to the wrong ground location.
        raise ValueError(f"padded DEM shape {zp.shape} != expected {exp}")
    zf = _nearest_fill(zp, validp)
    ch = terrain_channels(zf, px=abs(geobox.resolution.x))
    s = (slice(pad_px, pad_px + h), slice(pad_px, pad_px + w))
    return dem_dn(zf[s], validp[s], {k: v[s] for k, v in ch.items()}), geobox.transform, geobox.crs


def decode_dem(dn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """uint16 DN stack -> (float32 physical values, valid mask). Inverse of dem_dn."""
    dn = np.asarray(dn)
    valid = dn[0] > 0
    out = np.zeros(dn.shape, dtype="float32")
    for i, (off, sc) in enumerate(zip(DEM_OFFSET, DEM_SCALE)):
        out[i] = (dn[i].astype("float32") - 1.0) / sc - off
    out[:, ~valid] = np.nan
    return out, valid
