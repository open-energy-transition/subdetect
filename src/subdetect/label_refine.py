"""Refine oversized `power=substation` label polygons using S1 + S2 evidence.

Some OSM substation ways are drawn around the whole fenced property rather than the
actual switchyard footprint, which burns adjacent farmland/bare land as class-1 in
training masks. Two independent signals fix this:

- **S1 existence** (VV/VH hysteresis: seed >= local-background+6dB, grow through
  local-background+2dB): metal gantries/lattice structures produce a strong
  double-bounce return that bare ground and cropland don't. A polygon with no such
  core anywhere inside it likely has no real transmission-class infrastructure at
  all -- flag for exclusion (`status=no_signal`), not just refinement.
- **S2 NDVI** ((NIR-RED)/(NIR+RED) < 0.25 = non-vegetated): distinguishes the real
  bare gravel/concrete switchyard from vegetated property/farmland swept into the
  same way. S1 alone under-refines: real switchyards have a lot of legitimate empty
  clearance space between equipment that doesn't backscatter strongly, so a low S1
  hot-fraction does NOT mean the polygon is oversized (validated against two known-
  good 500kV substations, which score core_frac 0.16/0.42 despite being tightly
  drawn). NDVI is the actual boundary signal; S1 only gates whether refinement
  should run at all.

Only ever shrinks (refined = original polygon INTERSECT non-vegetated core anchored
to the S1 hotspot), never grows a polygon beyond what OSM drew.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio import features as rio_features
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

from subdetect.config import S1_OFFSET_DB, S1_SCALE, Settings, geodesic_area_m2
from subdetect.local_source import CompositeIndex

log = logging.getLogger(__name__)

HI_DELTA_DB = 6.0     # S1 seed: this many dB above the local background
LO_DELTA_DB = 2.0     # S1 grow: through pixels this many dB above background
MIN_CORE_M2 = 400.0   # ignore S1 cores smaller than ~4 S2 pixels (speckle)
NDVI_THRESHOLD = 0.25  # below this = non-vegetated (bare/gravel/concrete)
KEEP_RATIO = 0.9       # refined/original area >= this -> not worth changing, keep original
COMPACTNESS_MIN = 0.7  # largest connected piece must be this fraction of the refined core,
                       # else it's scattered bare patches (fallow fields, village rooftops)
                       # rather than one structure -- skip refinement, flag for review


def _window_bbox(geom, pad_frac: float = 0.5, pad_deg: float = 0.0015) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = geom.bounds
    pad = max(maxx - minx, maxy - miny) * pad_frac + pad_deg
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)


def _s1_core_mask(arr, transform, geom_proj):
    """Returns (core_mask, poly_mask) or (None, None) if there's not enough signal to judge."""
    valid = (arr[0] > 0) & (arr[1] > 0)
    if valid.sum() < 25:
        return None, None
    vv_db = np.where(valid, arr[0].astype(float) / S1_SCALE - S1_OFFSET_DB, np.nan)
    vh_db = np.where(valid, arr[1].astype(float) / S1_SCALE - S1_OFFSET_DB, np.nan)
    backscatter = np.nanmax(np.stack([vv_db, vh_db]), axis=0)

    poly_mask = rio_features.geometry_mask(
        [geom_proj.__geo_interface__], out_shape=backscatter.shape, transform=transform, invert=True
    ) & valid
    bg_mask = valid & ~poly_mask
    if bg_mask.sum() < 25 or poly_mask.sum() == 0:
        return None, None
    bg_median = np.nanmedian(backscatter[bg_mask])

    px_area = abs(transform.a * transform.e)
    lo_mask = valid & (backscatter >= bg_median + LO_DELTA_DB)
    hi_mask = valid & (backscatter >= bg_median + HI_DELTA_DB)
    lab, n = ndimage.label(lo_mask, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(poly_mask), poly_mask

    seed_ids = np.unique(lab[hi_mask])
    seed_ids = seed_ids[seed_ids != 0]
    sizes = ndimage.sum(lo_mask, lab, index=range(1, n + 1)) * px_area
    overlaps = ndimage.sum(poly_mask, lab, index=range(1, n + 1))
    keep = [k for k in seed_ids if sizes[k - 1] >= MIN_CORE_M2 and overlaps[k - 1] > 0]
    core = np.isin(lab, keep) & poly_mask
    return core, poly_mask


def refine_one(idx: CompositeIndex, geom) -> dict:
    """Refine a single substation polygon (EPSG:4326). Returns a result dict."""
    bbox = _window_bbox(geom)
    orig_area_m2 = geodesic_area_m2(geom)
    try:
        s1_res = idx.read_window(bbox, "composite_s1.tif")
    except FileNotFoundError:
        s1_res = None
    if s1_res is None:
        return dict(status="no_s1", geometry=geom, orig_area_m2=orig_area_m2, refined_area_m2=None)
    s1_arr, transform, crs = s1_res

    geom_proj = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(crs).iloc[0]
    core, poly_mask = _s1_core_mask(s1_arr, transform, geom_proj)
    if core is None:
        return dict(status="no_bg", geometry=geom, orig_area_m2=orig_area_m2, refined_area_m2=None)
    if core.sum() == 0:
        return dict(status="no_signal", geometry=geom, orig_area_m2=orig_area_m2, refined_area_m2=0.0)

    # NDVI on the co-registered S2 window.
    s2_res = idx.read_window(bbox, "composite_0.tif")
    if s2_res is None:
        return dict(status="no_s2", geometry=geom, orig_area_m2=orig_area_m2, refined_area_m2=None)
    s2_arr, s2_transform, s2_crs = s2_res
    if s2_arr.shape[1:] != core.shape:
        # Grids didn't align (rare edge case) -- skip refinement, keep original.
        return dict(status="grid_mismatch", geometry=geom, orig_area_m2=orig_area_m2,
                     refined_area_m2=orig_area_m2)
    red = s2_arr[2].astype(float)
    nir = s2_arr[6].astype(float)
    valid_s2 = (red > 0) & (nir > 0)
    ndvi = np.where(valid_s2, (nir - red) / (nir + red + 1e-6), np.nan)
    bare = valid_s2 & (ndvi < NDVI_THRESHOLD)

    lab, n = ndimage.label(bare, structure=np.ones((3, 3)))
    if n == 0:
        return dict(status="no_bare", geometry=geom, orig_area_m2=orig_area_m2, refined_area_m2=0.0)
    anchor_ids = np.unique(lab[core])
    anchor_ids = anchor_ids[anchor_ids != 0]
    if anchor_ids.size == 0:
        return dict(status="refine_failed", geometry=geom, orig_area_m2=orig_area_m2,
                    refined_area_m2=orig_area_m2)

    refined_mask = poly_mask & np.isin(lab, anchor_ids)
    px_area = abs(transform.a * transform.e)
    refined_area_m2 = float(refined_mask.sum() * px_area)
    ratio = refined_area_m2 / orig_area_m2 if orig_area_m2 else 0.0

    if ratio >= KEEP_RATIO or refined_area_m2 == 0.0:
        return dict(status="tight", geometry=geom, orig_area_m2=orig_area_m2,
                    refined_area_m2=refined_area_m2)

    frag_lab, frag_n = ndimage.label(refined_mask, structure=np.ones((3, 3)))
    comp_sizes = ndimage.sum(refined_mask, frag_lab, index=range(1, frag_n + 1)) * px_area
    dominant_frac = max(comp_sizes) / sum(comp_sizes) if frag_n else 0.0
    if dominant_frac < COMPACTNESS_MIN:
        # Scattered bare patches (fallow fields) or clustered building rooftops (villages)
        # both trip the NDVI+S1 gates without being one real structure -- don't guess, keep
        # the original polygon and flag for manual review instead of auto-shrinking onto noise.
        return dict(status="fragmented", geometry=geom, orig_area_m2=orig_area_m2,
                    refined_area_m2=orig_area_m2)

    polys = [shape(g) for g, _ in rio_features.shapes(refined_mask.astype("uint8"),
                                                       mask=refined_mask, transform=transform)]
    if not polys:
        return dict(status="refine_failed", geometry=geom, orig_area_m2=orig_area_m2,
                    refined_area_m2=orig_area_m2)
    refined_geom_proj = unary_union(polys)
    refined_geom = gpd.GeoSeries([refined_geom_proj], crs=crs).to_crs("EPSG:4326").iloc[0]
    return dict(status="refined", geometry=refined_geom, orig_area_m2=orig_area_m2,
                refined_area_m2=refined_area_m2)


def refine_labels(aoi: str, labels_dir: Path = Path("data/labels"),
                   composites_dir: Path = Path("data/composites")) -> gpd.GeoDataFrame:
    """Refine every >=floor substation polygon for `aoi`. Returns the full substations_poly
    table (all roles) with `status`, `orig_area_m2`, `refined_area_m2` columns added; only
    rows that were actually refined get a replaced geometry. Writes
    `<labels_dir>/<aoi>/substations_poly_refined.parquet` (does not touch the input file).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings.load()
    subs = gpd.read_parquet(Path(labels_dir) / aoi / "substations_poly.parquet")
    if "area_m2" not in subs.columns:
        subs["area_m2"] = [geodesic_area_m2(g) for g in subs.geometry]

    idx = CompositeIndex(Path(composites_dir) / aoi)
    big = subs.area_m2 >= settings.min_sub_area_m2
    results = []
    for i, row in subs.iterrows():
        if not big.loc[i]:
            results.append(dict(status="unchecked", geometry=row.geometry,
                                orig_area_m2=row.area_m2, refined_area_m2=row.area_m2))
            continue
        results.append(refine_one(idx, row.geometry))

    out = subs.copy()
    out["status"] = [r["status"] for r in results]
    out["orig_area_m2"] = [r["orig_area_m2"] for r in results]
    out["refined_area_m2"] = [r["refined_area_m2"] for r in results]
    out["geometry"] = [r["geometry"] for r in results]
    out["area_m2"] = [geodesic_area_m2(g) for g in out.geometry]

    out_path = Path(labels_dir) / aoi / "substations_poly_refined.parquet"
    out.to_parquet(out_path)
    log.info("Wrote %s (%d rows); status counts:\n%s", out_path, len(out),
             out.status.value_counts().to_string())
    return out
