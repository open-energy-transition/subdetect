"""Are BESS separable from *matched* controls in S2/S1? (Stage 2a)

The decisive feasibility question, and it costs no GPU to answer. If a plain feature
classifier cannot tell a battery yard from an area-matched battery-free substation, a
fine-tuned U-Net on ~259 examples almost certainly cannot either.

Structure follows `scripts/hydro_detectability.py`; `local_contrast`, `rank_auc`, `smd` and
the balance-table-first discipline are reused unchanged. Two things are different, and both
come out of that study's post-mortem.

TWO TIERS, AND WHY THE SECOND ONE IS THE ONLY HONEST ONE
--------------------------------------------------------
76% of the mapped battery sites in GB/CA/TX have a substation within 300 m and 41% of their
polygons physically overlap one (Stage 0, G0.6). A substation is exactly what this repo's
production model detects. So:

  Tier 1  BESS vs matched generic/industrial land. A canary: it tells a real null apart
          from broken imagery, and it is an upper bound. It can never carry the verdict,
          because most of what it measures is "this is infrastructure, that is a field".

  Tier 2  BESS vs area-matched, shape-matched, battery-free SUBSTATIONS. This is the number
          the gate is set on. It asks whether there is a battery-specific signal at all,
          over and above the switchyard the model already finds.

SHAPE-MATCHED CONTROLS, NOT DISCS
---------------------------------
The hydro study's permutation null centred at 0.564 rather than 0.500, and its own
post-mortem attributed that to comparing irregular real polygons against circular control
discs -- a pure geometry asymmetry that inflates every AUC built on it. Tier 1 controls
here are therefore a *rotated, translated copy of the site's own polygon* (`shape_mask`),
and Tier 2 controls are real substation polygons, which are shape-like by construction.
Gate 2's prerequisite P2 checks the null median to verify this actually worked.

    pixi run -e ml python scripts/bess_detectability.py --tier 1 --run-name bess_v1
    pixi run -e ml python scripts/bess_detectability.py --tier 2 --run-name bess_v1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from scipy import ndimage
from shapely import affinity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subdetect.config import S1_OFFSET_DB, S1_SCALE  # noqa: E402
from subdetect.terrain import decode_dem  # noqa: E402

log = logging.getLogger("bess_detectability")

LABELS_ROOT = ROOT / "data" / "labels"
COMPOSITES_ROOT = ROOT / "data" / "composites"
RESULTS = ROOT / "data" / "eval_results"
EVAL_VERSION = 1
REGION_AOI = {"gb": "gb_bess", "ca": "ca_bess", "tx": "tx_bess"}

# S2 L2A baseline 04.00+ carries a +1000 DN offset. optical_features.py subtracts it;
# datamodule.py does not (harmless there -- the model trained on that convention). For
# spectral indices it is NOT harmless: omitting it biases every dark-target ratio.
S2_DN_OFFSET, S2_DN_SCALE = 1000.0, 10000.0

S2_BAND_NAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# Tier-1 matching calipers, on covariates deliberately DISJOINT from the features under
# test (terrain and OSM geometry, never reflectance), so matching cannot destroy the signal
# being measured. Loosening order is fixed in advance; valid_frac_s2 is NEVER loosened,
# because relaxing it is what would manufacture a result out of cloud gaps.
CALIPERS = {
    "elev_m": 150.0,
    "slope_deg": 4.0,
    "d_road_m": None,        # handled in log space, see D_LOG_TOL
    "d_line_m": None,
    "d_industrial_m": None,
}
LOG_CALIPERS = ["d_road_m", "d_line_m", "d_industrial_m"]
D_LOG_TOL = 0.6              # ~ +/-80% in metres
VALID_FRAC_TOL = 0.15
LOOSEN_ORDER = ["d_industrial_m", "d_road_m", "elev_m"]
MIN_SEP_M = 300.0            # between accepted controls, so they are not one patch
MIN_SITES_FOR_FEATURE_AUC = 10

# Tier-1 eligibility: keep controls away from any battery and from the power network, so
# "generic land" means what it says.
EXCLUDE_BESS_M = 500.0
EXCLUDE_POWER_M = 300.0

# Tier-2 area caliper, loosened in this fixed order if the strata come out thin.
AREA_TOL_ORDER = [0.30, 0.50, 1.00]
CTRL_MAX_DIST_M = 40_000.0   # amendment A2: one S2 granule / one S1 look geometry


# --------------------------------------------------------------------------- helpers


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney rank AUC. Same estimator as hydro_detectability.rank_auc."""
    pos = np.asarray(pos, "float64")[np.isfinite(np.asarray(pos, "float64"))]
    neg = np.asarray(neg, "float64")[np.isfinite(np.asarray(neg, "float64"))]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    sc = np.r_[pos, neg]
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = sc.argsort()
    ranks = np.empty(len(sc))
    ranks[order] = np.arange(len(sc))
    return float((ranks[lab == 1].mean() - (lab.sum() - 1) / 2) / max((lab == 0).sum(), 1))


def smd(pos: np.ndarray, neg: np.ndarray) -> float:
    """Standardized mean difference -- Rubin's balance diagnostic."""
    pos = np.asarray(pos, "float64")[np.isfinite(np.asarray(pos, "float64"))]
    neg = np.asarray(neg, "float64")[np.isfinite(np.asarray(neg, "float64"))]
    if pos.size < 2 or neg.size < 2:
        return float("nan")
    denom = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2.0)
    return float((pos.mean() - neg.mean()) / denom) if denom > 0 else float("nan")


def local_contrast(band_db: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """31-px uniform-filter local contrast, verbatim from ehv/hydro_detectability."""
    band = np.where(valid, band_db, np.nan)
    local = ndimage.uniform_filter(np.nan_to_num(band), 31)
    norm = ndimage.uniform_filter(valid.astype("float32"), 31)
    local = np.where(norm > 0.3, local / np.maximum(norm, 1e-6), np.nan)
    return np.nan_to_num(band - local, nan=-99.0)


# ------------------------------------------------------------------- per-cell rasters


class CellStack:
    """Everything needed from one 0.1 deg cell, on its own metric grid.

    At 10 m/px in a UTM grid, pixel distance IS metric distance, so
    distance_transform_edt * 10 gives metres. `composite_pre.tif` is optional: without it
    the `chg_*` feature group is simply absent, which is what G2.3 expects for any site
    whose pre-construction epoch could not be established.
    """

    def __init__(self, aoi: str, cell: str):
        d = COMPOSITES_ROOT / aoi / "composites" / cell
        self.aoi, self.cell, self.ok, self.has_pre = aoi, cell, False, False
        s2p, s1p, demp = d / "composite_0.tif", d / "composite_s1.tif", d / "composite_dem.tif"
        if not (s2p.exists() and s1p.exists() and demp.exists()):
            return
        with rasterio.open(s2p) as src:
            s2 = src.read().astype("float32")
            self.transform, self.crs, self.shape = src.transform, src.crs, (src.height, src.width)
        with rasterio.open(s1p) as src:
            s1dn = src.read().astype("float32")
        with rasterio.open(demp) as src:
            dem_dn = src.read()
        self.s2_valid = s2[0] > 0
        self.refl = np.where(self.s2_valid, (s2 - S2_DN_OFFSET) / S2_DN_SCALE, np.nan)
        self.s1_valid = s1dn[0] > 0
        self.s1_db = np.where(self.s1_valid, s1dn / S1_SCALE - S1_OFFSET_DB, np.nan)
        self.con_vv = local_contrast(self.s1_db[0], self.s1_valid)
        self.con_vh = local_contrast(self.s1_db[1], self.s1_valid)
        self.dem, self.dem_valid = decode_dem(dem_dn)
        prep = d / "composite_pre.tif"
        if prep.exists():
            with rasterio.open(prep) as src:
                pre = src.read().astype("float32")
            if pre.shape == s2.shape:
                self.pre_valid = pre[0] > 0
                self.pre = np.where(self.pre_valid, (pre - S2_DN_OFFSET) / S2_DN_SCALE, np.nan)
                self.has_pre = True
        self.ok = True

    def rasterize(self, gdf) -> np.ndarray:
        if gdf is None or len(gdf) == 0:
            return np.zeros(self.shape, dtype=bool)
        g = gdf.to_crs(self.crs)
        g = g[~g.geometry.is_empty & g.geometry.notna()]
        if g.empty:
            return np.zeros(self.shape, dtype=bool)
        # all_touched=False, pre-registered: all_touched inflates a small polygon by
        # roughly its perimeter in pixels (measured 1.70x in the 1,000-2,000 m2 bucket).
        return rasterio.features.rasterize(
            [(x, 1) for x in g.geometry], out_shape=self.shape, transform=self.transform,
            fill=0, all_touched=False, dtype="uint8").astype(bool)

    def dist_m(self, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.full(self.shape, 1e6, dtype="float32")
        return (ndimage.distance_transform_edt(~mask) * 10.0).astype("float32")

    def covariates(self, roads, lines, industrial, bess, power_all) -> dict:
        s2_frac = ndimage.uniform_filter(self.s2_valid.astype("float32"), 31)
        cov = {
            "elev_m": self.dem[0], "slope_deg": self.dem[1],
            "d_road_m": self.dist_m(self.rasterize(roads)),
            "d_line_m": self.dist_m(self.rasterize(lines)),
            "d_industrial_m": self.dist_m(self.rasterize(industrial)),
            "valid_frac_s2": s2_frac,
        }
        cov["eligible"] = (
            self.s2_valid & self.s1_valid & self.dem_valid
            & (self.dist_m(self.rasterize(bess)) >= EXCLUDE_BESS_M)
            & (self.dist_m(self.rasterize(power_all)) >= EXCLUDE_POWER_M)
        )
        return cov


# ------------------------------------------------------------------------- features


def _ratio(x, y):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x - y) / (x + y)


def region_features(stack: CellStack, mask: np.ndarray) -> dict | None:
    """S2 + S1 (+ bi-temporal change) statistics over one region."""
    if mask.sum() < 3:
        return None
    r = stack.refl[:, mask]
    if not np.isfinite(r).any():
        return None
    b = {n: r[i] for i, n in enumerate(S2_BAND_NAMES)}
    bright = np.nanmean(r[:3], axis=0)
    ndvi = _ratio(b["B08"], b["B04"])
    f = {f"{n.lower()}_mean": float(np.nanmean(b[n])) for n in S2_BAND_NAMES}
    f.update({
        "bright_mean": float(np.nanmean(bright)),
        "bright_p95": float(np.nanpercentile(bright, 95)),
        "ndvi_mean": float(np.nanmean(ndvi)), "ndvi_min": float(np.nanmin(ndvi)),
        "ndwi_max": float(np.nanmax(_ratio(b["B03"], b["B08"]))),
        "mndwi_max": float(np.nanmax(_ratio(b["B03"], b["B11"]))),
        "ndbi_max": float(np.nanmax(_ratio(b["B11"], b["B08"]))),   # built surface
        "bsi_max": float(np.nanmax(_ratio(b["B11"] + b["B04"], b["B08"] + b["B02"]))),
        "tex_std": float(np.nanstd(bright)),
        "valid_frac_s2": float(np.isfinite(r[0]).mean()),
    })
    # Ring contrast: the area-generalized form of the tower prior's local contrast. A
    # container yard should be brighter than its own surroundings; a switchyard also is,
    # which is exactly why this feature alone cannot settle the question.
    ring = (ndimage.binary_dilation(mask, iterations=15)
            & ~ndimage.binary_dilation(mask, iterations=3))
    if ring.any():
        rb = np.nanmean(stack.refl[:3][:, ring], axis=0)
        f["bright_ring_contrast"] = float(np.nanmean(bright) - np.nanmean(rb))
    else:
        f["bright_ring_contrast"] = np.nan

    vv, vh = stack.s1_db[0][mask], stack.s1_db[1][mask]
    cvv, cvh = stack.con_vv[mask], stack.con_vh[mask]
    cvh_ok = cvh[cvh > -50]
    f.update({
        "vv_db_mean": float(np.nanmean(vv)), "vh_db_mean": float(np.nanmean(vh)),
        "vvvh_ratio_db": float(np.nanmean(vv) - np.nanmean(vh)),
        "vh_std_db": float(np.nanstd(vh)),
        "vv_contrast_max": float(np.nanmax(cvv[cvv > -50])) if (cvv > -50).any() else np.nan,
        "vh_contrast_max": float(np.nanmax(cvh_ok)) if cvh_ok.size else np.nan,
        "n_bright_px_5db_per_ha": float((cvh_ok >= 5.0).sum() / (mask.sum() * 100.0 / 1e4))
        if cvh_ok.size else np.nan,
    })

    # Bi-temporal change. Prefixed chg_ so group_of() can hold the whole family back as
    # the single pre-declared G2.3 extension rather than letting it into the headline.
    if stack.has_pre:
        p = stack.pre[:, mask]
        if np.isfinite(p).any():
            pb = {n: p[i] for i, n in enumerate(S2_BAND_NAMES)}
            pbright = np.nanmean(p[:3], axis=0)
            pndvi = _ratio(pb["B08"], pb["B04"])
            f.update({
                "chg_bright_mean": float(np.nanmean(bright) - np.nanmean(pbright)),
                "chg_ndvi_mean": float(np.nanmean(ndvi) - np.nanmean(pndvi)),
                "chg_ndbi_max": float(np.nanmax(_ratio(b["B11"], b["B08"]))
                                      - np.nanmax(_ratio(pb["B11"], pb["B08"]))),
                "chg_bsi_max": float(
                    np.nanmax(_ratio(b["B11"] + b["B04"], b["B08"] + b["B02"]))
                    - np.nanmax(_ratio(pb["B11"] + pb["B04"], pb["B08"] + pb["B02"]))),
                "chg_valid_frac": float(np.isfinite(p[0]).mean()),
            })
    return f


CONTEXT_FEATS = ["elev_m", "slope_deg", "d_road_m", "d_line_m", "d_industrial_m",
                 "valid_frac_s2"]


def context_features(cov: dict, mask: np.ndarray) -> dict:
    """The matched covariates, as features -- so G2.7 can measure the siting prior."""
    return {k: float(np.nanmean(cov[k][mask])) for k in CONTEXT_FEATS}


# ------------------------------------------------------------------------- controls


def shape_mask(stack: CellStack, geom_crs, r: int, c: int, rot_deg: float) -> np.ndarray:
    """The site's OWN polygon, rotated and moved to pixel (r, c).

    This is the fix for the defect the hydro study identified in its own post-mortem: its
    controls were circular discs of matched area while its positives were irregular
    polygons, and that asymmetry alone put the permutation null at 0.564 instead of 0.500.
    Matching shape as well as area removes the asymmetry at source.
    """
    x, y = stack.transform * (c + 0.5, r + 0.5)
    g = affinity.rotate(geom_crs, rot_deg, origin="centroid")
    cen = g.centroid
    g = affinity.translate(g, xoff=x - cen.x, yoff=y - cen.y)
    try:
        m = rasterio.features.rasterize(
            [(g, 1)], out_shape=stack.shape, transform=stack.transform,
            fill=0, all_touched=False, dtype="uint8").astype(bool)
    except Exception:  # noqa: BLE001 -- a degenerate translated geometry is just skipped
        return np.zeros(stack.shape, dtype=bool)
    return m


def draw_controls_tier1(stack: CellStack, cov: dict, pos: dict, geom_crs,
                        n: int, rng, calipers: dict) -> list[tuple[int, int, float]]:
    """Matched control centres + rotations for one site, or [] if the stratum is thin."""
    cand = cov["eligible"].copy()
    for k, tol in calipers.items():
        if tol is None:
            continue
        v = pos.get(k)
        if v is None or not np.isfinite(v):
            continue
        cand &= np.abs(cov[k] - v) <= tol
    for k in LOG_CALIPERS:
        v = pos.get(k)
        if v is not None and np.isfinite(v):
            cand &= np.abs(np.log1p(cov[k]) - np.log1p(v)) <= D_LOG_TOL
    if np.isfinite(pos.get("valid_frac_s2", np.nan)):
        cand &= np.abs(cov["valid_frac_s2"] - pos["valid_frac_s2"]) <= VALID_FRAC_TOL
    idx = np.flatnonzero(cand)
    if idx.size == 0:
        return []
    # Keep the footprint fully inside the raster, else it is silently clipped and its area
    # no longer matches its site's.
    minx, miny, maxx, maxy = geom_crs.bounds
    pad_px = max(2.0, max(maxx - minx, maxy - miny) / 10.0 / 2.0 + 2.0)
    out: list[tuple[int, int, float]] = []
    for flat in rng.permutation(idx)[: n * 20]:
        r, c = divmod(int(flat), stack.shape[1])
        if not (pad_px < r < stack.shape[0] - pad_px and pad_px < c < stack.shape[1] - pad_px):
            continue
        # Greedy min-separation: 20 controls from one autocorrelated patch have an
        # effective n of ~1, which would make every confidence interval a fiction.
        if any((r - rr) ** 2 + (c - cc) ** 2 < (MIN_SEP_M / 10.0) ** 2 for rr, cc, _ in out):
            continue
        out.append((r, c, float(rng.uniform(0, 360))))
        if len(out) >= n:
            break
    return out


def pick_controls_tier2(site, controls: gpd.GeoDataFrame, n: int,
                        area_tol: float) -> gpd.GeoDataFrame:
    """The k nearest eligible battery-free substations within the area caliper.

    Same-region only (a Texas site cannot borrow a Scottish control), same-cell candidates
    preferred, then anything within CTRL_MAX_DIST_M. `area_buf_m2` is the control's own
    30 m-buffered area, so the caliper compares like with like against the site's buffered
    label rather than biasing every pair toward larger substations.
    """
    c = controls[(controls.region == site.region) & controls.eligible]
    if c.empty:
        return c
    lo, hi = site.site_area_m2 * (1 - area_tol), site.site_area_m2 * (1 + area_tol)
    c = c[(c.area_buf_m2 >= lo) & (c.area_buf_m2 <= hi)]
    if c.empty:
        return c
    # Great-circle-ish metric distance is unnecessary here: at these separations a local
    # equirectangular approximation is accurate to well under a percent.
    coslat = np.cos(np.radians(site.lat))
    d = np.hypot((c.lon.values - site.lon) * coslat, c.lat.values - site.lat) * 111_320.0
    c = c.assign(_d=d)
    c = c[c._d <= CTRL_MAX_DIST_M].sort_values("_d")
    same = c[c.cell == site.cell]
    rest = c[c.cell != site.cell]
    return pd.concat([same, rest]).head(n)


# ----------------------------------------------------------------------------- main


def _load(region: str, name: str):
    p = LABELS_ROOT / f"bess_{region}" / f"{name}.parquet"
    if not p.exists():
        return None
    return gpd.read_parquet(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", type=int, choices=(1, 2), required=True)
    ap.add_argument("--run-name", default="bess_v1")
    ap.add_argument("--regions", default="gb,ca,tx")
    ap.add_argument("--n-controls", type=int, default=0,
                    help="0 = tier default (tier1 8, tier2 3)")
    ap.add_argument("--min-valid-frac", type=float, default=0.5)
    ap.add_argument("--headline-only", action="store_true", default=True,
                    help="restrict to the amendment-A1 headline set (>=20,000 m2)")
    ap.add_argument("--all-sizes", dest="headline_only", action="store_false")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max-loosen", type=int, default=len(LOOSEN_ORDER))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(a.seed)
    n_ctrl = a.n_controls or (8 if a.tier == 1 else 3)
    regions = [r.strip() for r in a.regions.split(",")]

    sites = pd.concat([s for s in (_load(r, "bess_sites") for r in regions) if s is not None],
                      ignore_index=True)
    sites = gpd.GeoDataFrame(sites, crs="EPSG:4326")
    controls = pd.concat([c for c in (_load(r, "substation_controls") for r in regions)
                          if c is not None], ignore_index=True)
    controls = gpd.GeoDataFrame(controls, crs="EPSG:4326")
    if a.headline_only:
        sites = sites[sites.headline].reset_index(drop=True)
    log.info("tier %d: %d sites, %d controls in pool (%d eligible)",
             a.tier, len(sites), len(controls), int(controls.eligible.sum()))

    ctx = {r: {k: _load(r, f"context_{k}") for k in ("roads", "lines", "industrial")}
           for r in regions}

    rows: list[dict] = []
    stacks: dict[tuple, CellStack] = {}
    stats = {"no_stack": 0, "thin": 0, "low_valid": 0, "no_feat": 0, "done": 0}
    loosened: dict[str, int] = {}

    def get_stack(aoi: str, cell: str) -> CellStack | None:
        key = (aoi, cell)
        if key not in stacks:
            if len(stacks) > 24:            # bounded LRU-ish: each stack is ~200 MB
                stacks.pop(next(iter(stacks)))
            st = CellStack(aoi, cell)
            stacks[key] = st if st.ok else None
        return stacks[key]

    for _, site in sites.iterrows():
        aoi = REGION_AOI[site.region]
        st = get_stack(aoi, site.cell)
        if st is None:
            stats["no_stack"] += 1
            continue
        c = ctx[site.region]
        cov = st.covariates(c["roads"], c["lines"], c["industrial"],
                            sites[sites.region == site.region],
                            pd.concat([controls[controls.region == site.region],
                                       sites[sites.region == site.region]], ignore_index=True))
        site_gdf = gpd.GeoDataFrame(geometry=[site.geometry], crs="EPSG:4326")
        pmask = st.rasterize(site_gdf)
        if pmask.sum() < 3:
            stats["thin"] += 1
            continue
        pf = region_features(st, pmask)
        if pf is None:
            stats["no_feat"] += 1
            continue
        pf.update(context_features(cov, pmask))
        if pf["valid_frac_s2"] < a.min_valid_frac:
            stats["low_valid"] += 1
            continue
        base = {"site_id": site.site_id, "region": site.region, "cell": site.cell,
                "cell_0p5": site.cell_0p5, "label": 1, "role": "site",
                "site_area_m2": site.site_area_m2, "core_area_m2": site.core_area_m2,
                "size_bucket": site.size_bucket, "is_hybrid": bool(site.is_hybrid),
                "capacity_mw": site.capacity_mw, "energy_mwh": site.energy_mwh,
                "construction_year": site.construction_year,
                "d_substation_m": site.d_substation_m,
                "overlaps_substation": bool(site.overlaps_substation),
                "has_pre": st.has_pre}
        # G2.4-T2: the same site, with any overlapping substation subtracted. Emitted as
        # an extra positive row (role "site_bo") rather than a separate run, so the
        # battery-only arm is scored against the identical control set. ALL substations
        # are subtracted, not just the eligible ones -- a substation excluded from the
        # control pool for being next to a battery is precisely the switchyard to remove.
        allsub = controls[controls.region == site.region]
        sub_mask = ndimage.binary_dilation(st.rasterize(allsub), iterations=3)
        bo_mask = pmask & ~sub_mask
        if bo_mask.sum() * 100.0 >= 2000.0:
            bf = region_features(st, bo_mask)
            if bf is not None:
                bf.update(context_features(cov, bo_mask))
                rows.append({**base, "role": "site_bo",
                             "battery_only_area_m2": float(bo_mask.sum() * 100.0), **bf})

        got = []
        if a.tier == 1:
            geom_crs = site_gdf.to_crs(st.crs).geometry.iloc[0]
            picks = []
            for k, extra in enumerate([{}, {"d_industrial_m": None},
                                       {"d_industrial_m": None, "d_road_m": None}]):
                if k > a.max_loosen:
                    break
                cal = {**CALIPERS, **extra}
                picks = draw_controls_tier1(st, cov, pf, geom_crs, n_ctrl, rng, cal)
                if len(picks) >= max(1, n_ctrl // 2):
                    if k:
                        loosened[LOOSEN_ORDER[k - 1]] = loosened.get(LOOSEN_ORDER[k - 1], 0) + 1
                    break
            for j, (r, cc, rot) in enumerate(picks):
                m = shape_mask(st, geom_crs, r, cc, rot)
                cf = region_features(st, m)
                if cf is None:
                    continue
                cf.update(context_features(cov, m))
                got.append({**base, "label": 0, "role": "control",
                            "ctrl_id": f"{site.site_id}-c{j}", **cf})
        else:
            picks = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            for k, tol in enumerate(AREA_TOL_ORDER):
                if k > a.max_loosen:
                    break
                picks = pick_controls_tier2(site, controls, n_ctrl, tol)
                if len(picks) >= max(1, n_ctrl // 2):
                    if k:
                        loosened[f"area_tol_{tol}"] = loosened.get(f"area_tol_{tol}", 0) + 1
                    break
            for j, (_, ctrl) in enumerate(picks.iterrows()):
                cst = get_stack(REGION_AOI[ctrl.region], ctrl.cell)
                if cst is None:
                    continue
                cgdf = gpd.GeoDataFrame(geometry=[ctrl.geometry], crs="EPSG:4326")
                # The control's label geometry is buffered by the same 30 m as the site's,
                # so the two footprints are constructed identically.
                cbuf = cgdf.to_crs(cst.crs).buffer(30.0)
                m = cst.rasterize(gpd.GeoDataFrame(geometry=cbuf, crs=cst.crs))
                if m.sum() < 3:
                    continue
                cf = region_features(cst, m)
                if cf is None:
                    continue
                cc2 = ctx[ctrl.region]
                ccov = cst.covariates(cc2["roads"], cc2["lines"], cc2["industrial"],
                                      sites[sites.region == ctrl.region],
                                      controls[controls.region == ctrl.region])
                cf.update(context_features(ccov, m))
                if cf["valid_frac_s2"] < a.min_valid_frac:
                    continue
                got.append({**base, "label": 0, "role": "control",
                            "ctrl_id": ctrl.ctrl_id, "ctrl_cell": ctrl.cell,
                            "ctrl_same_cell": bool(ctrl.cell == site.cell),
                            "ctrl_dist_m": float(ctrl._d) if hasattr(ctrl, "_d") else np.nan,
                            **cf})
        if not got:
            stats["thin"] += 1
            continue
        rows.append({**base, **pf})
        rows.extend(got)
        stats["done"] += 1
        if stats["done"] % 25 == 0:
            log.info("  %d sites matched (%d rows)", stats["done"], len(rows))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no matched samples: {stats}")
    log.info("matched %d sites -> %d rows  %s", stats["done"], len(df), stats)
    if loosened:
        log.info("caliper loosening used: %s", loosened)

    RESULTS.mkdir(parents=True, exist_ok=True)
    sp = RESULTS / f"bess_detectability_samples_tier{a.tier}.csv"
    df.assign(run_name=a.run_name, eval_version=EVAL_VERSION).to_csv(sp, index=False)

    # ---- the covariate balance table. Printed FIRST, because it is what licenses
    # reading any AUC below. A gate outcome must not be declared if matching failed.
    # role "site_bo" rows are positives too, but they are the G2.4 arm's positives, not
    # the headline's -- keeping them out of the balance table and the per-feature AUCs
    # stops one site contributing two correlated rows to the same comparison.
    pos, neg = df[df.role == "site"], df[df.label == 0]
    bal = []
    for k in CONTEXT_FEATS:
        p_, n_ = pos[k].values, neg[k].values
        varies = bool(np.nanstd(np.r_[p_, n_]) > 1e-9)
        bal.append({"covariate": k, "varies": varies,
                    "pos_mean": float(np.nanmean(p_)), "neg_mean": float(np.nanmean(n_)),
                    "smd": smd(p_, n_), "auc": rank_auc(p_, n_)})
    balt = pd.DataFrame(bal)
    balt["ok"] = (~balt.varies) | ((balt.smd.abs() <= 0.10)
                                   & balt.auc.between(0.45, 0.55))
    balt.assign(run_name=a.run_name, tier=a.tier).to_csv(
        RESULTS / f"bess_balance_tier{a.tier}.csv", index=False)

    print(f"\n=== Tier {a.tier} covariate balance (READ THIS BEFORE ANY AUC) ===")
    print(balt.round(4).to_string(index=False))
    bad = balt[~balt.ok]
    print(f"\nbalance: {'PASS' if bad.empty else 'FAIL on ' + ', '.join(bad.covariate)}")

    skip = {"site_id", "region", "cell", "cell_0p5", "label", "role", "ctrl_id",
            "battery_only_area_m2",
            "ctrl_cell", "ctrl_same_cell", "ctrl_dist_m", "size_bucket", "is_hybrid",
            "site_area_m2", "core_area_m2", "capacity_mw", "energy_mwh",
            "construction_year", "d_substation_m", "overlaps_substation", "has_pre",
            "run_name", "eval_version"}
    feats = [c for c in df.columns if c not in skip and df[c].dtype.kind in "fi"]
    per = pd.DataFrame([{"feature": f, "auc": rank_auc(pos[f].values, neg[f].values),
                         "smd": smd(pos[f].values, neg[f].values),
                         "n_pos": int(pos[f].notna().sum()),
                         "n_neg": int(neg[f].notna().sum())} for f in feats])
    per = per.sort_values("auc", ascending=False, key=lambda s: (s - 0.5).abs())
    per.assign(run_name=a.run_name, tier=a.tier).to_csv(
        RESULTS / f"bess_detectability_tier{a.tier}.csv", index=False)

    print(f"\n=== Tier {a.tier}: {stats['done']} sites, {len(neg)} controls ===")
    if stats["done"] < MIN_SITES_FOR_FEATURE_AUC:
        print("too few sites for per-feature AUC to mean anything")
    else:
        print("\ntop single features by |AUC - 0.5|:")
        print(per.head(15).round(4).to_string(index=False))

    summ = {
        "run_name": a.run_name, "tier": a.tier, "eval_version": EVAL_VERSION,
        "regions": a.regions, "headline_only": a.headline_only,
        "n_sites": stats["done"], "n_controls": len(neg),
        "n_sites_by_region": json.dumps(pos.region.value_counts().to_dict()),
        "balance_pass": bool(bad.empty),
        "balance_fail_on": ",".join(bad.covariate),
        "frac_ctrl_same_cell": (round(float(neg.ctrl_same_cell.mean()), 4)
                                if "ctrl_same_cell" in neg else ""),
        "frac_ctrl_in_range": (round(float((neg.ctrl_dist_m <= CTRL_MAX_DIST_M).mean()), 4)
                               if "ctrl_dist_m" in neg else ""),
        "frac_has_pre": round(float(pos.has_pre.mean()), 4),
        "n_site_bo": int((df.role == "site_bo").sum()),
        "skipped": json.dumps(stats), "loosened": json.dumps(loosened),
        "seed": a.seed, "generated": dt.datetime.now().isoformat(timespec="seconds"),
    }
    sfile = RESULTS / "bess_detectability_summary.csv"
    row = pd.DataFrame([summ])
    if sfile.exists():
        prev = pd.read_csv(sfile)
        prev = prev[~((prev.run_name == a.run_name) & (prev.tier == a.tier)
                      & (prev.eval_version == EVAL_VERSION))]
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(sfile, index=False)
    print(f"\nwrote {sp.name}, bess_balance_tier{a.tier}.csv, {sfile.name}")


if __name__ == "__main__":
    main()
