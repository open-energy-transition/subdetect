"""Are small hydro sites separable from *matched* terrain in S2/S1? (Stage 4)

The decisive feasibility question. If a plain feature classifier cannot tell a hydro site
complex from a terrain-matched patch of the same valley, a U-Net trained on ~159 examples
almost certainly cannot either -- and this costs no GPU to find out.

Structure follows `scripts/ehv_detectability.py`; `local_contrast` is its 31-px
uniform-filter formula, reused unchanged.

WHY THE SINDH TEMPLATES' CONTROL SAMPLING WOULD BE INVALID HERE
---------------------------------------------------------------
`ehv_detectability.py` draws controls uniformly across a cell, rejecting only pixels near
a tower. On the flat Indus plain that is fine. On a Karakoram slope the resulting controls
differ from the positives in elevation (by up to 2,000 m), slope (30 deg), aspect -- hence
winter illumination, hence whether the pixel is nodata at all -- and distance to the
nearest stream. Every one of those drives S2 brightness and S1 backscatter far harder than
the presence of a 500 m2 powerhouse does. A naive AUC would come out impressively high and
would mean nothing except "hydro plants are in valleys", which we already know.

So each site is matched to controls drawn from the same terrain stratum, on covariates
that are deliberately DISJOINT from the features under test (terrain and OSM geometry, not
reflectance), so that matching cannot destroy the signal being measured. Two constraints
the flat-terrain templates do not need:

  Same cell. Different cells were composited from different scene sets, snow states and S1
  relative orbits -- an acquisition confound no terrain covariate captures.

  Equal area. p95-brightness has an area-dependent sampling distribution, so a site of
  30,000 m2 against a single-pixel control is confounded by the window alone. Each control
  is a disc of its site's area.

The covariate balance table is therefore the real deliverable: it is what licenses reading
any of the AUCs below. It is printed first, and the summary refuses to declare a gate
outcome if the matching failed.

    pixi run -e ml python scripts/hydro_detectability.py --run-name north_pk_v1
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.mask
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subdetect.config import S1_OFFSET_DB, S1_SCALE  # noqa: E402
from subdetect.terrain import decode_dem, slope_aspect  # noqa: E402

log = logging.getLogger("hydro_detectability")

LABELS = ROOT / "data" / "labels" / "pakistan_hydro"
PROD_LABELS = ROOT / "data" / "labels" / "pakistan"
COMPOSITES = ROOT / "data" / "composites" / "pakistan" / "composites"
RESULTS = ROOT / "data" / "eval_results"
EVAL_VERSION = 1

# S2 L2A processing baseline 04.00+ carries a +1000 DN offset. optical_features.py
# subtracts it; datamodule.py does not (harmless there, since the model trained on that
# convention). For spectral indices it is NOT harmless -- omitting it biases every
# dark-target ratio badly -- and measured B02 floors of 737-1111 across these cells
# confirm the offset is present.
S2_DN_OFFSET, S2_DN_SCALE = 1000.0, 10000.0

# Matching calipers. Loosening order if strata come out too thin is fixed in advance:
# relief_1km -> rel_elev_500m -> elev_m. northness / d_river_m / valid_frac are NEVER
# loosened -- those three are the ones whose relaxation would manufacture a result.
CALIPERS = {
    "elev_m": 200.0,
    "slope_deg": 5.0,
    "rel_elev_500m": 0.15,
    "relief_1km": 200.0,
    "northness": 0.35,
}
LOOSEN_ORDER = ["relief_1km", "rel_elev_500m", "elev_m"]
D_RIVER_LOG_TOL = 0.5      # ~ +/-65% in metres
VALID_FRAC_TOL = 0.15
MIN_SEP_M = 300.0          # between accepted controls, so they are not one patch
EXCLUDE_HYDRO_M = 1000.0
EXCLUDE_INFRA_M = 500.0

S2_BAND_NAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


# --------------------------------------------------------------------------- helpers


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney rank AUC. Same estimator as ehv_detectability.rank_auc."""
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


def local_contrast(band_db: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """31-px uniform-filter local contrast, verbatim from ehv_detectability.py."""
    band = np.where(valid, band_db, np.nan)
    local = ndimage.uniform_filter(np.nan_to_num(band), 31)
    norm = ndimage.uniform_filter(valid.astype("float32"), 31)
    local = np.where(norm > 0.3, local / np.maximum(norm, 1e-6), np.nan)
    return np.nan_to_num(band - local, nan=-99.0)


def smd(pos: np.ndarray, neg: np.ndarray) -> float:
    """Standardized mean difference -- Rubin's balance diagnostic."""
    pos = np.asarray(pos, "float64")[np.isfinite(np.asarray(pos, "float64"))]
    neg = np.asarray(neg, "float64")[np.isfinite(np.asarray(neg, "float64"))]
    if pos.size < 2 or neg.size < 2:
        return float("nan")
    denom = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2.0)
    return float((pos.mean() - neg.mean()) / denom) if denom > 0 else float("nan")


def _idx(index: str) -> None:  # documentation anchor, keeps ruff quiet about unused
    return None


# ------------------------------------------------------------------- per-cell rasters


class CellStack:
    """Everything needed from one 0.1 deg cell, on its own metric grid.

    At 10 m/px in a UTM grid, pixel distance IS metric distance, so
    distance_transform_edt * 10 gives metres -- the argument tower_prior.py already makes,
    and it avoids EPSG:6933's ~6% anisotropy at this latitude.
    """

    def __init__(self, cell: str):
        d = COMPOSITES / cell
        self.cell = cell
        self.ok = False
        s2p, s1p, demp = d / "composite_0.tif", d / "composite_s1.tif", d / "composite_dem.tif"
        if not (s2p.exists() and s1p.exists() and demp.exists()):
            return
        with rasterio.open(s2p) as src:
            self.s2 = src.read().astype("float32")
            self.transform, self.crs, self.shape = src.transform, src.crs, (src.height, src.width)
        with rasterio.open(s1p) as src:
            s1dn = src.read().astype("float32")
        with rasterio.open(demp) as src:
            dem_dn_arr = src.read()
        self.s2_valid = self.s2[0] > 0
        self.refl = np.where(self.s2_valid, (self.s2 - S2_DN_OFFSET) / S2_DN_SCALE, np.nan)
        self.s1_valid = s1dn[0] > 0
        self.s1_db = np.where(self.s1_valid, s1dn / S1_SCALE - S1_OFFSET_DB, np.nan)
        self.con_vv = local_contrast(self.s1_db[0], self.s1_valid)
        self.con_vh = local_contrast(self.s1_db[1], self.s1_valid)
        self.dem, self.dem_valid = decode_dem(dem_dn_arr)
        _, aspect = slope_aspect(np.nan_to_num(self.dem[0], nan=0.0), px=10.0)
        self.northness = np.cos(np.radians(aspect)).astype("float32")
        self.ok = True

    def rasterize(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        if gdf is None or gdf.empty:
            return np.zeros(self.shape, dtype=bool)
        g = gdf.to_crs(self.crs)
        g = g[~g.geometry.is_empty & g.geometry.notna()]
        if g.empty:
            return np.zeros(self.shape, dtype=bool)
        return rasterio.features.rasterize(
            [(x, 1) for x in g.geometry], out_shape=self.shape, transform=self.transform,
            fill=0, all_touched=True, dtype="uint8").astype(bool)

    def dist_m(self, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.full(self.shape, 1e6, dtype="float32")
        return (ndimage.distance_transform_edt(~mask) * 10.0).astype("float32")

    def covariates(self, rivers, hydro, infra) -> dict:
        s2_frac = ndimage.uniform_filter(self.s2_valid.astype("float32"), 31)
        cov = {
            "elev_m": self.dem[0], "slope_deg": self.dem[1],
            "rel_elev_500m": self.dem[2], "relief_1km": self.dem[3],
            "northness": self.northness,
            "d_river_m": self.dist_m(self.rasterize(rivers)),
            "valid_frac_s2": s2_frac,
        }
        cov["eligible"] = (
            self.s2_valid & self.s1_valid & self.dem_valid
            & (self.dist_m(self.rasterize(hydro)) >= EXCLUDE_HYDRO_M)
            & (self.dist_m(self.rasterize(infra)) >= EXCLUDE_INFRA_M)
        )
        return cov


# ------------------------------------------------------------------------- features


def region_features(stack: CellStack, mask: np.ndarray) -> dict | None:
    """S2 + S1 + terrain statistics over one region (site footprint or control disc)."""
    if mask.sum() < 3:
        return None
    r = stack.refl[:, mask]
    if not np.isfinite(r).any():
        return None
    b = {n: r[i] for i, n in enumerate(S2_BAND_NAMES)}

    def ratio(x, y):
        with np.errstate(invalid="ignore", divide="ignore"):
            return (x - y) / (x + y)

    bright = np.nanmean(r[:3], axis=0)
    ndvi = ratio(b["B08"], b["B04"])
    f = {f"{n.lower()}_mean": float(np.nanmean(b[n])) for n in S2_BAND_NAMES}
    f.update({
        "bright_mean": float(np.nanmean(bright)),
        "bright_p95": float(np.nanpercentile(bright, 95)),
        "ndvi_mean": float(np.nanmean(ndvi)), "ndvi_min": float(np.nanmin(ndvi)),
        "ndwi_max": float(np.nanmax(ratio(b["B03"], b["B08"]))),
        "mndwi_max": float(np.nanmax(ratio(b["B03"], b["B11"]))),   # turbid alpine water
        "ndbi_max": float(np.nanmax(ratio(b["B11"], b["B08"]))),    # built surface
        "bsi_max": float(np.nanmax(ratio(b["B11"] + b["B04"], b["B08"] + b["B02"]))),
        "valid_frac_s2": float(np.isfinite(r[0]).mean()),
    })
    # Ring contrast: the area-generalized form of isolated_tower_prior's local contrast.
    ring = ndimage.binary_dilation(mask, iterations=15) & ~ndimage.binary_dilation(mask, iterations=3)
    if ring.any():
        rb = np.nanmean(stack.refl[:3][:, ring], axis=0)
        f["bright_ring_contrast"] = float(np.nanmean(bright) - np.nanmean(rb))
    else:
        f["bright_ring_contrast"] = np.nan
    bmap = np.full(stack.shape, np.nan, "float32")
    bmap[mask] = bright
    f["tex_std"] = float(np.nanstd(bright))

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
    # slope_anomaly is terrain-derived and correlates with the matched covariates, so a
    # high AUC here is weaker evidence than the same AUC on an S1 feature. Reported
    # separately for that reason.
    sl = stack.dem[1]
    f["slope_anomaly"] = float(np.nanmean(ndimage.uniform_filter(np.nan_to_num(sl), 31)[mask] - sl[mask]))
    for i, n in enumerate(["elev_m", "slope_deg", "rel_elev_500m", "relief_1km"]):
        f[n] = float(np.nanmean(stack.dem[i][mask]))
    f["northness"] = float(np.nanmean(stack.northness[mask]))
    return f


TERRAIN_FEATS = ["elev_m", "slope_deg", "rel_elev_500m", "relief_1km", "northness",
                 "d_river_m", "valid_frac_s2"]


def draw_controls(stack: CellStack, cov: dict, pos: dict, area_m2: float,
                  n: int, rng, calipers: dict) -> list[tuple[int, int]]:
    """Terrain-matched control pixel centres for one site, or [] if the stratum is thin."""
    cand = cov["eligible"].copy()
    for k, tol in calipers.items():
        v = pos.get(k)
        if v is None or not np.isfinite(v):
            continue
        cand &= np.abs(cov[k] - v) <= tol
    if np.isfinite(pos.get("d_river_m", np.nan)):
        cand &= np.abs(np.log1p(cov["d_river_m"]) - np.log1p(pos["d_river_m"])) <= D_RIVER_LOG_TOL
    if np.isfinite(pos.get("valid_frac_s2", np.nan)):
        cand &= np.abs(cov["valid_frac_s2"] - pos["valid_frac_s2"]) <= VALID_FRAC_TOL
    idx = np.flatnonzero(cand)
    if idx.size == 0:
        return []
    rad_px = max(1.0, np.sqrt(area_m2 / np.pi) / 10.0)
    # Keep discs fully inside the raster, else the region is silently clipped and its area
    # no longer matches its site's.
    pick = rng.permutation(idx)[: n * 10]
    out: list[tuple[int, int]] = []
    for flat in pick:
        r, c = divmod(int(flat), stack.shape[1])
        if not (rad_px < r < stack.shape[0] - rad_px and rad_px < c < stack.shape[1] - rad_px):
            continue
        # Greedy min-separation: 20 controls from one autocorrelated patch have an
        # effective n of ~1, which would make every confidence interval a fiction.
        if any((r - rr) ** 2 + (c - cc) ** 2 < (MIN_SEP_M / 10.0) ** 2 for rr, cc in out):
            continue
        out.append((r, c))
        if len(out) >= n:
            break
    return out


def disc_mask(shape, r: int, c: int, area_m2: float) -> np.ndarray:
    rad = max(1.0, np.sqrt(area_m2 / np.pi) / 10.0)
    rr, cc = np.ogrid[: shape[0], : shape[1]]
    return ((rr - r) ** 2 + (cc - c) ** 2) <= rad ** 2


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="north_pk_v1")
    ap.add_argument("--sites", type=Path, default=LABELS / "hydro_sites.parquet")
    ap.add_argument("--n-controls", type=int, default=20)
    ap.add_argument("--min-valid-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max-loosen", type=int, default=len(LOOSEN_ORDER))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(a.seed)

    sites = gpd.read_parquet(a.sites)
    sites = sites[sites.has_composite & sites.has_s1].copy()
    log.info("%d sites with S2+S1 composites", len(sites))

    rivers = None
    for p in [LABELS / "waterways.parquet", LABELS / "hydro_conduit.parquet"]:
        if p.exists():
            rivers = gpd.read_parquet(p)[["geometry"]]
            break
    hydro_all = gpd.GeoDataFrame(sites[["geometry"]], crs="EPSG:4326")
    infra = []
    for f in ["substations_poly.parquet", "substations_node.parquet", "plants.parquet"]:
        p = PROD_LABELS / f
        if p.exists():
            infra.append(gpd.read_parquet(p)[["geometry"]])
    infra = gpd.GeoDataFrame(pd.concat(infra, ignore_index=True), crs="EPSG:4326") if infra else None

    rows, loosened_at = [], {}
    for cell, grp in sites.groupby("cell"):
        stack = CellStack(cell)
        if not stack.ok:
            log.info("cell %s: missing a composite layer, skipped", cell)
            continue
        cov = stack.covariates(rivers, hydro_all, infra)
        for r in grp.itertuples():
            m = stack.rasterize(gpd.GeoDataFrame({"geometry": [r.geometry]}, crs="EPSG:4326"))
            fs = region_features(stack, m)
            if fs is None or fs["valid_frac_s2"] < a.min_valid_frac:
                continue
            fs.update({"d_river_m": float(np.nanmean(cov["d_river_m"][m]))})
            base = {"site_id": r.site_id, "cell": cell, "cluster": r.cluster,
                    "size_bucket": r.size_bucket, "core_area_m2": r.core_area_m2,
                    "capacity_mw": r.capacity_mw, "has_penstock": r.has_penstock,
                    "conduit_tunnel_frac": r.conduit_tunnel_frac, "split": r.split,
                    "area_m2": r.site_area_m2, "lon": r.lon, "lat": r.lat}
            rows.append({**base, "is_control": False, **fs})

            cal = dict(CALIPERS)
            ctrls = draw_controls(stack, cov, fs, r.site_area_m2, a.n_controls, rng, cal)
            for k in LOOSEN_ORDER[: a.max_loosen]:
                if len(ctrls) >= 5:
                    break
                cal[k] *= 2.0                       # documented, pre-registered order
                loosened_at[r.site_id] = loosened_at.get(r.site_id, 0) + 1
                ctrls = draw_controls(stack, cov, fs, r.site_area_m2, a.n_controls, rng, cal)
            for (rr, cc) in ctrls:
                cm = disc_mask(stack.shape, rr, cc, r.site_area_m2)
                cf = region_features(stack, cm)
                if cf is None or cf["valid_frac_s2"] < a.min_valid_frac:
                    continue
                cf["d_river_m"] = float(np.nanmean(cov["d_river_m"][cm]))
                x, y = rasterio.transform.xy(stack.transform, rr, cc)
                lon, lat = gpd.GeoSeries([gpd.points_from_xy([x], [y])[0]],
                                         crs=stack.crs).to_crs("EPSG:4326").iloc[0].coords[0]
                rows.append({**base, "is_control": True, "lon": lon, "lat": lat, **cf})
        log.info("cell %s: %d sites -> %d rows so far", cell, len(grp), len(rows))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no samples -- are the DEM composites built? (compose --sensor dem)")
    df.insert(0, "run_name", a.run_name)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "hydro_detectability_samples.csv", index=False)

    pos = df[~df.is_control]
    neg = df[df.is_control]
    matched = set(neg.site_id)
    print("\n=== Stage 4: hydro site vs terrain-matched control ===")
    print(f"sites with features: {len(pos)}   with >=1 control: {len(matched)}   "
          f"controls: {len(neg)}   median controls/site: "
          f"{neg.groupby('site_id').size().median() if len(neg) else 0:.0f}")
    print(f"sites needing a loosened caliper: {len(loosened_at)}")

    # ---- balance table FIRST: it licenses everything below it.
    bal = []
    for k in TERRAIN_FEATS:
        if k not in df.columns:
            continue
        bal.append({"covariate": k, "smd": smd(pos[k].values, neg[k].values),
                    "auc": rank_auc(pos[k].values, neg[k].values),
                    "median_site": float(pos[k].median()), "median_ctrl": float(neg[k].median())})
    balance = pd.DataFrame(bal)
    balance["balanced"] = (balance.smd.abs() <= 0.10) & balance.auc.between(0.45, 0.55)
    print("\n--- covariate balance (must pass before any AUC below is meaningful) ---")
    print(balance.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    balance.assign(run_name=a.run_name).to_csv(RESULTS / "hydro_balance.csv", index=False)
    ok_balance = bool(balance.balanced.all())
    if not ok_balance:
        print("  !! unbalanced covariates: any S2/S1 AUC below is partly those covariates.")

    feat_cols = [c for c in df.columns if c not in
                 {"run_name", "site_id", "cell", "cluster", "size_bucket", "split",
                  "is_control", "lon", "lat", "area_m2", "core_area_m2", "capacity_mw",
                  "has_penstock", "conduit_tunnel_frac"}]
    # Within-site AUC then pooled: pooling all sites against all controls would
    # reintroduce exactly the between-site terrain variance the matching removes.
    per_feat = []
    for f in feat_cols:
        aucs = [rank_auc(pos.loc[pos.site_id == s, f].values, neg.loc[neg.site_id == s, f].values)
                for s in matched]
        aucs = np.array([x for x in aucs if np.isfinite(x)])
        if aucs.size < 10:
            continue
        per_feat.append({"feature": f, "n_sites": aucs.size, "auc": float(aucs.mean()),
                         "auc_pooled": rank_auc(pos[f].values, neg[f].values),
                         "smd": smd(pos[f].values, neg[f].values)})
    pf = pd.DataFrame(per_feat).sort_values("auc", key=lambda s: (s - 0.5).abs(), ascending=False)
    print("\n--- top per-feature AUC (within-site, then averaged) ---")
    print(pf.head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    summ = {"run_name": a.run_name, "eval_version": EVAL_VERSION, "seed": a.seed,
            "n_sites": len(pos), "n_sites_matched": len(matched), "n_controls": len(neg),
            "balance_ok": ok_balance, "n_loosened": len(loosened_at),
            "best_feature": pf.iloc[0].feature if len(pf) else "",
            "best_feature_auc": float(pf.iloc[0].auc) if len(pf) else np.nan,
            "generated": dt.datetime.now().isoformat(timespec="seconds")}
    out = RESULTS / "hydro_detectability.csv"
    pf.assign(run_name=a.run_name, eval_version=EVAL_VERSION).to_csv(out, index=False)
    outs = RESULTS / "hydro_detectability_summary.csv"
    row = pd.DataFrame([summ])
    if outs.exists():
        prev = pd.read_csv(outs)
        prev = prev[~((prev.run_name == a.run_name) & (prev.eval_version == EVAL_VERSION))]
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(outs, index=False)

    print(f"\nG2.0 n: {len(matched)} matched sites, {len(neg)} controls "
          f"(need >=80 and >=240) -> {'PASS' if len(matched) >= 80 and len(neg) >= 240 else 'FAIL'}")
    print(f"balance: {'PASS' if ok_balance else 'FAIL'}")
    print(f"\nwrote {out.relative_to(ROOT)}, hydro_balance.csv, hydro_detectability_samples.csv")
    print("Multivariate GroupKFold + permutation null + holdout: run "
          "hydro_stage4_model.py on the samples CSV once the balance table passes.")


if __name__ == "__main__":
    main()
