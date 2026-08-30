"""Does the existing substation model already fire at small hydro sites? (Stage 1)

Free probe: reads probability rasters already on disk (`data/predictions_v9_fused/
pakistan/prob_mean`, 1,564 cells covering the whole Pakistan ROI including the north) and
asks what they say inside the hydro site complexes built by `build_hydro_labels.py`.
No GPU, no STAC, no training.

Worth doing first because it is nearly free and because a *positive* answer would change
the plan entirely: many small hydro plants have a co-located switchyard, so if v9 already
lights them up, the cheapest product is a re-ranking of existing candidates rather than a
new model.

**A null result here is expected and is NOT evidence against feasibility.** Hydro
`power=plant` polygons are burned as ignore (-1) in every substation chip mask
(`local_source.load_substation_labels` -> role "plant", `chips._burn_mask`), and
`min_sub_area_m2` was 20,000 m2 -- roughly 37x the median hydro powerhouse -- in a
training domain (Indus plain + Indian Punjab) containing no mountain terrain at all. The
model was never taught this class as either positive or negative. So this is a null probe:
it can only kill the free-lunch route, not the study.

Two guards make the numbers mean something:

  *Matched controls.* A bare "p_max = 0.03" is uninterpretable. Scored against
  terrain-matched control patches (from `hydro_detectability.py`, via --controls) it
  becomes a rank-AUC. Without controls the script still runs and reports the raw
  distribution, flagged as such.

  *A buffer sweep.* p_max over a larger polygon is mechanically larger, so a "signal" that
  grows with the buffer is an artefact. The site footprint is rebuilt from its components
  at 25/75/150 m and all three are reported, alongside p90/p_mean which are buffer-robust
  where p_max is not.

Sites are also split on whether a mapped substation sits within 300 m: any site that does
fire is far more likely to be firing on its switchyard than on its powerhouse, and without
that column the tail cannot be read.

    pixi run python scripts/hydro_prob_probe.py
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
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_missed_large import _prob_stats, _raster_index  # noqa: E402  (label-agnostic)
from subdetect.config import geodesic_area_m2  # noqa: E402

log = logging.getLogger("hydro_prob_probe")

LABELS = ROOT / "data" / "labels" / "pakistan_hydro"
PROD_LABELS = ROOT / "data" / "labels" / "pakistan"
PROB_DIR = ROOT / "data" / "predictions_v9_fused" / "pakistan" / "prob_mean"
RESULTS = ROOT / "data" / "eval_results"
EVAL_VERSION = 1
BUFFERS = (25.0, 75.0, 150.0)
SWITCHYARD_M = 300.0


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney rank AUC, ties averaged. Same estimator as ehv_detectability.py."""
    pos = np.asarray(pos, "float64")
    neg = np.asarray(neg, "float64")
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def bootstrap_ci(pos: np.ndarray, neg: np.ndarray, n: int = 1000, seed: int = 0):
    """95% CI on the AUC, resampling SITES (not rows) -- at n~159 this is wide."""
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, "float64"), np.asarray(neg, "float64")
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size < 5 or neg.size < 5:
        return float("nan"), float("nan")
    vals = [rank_auc(rng.choice(pos, pos.size, replace=True),
                     rng.choice(neg, neg.size, replace=True)) for _ in range(n)]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def site_geoms_at_buffer(comps: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    """Rebuild each site footprint from its stored components at a different buffer.

    Rebuilt rather than re-buffering the stored 75 m footprint: buffer(-50) on a dissolved
    multi-part shape can collapse thin conduit arms entirely, which would change *which*
    parts of the site are measured rather than just how much margin they carry.
    """
    rows = []
    for sid, grp in comps.groupby("site_id"):
        utm = grp.estimate_utm_crs()
        g = unary_union([x.buffer(buffer_m) for x in grp.to_crs(utm).geometry]).buffer(0)
        g4326 = gpd.GeoSeries([g], crs=utm).to_crs("EPSG:4326").iloc[0]
        rows.append({"site_id": sid, "geometry": g4326, "area_m2": geodesic_area_m2(g4326)})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def flag_switchyards(sites: gpd.GeoDataFrame) -> pd.Series:
    """True where a mapped OSM substation sits within SWITCHYARD_M of the site."""
    try:
        poly = gpd.read_parquet(PROD_LABELS / "substations_poly.parquet")[["geometry"]]
        node = gpd.read_parquet(PROD_LABELS / "substations_node.parquet")[["geometry"]]
    except FileNotFoundError:
        return pd.Series(False, index=sites.index)
    subs = gpd.GeoDataFrame(pd.concat([poly, node], ignore_index=True), crs="EPSG:4326")
    eq = "EPSG:6933"
    subs_eq = subs.to_crs(eq)
    sites_eq = sites.to_crs(eq)
    near = sites_eq.geometry.buffer(SWITCHYARD_M)
    tree = subs_eq.sindex
    return pd.Series([len(tree.query(g, predicate="intersects")) > 0 for g in near],
                     index=sites.index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", type=Path, default=LABELS / "hydro_sites.parquet")
    ap.add_argument("--components", type=Path, default=LABELS / "hydro_sites_components.parquet")
    ap.add_argument("--prob-dir", type=Path, default=PROB_DIR)
    ap.add_argument("--controls", type=Path, default=RESULTS / "hydro_detectability_samples.csv",
                    help="matched controls from hydro_detectability.py; optional")
    ap.add_argument("--run-name", default="v9_meanfusion_north")
    ap.add_argument("--bbox", default="70.5,33.0,77.5,37.5")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sites = gpd.read_parquet(a.sites)
    comps = gpd.read_parquet(a.components)
    log.info("%d hydro sites, %d components", len(sites), len(comps))

    log.info("indexing probability rasters in %s", a.prob_dir)
    rindex = _raster_index(a.prob_dir)
    x0, y0, x1, y1 = (float(v) for v in a.bbox.split(","))
    rindex = rindex.cx[x0:x1, y0:y1]
    log.info("%d rasters overlap the study window", len(rindex))
    if rindex.empty:
        raise SystemExit(f"no probability rasters in {a.prob_dir} within {a.bbox}")

    sites = sites.assign(has_switchyard_300m=flag_switchyards(sites).values)

    rows = []
    for bm in BUFFERS:
        geoms = site_geoms_at_buffer(comps, bm).merge(
            sites.drop(columns=["geometry"]), on="site_id", how="left")
        for r in geoms.itertuples():
            st = _prob_stats(r.geometry, rindex)
            rows.append({"site_id": r.site_id, "buffer_m": bm, "is_control": False,
                         "area_m2": r.area_m2, "cluster": getattr(r, "cluster", ""),
                         "size_bucket": getattr(r, "size_bucket", ""),
                         "core_area_m2": getattr(r, "core_area_m2", np.nan),
                         "capacity_mw": getattr(r, "capacity_mw", np.nan),
                         "has_penstock": getattr(r, "has_penstock", False),
                         "has_switchyard_300m": getattr(r, "has_switchyard_300m", False),
                         "split": getattr(r, "split", ""), **st})
        log.info("buffer %.0f m done", bm)

    ctrl = None
    if a.controls and Path(a.controls).exists():
        cdf = pd.read_csv(a.controls)
        cdf = cdf[cdf.get("is_control", False) == True]  # noqa: E712
        if len(cdf) and {"lon", "lat", "area_m2"} <= set(cdf.columns):
            log.info("scoring %d matched controls from %s", len(cdf), a.controls.name)
            eq = "EPSG:6933"
            pts = gpd.GeoDataFrame(cdf, geometry=gpd.points_from_xy(cdf.lon, cdf.lat),
                                   crs="EPSG:4326").to_crs(eq)
            discs = pts.geometry.buffer(np.sqrt(cdf.area_m2.values / np.pi))
            discs = gpd.GeoSeries(discs, crs=eq).to_crs("EPSG:4326")
            for (_, c), g in zip(cdf.iterrows(), discs):
                st = _prob_stats(g, rindex)
                rows.append({"site_id": c.get("site_id", ""), "buffer_m": 75.0,
                             "is_control": True, "area_m2": c.get("area_m2", np.nan),
                             "cluster": c.get("cluster", ""), **st})
            ctrl = True

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.insert(0, "run_name", a.run_name)
    df.to_csv(RESULTS / "hydro_prob_probe.csv", index=False)

    # ---- report
    print("\n=== Stage 1: does the v9 substation model fire at hydro sites? ===")
    print(f"prob dir: {a.prob_dir.relative_to(ROOT)}   rasters in window: {len(rindex)}")
    summ = []
    for bm in BUFFERS:
        s = df[(df.buffer_m == bm) & (~df.is_control)]
        cov = s[s.covered]
        n_cov, n_unc = len(cov), int((~s.covered).sum())
        frac04 = float((cov.p_max >= 0.4).mean()) if len(cov) else float("nan")
        auc = auc_lo = auc_hi = float("nan")
        if ctrl:
            c = df[(df.is_control) & (df.covered)]
            auc = rank_auc(cov.p_max.values, c.p_max.values)
            auc_lo, auc_hi = bootstrap_ci(cov.p_max.values, c.p_max.values)
        summ.append({"buffer_m": bm, "n_covered": n_cov, "n_no_coverage": n_unc,
                     "median_p_max": float(cov.p_max.median()) if n_cov else np.nan,
                     "median_p90": float(cov.p90.median()) if n_cov else np.nan,
                     "frac_p_max_ge_0.4": frac04, "auc_vs_control": auc,
                     "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
                     "frac_silent_p_max_lt_0.05": float((cov.p_max < 0.05).mean()) if n_cov else np.nan})
    st = pd.DataFrame(summ)
    print("\n" + st.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  n_no_coverage is reported separately and never enters a denominator "
          "(the eval_version-2 lesson).")

    main_row = st[st.buffer_m == 75.0].iloc[0]
    n_cov = int(main_row.n_covered)
    print("\n--- gates (75 m footprint) ---")
    g10 = n_cov >= 60
    print(f"  {'PASS' if g10 else 'FAIL'}  G1.0 power        {n_cov} sites covered (need >=60)")
    if ctrl and np.isfinite(main_row.auc_vs_control):
        auc = main_row.auc_vs_control
        print(f"        site-vs-matched-control AUC on p_max: {auc:.3f} "
              f"[{main_row.auc_ci_lo:.3f}, {main_row.auc_ci_hi:.3f}]")
        rise = float(st[st.buffer_m == 150.0].auc_vs_control.iloc[0] - st[st.buffer_m == 25.0].auc_vs_control.iloc[0])
        g13 = abs(rise) <= 0.10
        print(f"  {'PASS' if g13 else 'FAIL'}  G1.3 buffer ctrl  AUC moves {rise:+.3f} across 25->150 m")
        if auc >= 0.70 and main_row["frac_p_max_ge_0.4"] >= 0.25:
            verdict = "G1.1 PASS -- v9 partially sees hydro sites; consider re-ranking before training"
        elif auc >= 0.60:
            verdict = "G1.2 WEAK -- rerun with switchyards masked before believing this"
        elif auc < 0.50:
            verdict = "G1.FATAL -- controls outscore sites; terrain may be suppressing the signature"
        else:
            verdict = "G1.4 NULL (expected) -- v9 is blind to this class; proceed to Stage 4"
    else:
        verdict = ("no matched controls yet -- run hydro_detectability.py first for an AUC. "
                   "Raw distribution above is descriptive only.")
    print(f"\n  {main_row['frac_silent_p_max_lt_0.05']:.0%} of covered sites are effectively "
          f"silent (p_max < 0.05); {main_row['frac_p_max_ge_0.4']:.0%} reach the 0.4 hysteresis seed")
    print("\nby switchyard proximity (75 m, covered):")
    s75 = df[(df.buffer_m == 75.0) & (~df.is_control) & (df.covered)]
    if len(s75):
        print(s75.groupby("has_switchyard_300m").p_max.agg(["count", "median", "max"]).to_string())
    print(f"\nVERDICT: {verdict}")

    st.insert(0, "run_name", a.run_name)
    st["eval_version"] = EVAL_VERSION
    st["generated"] = dt.datetime.now().isoformat(timespec="seconds")
    out = RESULTS / "hydro_prob_probe_summary.csv"
    if out.exists():
        prev = pd.read_csv(out)
        prev = prev[~((prev.run_name == a.run_name) & (prev.eval_version == EVAL_VERSION))]
        st = pd.concat([prev, st], ignore_index=True)
    st.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)} and hydro_prob_probe.csv")


if __name__ == "__main__":
    main()
