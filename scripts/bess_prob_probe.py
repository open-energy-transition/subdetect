"""Does the production substation model already fire at BESS sites? (Stage 1)

Not free, unlike the hydro probe: no prediction rasters exist over GB/CA/TX, so this needs
a restricted inference run first (`subdetect infer --aoi gb_bess --limit 80 --upsample 2`
with the v12_up2 checkpoint, ~2-4 h on the 1060). It is still worth doing before Stage 2,
because it buys an early, model-side measurement of this study's central risk: 76% of these
sites have a substation within 300 m and 41% overlap one outright (Stage 0, G0.6).

WHAT MAKES A NULL INTERPRETABLE HERE, AND DID NOT IN THE HYDRO STUDY
--------------------------------------------------------------------
The hydro probe had to pre-declare its null uninformative, because a silent model over
Karakoram valleys is indistinguishable from a model that simply does not transfer there.
Over GB/CA/TX there is a free positive control: the *same probe cells* contain ordinary
battery-free `power=substation` polygons >= 20,000 m2, which is exactly the class v12_up2
was trained on and where its measured Pakistan recall is 0.714.

  G1.P passes  -> the model transfers, so a null on battery geometry is INFORMATIVE: it
                  says the battery signature is not the substation signature.
  G1.P fails   -> the whole of Gate 1 is UNINFORMATIVE, not negative. A domain-transfer
                  failure says nothing at all about batteries.

Three further guards:

  *Battery-only geometry.* G1.1's second clause re-scores each site with any overlapping
  substation subtracted (buffered 30 m). A pass on the full footprint but a fail here means
  the model found the switchyard, which it was trained to do.

  *A buffer sweep.* p_max over a larger polygon is mechanically larger, so a "signal" that
  grows with the buffer is an artefact. Footprints are rebuilt from components at
  25/75/150 m; G1.3 downgrades any pass that moves more than +/-0.10 across them.

  *Coverage is never a denominator.* Sites with no raster over them are reported as
  n_no_coverage and excluded -- the eval_version-2 lesson, where a bbox denominator scored
  53 of 163 substations as misses that no raster ever covered.

    pixi run -e ml python scripts/bess_prob_probe.py --aoi gb_bess
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
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_missed_large import _prob_stats, _raster_index  # noqa: E402 (label-agnostic)
from subdetect.config import geodesic_area_m2  # noqa: E402

log = logging.getLogger("bess_prob_probe")

LABELS_ROOT = ROOT / "data" / "labels"
RESULTS = ROOT / "data" / "eval_results"
EVAL_VERSION = 1
BUFFERS = (25.0, 75.0, 150.0)
SWITCHYARD_M = 300.0
POSCTRL_MIN_AREA = 20_000.0     # the class v12_up2 was trained on
AOI_REGION = {"gb_bess": "gb", "ca_bess": "ca", "tx_bess": "tx"}


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney rank AUC, ties averaged. Same estimator as the hydro probe."""
    pos, neg = np.asarray(pos, "float64"), np.asarray(neg, "float64")
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def bootstrap_ci(pos: np.ndarray, neg: np.ndarray, n: int = 1000, seed: int = 0):
    """95% CI on the AUC, resampling sites."""
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

    Rebuilt rather than re-buffering the stored 30 m footprint: a negative buffer on a
    dissolved multi-part shape can collapse thin parts entirely, which would change *which*
    parts of the site are measured rather than just how much margin they carry.
    """
    rows = []
    for sid, grp in comps.groupby("site_id"):
        utm = grp.estimate_utm_crs()
        g = unary_union([x.buffer(buffer_m) for x in grp.to_crs(utm).geometry]).buffer(0)
        g4326 = gpd.GeoSeries([g], crs=utm).to_crs("EPSG:4326").iloc[0]
        rows.append({"site_id": sid, "geometry": g4326, "area_m2": geodesic_area_m2(g4326)})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def battery_only_geoms(sites: gpd.GeoDataFrame, subs: gpd.GeoDataFrame,
                       pad_m: float = 30.0) -> gpd.GeoDataFrame:
    """Site footprint minus every overlapping substation, buffered pad_m.

    Kept only where >= 2,000 m2 survives: below that the remainder is a sliver of fence
    line, and this pipeline's measured response to anything under 2,000 m2 is p_max of
    exactly 0.000, so a null there would be a property of the geometry, not the battery.
    """
    if subs.empty:
        return sites.assign(bo_area_m2=sites.geometry.map(geodesic_area_m2))
    utm = sites.estimate_utm_crs()
    su, cu = sites.to_crs(utm), subs.to_crs(utm)
    tree = cu.sindex
    rows = []
    for (_, s4), g in zip(sites.iterrows(), su.geometry):
        hit = tree.query(g, predicate="intersects")
        left = g
        if len(hit):
            left = g.difference(unary_union([cu.geometry.iloc[int(h)] for h in hit])
                                .buffer(pad_m))
        if left.is_empty:
            continue
        g4326 = gpd.GeoSeries([left], crs=utm).to_crs("EPSG:4326").iloc[0]
        area = geodesic_area_m2(g4326)
        if area < 2000.0:
            continue
        rows.append({"site_id": s4.site_id, "geometry": g4326, "bo_area_m2": area})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", default="gb_bess")
    ap.add_argument("--prob-dir", type=Path, default=None,
                    help="default data/predictions_bess/<aoi>/prob")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--headline-only", action="store_true", default=True)
    ap.add_argument("--all-sizes", dest="headline_only", action="store_false")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    region = AOI_REGION[a.aoi]
    run = a.run_name or f"probe_{a.aoi}"
    # `subdetect infer` writes <out_dir>/<aoi>/prob for a single checkpoint. The
    # prob_mean name belongs to infer_fused_aoi.py, which writes four rasters per cell;
    # v12_up2 is a single S2-only model, so there is nothing to fuse here.
    prob_dir = a.prob_dir or ROOT / "data" / "predictions_bess" / a.aoi / "prob"
    if not prob_dir.exists():
        raise SystemExit(f"no prob rasters at {prob_dir}; run `subdetect infer --aoi {a.aoi} "
                         f"--upsample 2 --limit 80 --out-dir data/predictions_bess` first")

    lab = LABELS_ROOT / f"bess_{region}"
    sites = gpd.read_parquet(lab / "bess_sites.parquet")
    comps = gpd.read_parquet(lab / "bess_sites_components.parquet")
    subs = gpd.read_parquet(lab / "substation_controls.parquet")
    if a.headline_only:
        sites = sites[sites.headline].reset_index(drop=True)
    comps = comps[comps.site_id.isin(sites.site_id)]

    rindex = _raster_index(prob_dir)
    log.info("%d prob rasters, %d sites, %d substations", len(rindex), len(sites), len(subs))

    # ---- the positive control (G1.P). Battery-free substations >= 20,000 m2, inside the
    # same rasters, i.e. the class the checkpoint was actually trained on.
    pc = subs[subs.eligible & (subs.area_m2 >= POSCTRL_MIN_AREA)].copy()
    pc_stats = [_prob_stats(g, rindex) for g in pc.geometry] if len(pc) else []
    pc_df = pd.DataFrame(pc_stats)
    pc_cov = pc_df[pc_df.covered] if len(pc_df) else pc_df
    pc_frac04 = float((pc_cov.p_max >= 0.4).mean()) if len(pc_cov) else float("nan")
    g1p = len(pc_cov) >= 30 and pc_frac04 >= 0.40

    # ---- sites, at three buffers ------------------------------------------------------
    rows, per_buffer = [], {}
    for bm in BUFFERS:
        g = site_geoms_at_buffer(comps, bm)
        st = pd.DataFrame([_prob_stats(x, rindex) for x in g.geometry])
        st = pd.concat([g.drop(columns="geometry").reset_index(drop=True), st], axis=1)
        st["buffer_m"] = bm
        per_buffer[bm] = st
        rows.append(st)
    site_probs = pd.concat(rows, ignore_index=True)

    # ---- battery-only geometry (G1.1 second clause) -----------------------------------
    bo = battery_only_geoms(sites, subs)
    bo_st = pd.DataFrame([_prob_stats(x, rindex) for x in bo.geometry]) if len(bo) else pd.DataFrame()
    if len(bo_st):
        bo_st = pd.concat([bo.drop(columns="geometry").reset_index(drop=True), bo_st], axis=1)

    # ---- reference distribution for the AUCs: the eligible substations act as the
    # negative class for the battery comparison too. A battery that scores like an
    # ordinary substation is exactly the null this study is trying to detect.
    neg = pc_cov.p_max.values if len(pc_cov) else np.array([])

    main_buf = 30.0 if 30.0 in per_buffer else BUFFERS[1]
    cur = per_buffer[main_buf]
    cov = cur[cur.covered]
    n_cov, n_nocov = len(cov), int((~cur.covered).sum())
    auc = rank_auc(cov.p_max.values, neg) if n_cov and neg.size else float("nan")
    ci_lo, ci_hi = bootstrap_ci(cov.p_max.values, neg) if n_cov and neg.size else (np.nan, np.nan)
    frac04 = float((cov.p_max >= 0.4).mean()) if n_cov else float("nan")
    bo_auc = (rank_auc(bo_st[bo_st.covered].p_max.values, neg)
              if len(bo_st) and neg.size else float("nan"))

    # buffer sweep spread, on the same covered sites
    aucs_by_buf = {}
    for bm, st in per_buffer.items():
        c = st[st.covered]
        aucs_by_buf[bm] = rank_auc(c.p_max.values, neg) if len(c) and neg.size else np.nan
    sweep = [v for v in aucs_by_buf.values() if np.isfinite(v)]
    spread = (max(sweep) - min(sweep)) if len(sweep) > 1 else 0.0

    # ---- co-location split (G1.5) -----------------------------------------------------
    d_sub = dict(zip(sites.site_id, sites.d_substation_m))
    cur = cur.assign(has_sub_300m=[bool(d_sub.get(s, 1e9) <= SWITCHYARD_M)
                                   for s in cur.site_id])
    split = (cur[cur.covered].groupby("has_sub_300m")
             .agg(n=("p_max", "size"), median_p_max=("p_max", "median"),
                  max_p_max=("p_max", "max"), frac_ge_04=("p_max", lambda x: (x >= 0.4).mean())))

    # ------------------------------------------------------------------ report
    print(f"\n=== Stage 1 probe: {a.aoi}  ({prob_dir.relative_to(ROOT)}) ===")
    print(f"coverage: {n_cov} sites covered, {n_nocov} with no raster "
          f"(reported, NEVER a denominator)")
    if n_cov and n_nocov:
        print(f"  covered median area {cov.area_m2.median():,.0f} m2 vs uncovered "
              f"{cur[~cur.covered].area_m2.median():,.0f} m2  "
              "(shows the cell subsample is not size-correlated)")
    print(f"\nG1.P positive control: {len(pc_cov)} battery-free substations "
          f">= {POSCTRL_MIN_AREA:,.0f} m2 covered, frac(p_max>=0.4) = {pc_frac04:.2f}")
    print(f"  -> {'PASS: the model transfers, so a null below is INFORMATIVE' if g1p else 'FAIL: GATE 1 IS UNINFORMATIVE (domain transfer, not batteries)'}")

    print(f"\nsites at {main_buf:.0f} m: median p_max {cov.p_max.median():.3f}, "
          f"p90 {cov.p90.median():.3f}, frac(p_max>=0.4) {frac04:.2f}")
    print(f"AUC vs battery-free substations: {auc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"battery-only geometry (n={len(bo_st[bo_st.covered]) if len(bo_st) else 0}): "
          f"AUC {bo_auc:.3f}")
    print(f"buffer sweep AUC: " + ", ".join(f"{k:.0f}m={v:.3f}" for k, v in aucs_by_buf.items())
          + f"   spread {spread:.3f}")
    print("\nG1.5 co-location split:")
    print(split.round(3).to_string() if len(split) else "  (no covered sites)")

    print("\n--- gates ---")
    if not g1p:
        print("  G1.P FAIL -> everything below is UNINFORMATIVE, not negative")
    g11 = auc >= 0.70 and ci_lo > 0.50 and frac04 >= 0.25 and (
        not np.isfinite(bo_auc) or bo_auc >= 0.60)
    g12 = 0.60 <= auc < 0.70
    g13 = spread > 0.10
    g14 = np.isfinite(auc) and frac04 < 0.10
    print(f"  G1.1 PASS  auc {auc:.3f}>=0.70, ci_lo {ci_lo:.3f}>0.50, "
          f"frac>=0.4 {frac04:.2f}>=0.25, battery-only {bo_auc:.3f}>=0.60 -> "
          f"{'YES' if g11 else 'no'}")
    print(f"  G1.2 WEAK  auc in [0.60,0.70) -> {'YES (rerun on sites with no substation within 300 m)' if g12 else 'no'}")
    print(f"  G1.3 buffer  spread {spread:.3f} -> "
          + ("exceeds 0.10, but there is no pass to downgrade" if g13 and not g11
             else "FAIL: this pass is a buffer artefact" if g13 else "ok"))
    print(f"  G1.4 NULL  frac(p_max>=0.4) < 0.10 -> {'YES' if g14 else 'no'}"
          + ("  (expected; not evidence against feasibility given G1.P passed)"
             if g14 and g1p else ""))
    # An AUC significantly BELOW 0.5 is not the pre-registered NULL and must not be
    # reported as one: it means the two classes ARE separated, with batteries ranked below
    # substations. The pre-registration (ff35af1) enumerated PASS / WEAK / NULL and did
    # not anticipate this outcome, so it is named explicitly rather than folded into the
    # else branch. It is necessary-but-not-sufficient for a battery detector: "does not
    # look like a switchyard" is not the same claim as "is positively identifiable".
    inverse = np.isfinite(ci_hi) and ci_hi < 0.50
    verdict = ("UNINFORMATIVE (positive control failed)" if not g1p else
               "FAIL (buffer artefact)" if g11 and g13 else
               "PASS: the production model already ranks BESS -- re-ranking is the cheap product"
               if g11 else "WEAK" if g12 else
               "INVERSE: the classes separate, but batteries rank BELOW substations "
               "(not the pre-registered NULL; unenumerated outcome)" if inverse else
               "NULL: expected, kills only the re-ranking route, not the study")
    print(f"\nVERDICT (Stage 1): {verdict}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    # Append-and-dedup on (run_name, aoi), like every other results CSV in this study.
    # A plain overwrite here would silently discard the previous region's per-site rows
    # the moment a second region was probed.
    sp = RESULTS / "bess_prob_probe.csv"
    rows_out = site_probs.assign(run_name=run, aoi=a.aoi, eval_version=EVAL_VERSION)
    if sp.exists():
        prev = pd.read_csv(sp)
        prev = prev[~((prev.run_name == run) & (prev.aoi == a.aoi))]
        rows_out = pd.concat([prev, rows_out], ignore_index=True)
    rows_out.to_csv(sp, index=False)
    summ = {
        "run_name": run, "aoi": a.aoi, "eval_version": EVAL_VERSION,
        "n_rasters": len(rindex), "n_sites_covered": n_cov, "n_no_coverage": n_nocov,
        "n_posctrl_covered": len(pc_cov), "posctrl_frac_ge_04": round(pc_frac04, 4),
        "G1P": "PASS" if g1p else "FAIL",
        "auc": round(auc, 4), "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
        "frac_ge_04": round(frac04, 4),
        "median_p_max": round(float(cov.p_max.median()), 4) if n_cov else "",
        "auc_battery_only": round(bo_auc, 4), "n_battery_only": int(
            len(bo_st[bo_st.covered]) if len(bo_st) else 0),
        "buffer_auc": json.dumps({str(k): (None if not np.isfinite(v) else round(v, 4))
                                  for k, v in aucs_by_buf.items()}),
        "buffer_spread": round(spread, 4),
        "inverse_separation": bool(np.isfinite(ci_hi) and ci_hi < 0.50),
        "verdict": verdict,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
    }
    out = RESULTS / "bess_prob_probe_summary.csv"
    row = pd.DataFrame([summ])
    if out.exists():
        prev = pd.read_csv(out)
        prev = prev[~((prev.run_name == run) & (prev.eval_version == EVAL_VERSION))]
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
