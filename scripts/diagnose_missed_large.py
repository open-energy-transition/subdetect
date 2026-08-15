"""Where in the pipeline are large substations lost?

Written 2026-08-15, when `field_eval.py` reported ~0.33-0.53 installation recall in the
>=20,000 m2 bucket -- objects roughly 200 m across, 78% voltage-tagged, unmistakable in the
imagery. Missing half of those did not look like a model-capacity problem, so before
retraining anything it was worth asking *where* they were lost. Two of the answers below
turned out to be evaluation defects and were fixed in field_eval eval_version 2; the script
is kept because the same question recurs with every new run, and because it is the only
tool here that separates a model miss from a pipeline loss. There are five possible answers
and they point in completely different directions:

  HIT              a surviving candidate intersects the substation. Not lost.
  lost:area        the probability raster fires and a component forms, but the component is
                   dropped by field_eval's candidate area filter (or fragments into pieces
                   each below it). Fix is a filter/merge change. Free.
  lost:threshold   the raster fires inside the substation, but never reaches the
                   hysteresis seed (`hi`), so no component is ever created. Fix is
                   threshold/calibration -- also free, no retraining.
  lost:silent      the raster is flat inside the substation. A genuine model miss, and the
                   only outcome that argues for retraining.
  no-coverage      no probability raster covers this substation at all -- an ROI/inference
                   coverage gap, not a detection failure. Counting these as misses
                   understates recall.

For each ground-truth substation the script reports the raw probability statistics inside
its polygon (max / p90 / mean, straight off the uint8 raster, before any thresholding) plus
the largest pre-filter component that touches it, then assigns one of the labels above.

This is read-only: it consumes probability rasters and labels already on disk, needs no GPU
and trains nothing.

Usage:
  pixi run -e ml python scripts/diagnose_missed_large.py \
      --prob-dir data/osmose_regions/sindh_test_v12mean/prob --labels pakistan
  # all buckets, not just the >=20,000 m2 one:
  pixi run -e ml python scripts/diagnose_missed_large.py --prob-dir ... --labels pakistan \
      --min-area 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subdetect.config import geodesic_area_m2  # noqa: E402
from subdetect.postprocess import polygonize_chips_v2  # noqa: E402

EQ = "EPSG:6933"


def _raster_index(prob_dir: Path) -> gpd.GeoDataFrame:
    """Footprint of every probability raster, in EPSG:4326, for point-in-cell lookup."""
    from shapely.geometry import box

    rows = []
    for tif in sorted(prob_dir.glob("*.tif")):
        with rasterio.open(tif) as src:
            geom = gpd.GeoSeries([box(*src.bounds)], crs=src.crs).to_crs("EPSG:4326").iloc[0]
        rows.append({"path": str(tif), "geometry": geom})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _prob_stats(geom_4326, rindex: gpd.GeoDataFrame) -> dict:
    """Raw probability inside one substation polygon, pooled across every raster that
    overlaps it (large substations can straddle a cell edge). Returns NaNs when no raster
    covers it -- a coverage gap, which is not the same as a zero prediction."""
    vals = []
    hits = rindex.iloc[rindex.sindex.query(geom_4326, predicate="intersects")]
    for _, r in hits.iterrows():
        with rasterio.open(r["path"]) as src:
            g = gpd.GeoSeries([geom_4326], crs="EPSG:4326").to_crs(src.crs).iloc[0]
            try:
                arr, _ = rasterio.mask.mask(src, [g], crop=True, filled=False)
            except ValueError:  # polygon outside this raster's extent after reprojection
                continue
            m = arr[0]
            v = m.compressed() if np.ma.isMaskedArray(m) else m.ravel()
            if v.size:
                vals.append(v.astype("float32") / 255.0)
    if not vals:
        return {"covered": False, "p_max": np.nan, "p90": np.nan, "p_mean": np.nan,
                "n_px": 0}
    v = np.concatenate(vals)
    return {"covered": True, "p_max": float(v.max()), "p90": float(np.percentile(v, 90)),
            "p_mean": float(v.mean()), "n_px": int(v.size)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob-dir", required=True)
    ap.add_argument("--labels", required=True, help="AOI under data/labels/")
    ap.add_argument("--min-area", type=float, default=20000.0,
                    help="Only diagnose ground-truth substations at least this big")
    ap.add_argument("--lo", type=float, default=0.2, help="hysteresis grow (field_eval default)")
    ap.add_argument("--hi", type=float, default=0.4, help="hysteresis seed (field_eval default)")
    ap.add_argument("--min-cand-area", type=float, default=None,
                    help="Candidate area filter to diagnose against (default: field_eval's "
                         "current MIN_CAND_AREA_M2; pass 20000 to reproduce the "
                         "eval_version 1 behaviour that motivated this script)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    prob_dir = ROOT / a.prob_dir if not Path(a.prob_dir).is_absolute() else Path(a.prob_dir)
    sys.path.insert(0, str(ROOT / "scripts"))
    from field_eval import MIN_CAND_AREA_M2

    min_cand = a.min_cand_area if a.min_cand_area is not None else MIN_CAND_AREA_M2

    # Candidates exactly as field_eval builds them, but keep the PRE-filter set too: the
    # difference between the two is the "lost:area" population.
    cand = polygonize_chips_v2(prob_dir, lo=a.lo, hi=a.hi)
    cand["area_m2"] = [geodesic_area_m2(g) for g in cand.geometry]
    kept = cand[cand.area_m2 >= min_cand].reset_index(drop=True)
    print(f"candidates: {len(cand)} components, {len(kept)} survive the "
          f"{min_cand:.0f} m2 filter")

    rindex = _raster_index(prob_dir)
    bounds = rindex.total_bounds
    polys = gpd.read_parquet(ROOT / "data" / "labels" / a.labels / "substations_poly.parquet")
    polys = polys.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()
    if "area_m2" not in polys.columns:
        polys["area_m2"] = [geodesic_area_m2(g) for g in polys.geometry]
    polys = polys[polys.area_m2 >= a.min_area].reset_index(drop=True)
    print(f"ground truth: {len(polys)} substations >= {a.min_area:.0f} m2 in raster bounds")
    if polys.empty:
        return

    cand_eq, kept_eq = cand.to_crs(EQ), kept.to_crs(EQ)
    rows = []
    for _, inst in polys.iterrows():
        g_eq = gpd.GeoSeries([inst.geometry], crs="EPSG:4326").to_crs(EQ).iloc[0]
        touch_all = cand_eq[cand_eq.intersects(g_eq)]
        touch_kept = kept_eq[kept_eq.intersects(g_eq)]
        st = _prob_stats(inst.geometry, rindex)
        if not st["covered"]:
            verdict = "no-coverage"
        elif len(touch_kept):
            verdict = "HIT"
        elif len(touch_all):
            verdict = "lost:area"
        elif st["p_max"] >= a.hi:
            # raster reaches the seed level somewhere inside, but no component formed that
            # touches the polygon -- the seed sits just outside, or the component is
            # detached. Treat as a threshold/geometry loss, not a silent miss.
            verdict = "lost:threshold"
        elif st["p_max"] >= a.lo:
            verdict = "lost:threshold"
        else:
            verdict = "lost:silent"
        rows.append({
            "area_m2": round(float(inst.area_m2)),
            "voltage_v": inst.get("voltage_v", np.nan),
            "verdict": verdict,
            "p_max": round(st["p_max"], 3) if st["covered"] else np.nan,
            "p90": round(st["p90"], 3) if st["covered"] else np.nan,
            "p_mean": round(st["p_mean"], 3) if st["covered"] else np.nan,
            "n_components_touching": len(touch_all),
            "largest_component_m2": round(float(touch_all.to_crs("EPSG:4326").pipe(
                lambda d: cand.loc[d.index, "area_m2"]).max())) if len(touch_all) else 0,
            "lon": round(float(inst.geometry.centroid.x), 5),
            "lat": round(float(inst.geometry.centroid.y), 5),
        })

    df = pd.DataFrame(rows).sort_values(["verdict", "area_m2"], ascending=[True, False])
    counts = df.verdict.value_counts()
    print("\nverdict:")
    for k, v in counts.items():
        print(f"  {k:<16} {v:>4}  ({100 * v / len(df):.0f}%)")
    print(f"\nrecall as field_eval computes it: {counts.get('HIT', 0)}/{len(df)} = "
          f"{counts.get('HIT', 0) / len(df):.3f}")
    recoverable = counts.get("lost:area", 0) + counts.get("lost:threshold", 0)
    print(f"recoverable without retraining (area+threshold): {recoverable} "
          f"-> ceiling {(counts.get('HIT', 0) + recoverable) / len(df):.3f}")
    print(f"genuine model misses (silent): {counts.get('lost:silent', 0)}")
    print(f"\nprobability inside missed substations, by verdict:")
    print(df.groupby("verdict")[["p_max", "p90", "p_mean"]].median().to_string())

    out = a.out or ROOT / "data" / "eval_results" / f"missed_diagnosis_{prob_dir.parent.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
