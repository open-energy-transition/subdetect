"""Persisted field evaluation: score a run's prob rasters against real substations and
append one row to data/eval_results/field_eval.csv.

Unlike val/mIoU (which is not comparable across chip-index rebuilds -- see the config
header warnings in configs/terramind_sub_v6*.yaml), this evaluates directly against the
raw, un-refined OSM ground truth (substations_poly.parquet + node discs, min_area=0) that
never changes between label-refinement experiments, so runs are comparable over time.

Metrics: component-level AUC/P@20/P@50 (does a ranked candidate hit a real substation),
installation-level recall bucketed by area (does ANY candidate intersect this real
substation), and a false-positive proxy (ranked candidates that hit nothing).

Usage:
  pixi run -e ml python scripts/field_eval.py --run-name v8_run2 \
      --prob-dir data/osmose_regions/sindh_test_v8run2/prob --labels pakistan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from subdetect.config import Settings, geodesic_area_m2  # noqa: E402
from subdetect.evaluate import AREA_BUCKETS, auc, precision_at  # noqa: E402
from subdetect.postprocess import polygonize_chips_v2, raster_coverage  # noqa: E402
from eval_polygonize_v2 import _ground_truth  # noqa: E402

EQ = "EPSG:6933"
RESULTS_CSV = ROOT / "data" / "eval_results" / "field_eval.csv"
# Bumped 2026-08-15 with two fixes that both change every number in the CSV: the recall
# denominator is now the inferred footprint rather than the candidate bounding box, and the
# candidate area filter is decoupled from the label floor (see field_eval's docstring).
# Rows carry this so v1 and v2 results coexist instead of being silently mixed -- only
# compare rows of equal eval_version.
EVAL_VERSION = 2
# Smallest component kept as a candidate. 1,000 m2 = the lower edge of the smallest
# evaluated bucket, so the filter no longer excludes a size class by construction.
MIN_CAND_AREA_M2 = 1000.0


def _label_polys(labels_aoi: str, bounds, covered=None) -> gpd.GeoDataFrame:
    """Raw (un-refined) substation polygons + node discs, every size, for recall buckets.

    `covered` (the union of the run's probability-raster footprints) restricts the recall
    DENOMINATOR to substations that were actually inferred. Without it the denominator is a
    bounding box, and the inferred area is a ragged set of ROI-selected cells rather than a
    rectangle -- so substations that no raster ever covered are scored as misses. Measured
    on sindh_test: 51 of 163 (31%), including the region's two largest substations, one of
    them 51 km outside the nearest inferred cell. Membership is by centroid, so a substation
    is counted exactly once and only where it could have been seen.
    """
    b = bounds
    polys = gpd.read_parquet(ROOT / "data" / "labels" / labels_aoi / "substations_poly.parquet")
    polys = polys.cx[b[0]:b[2], b[1]:b[3]].copy()
    if "area_m2" not in polys.columns:
        polys["area_m2"] = [geodesic_area_m2(g) for g in polys.geometry]
    polys = polys[["geometry", "area_m2"]]
    node_p = ROOT / "data" / "labels" / labels_aoi / "substations_node.parquet"
    if node_p.exists():
        nodes = gpd.read_parquet(node_p).cx[b[0]:b[2], b[1]:b[3]]
        if len(nodes):
            nodes = nodes[["geometry"]].copy()
            nodes["geometry"] = nodes.to_crs(EQ).buffer(100.0).to_crs("EPSG:4326")
            nodes["area_m2"] = 0.0  # nodes carry no area; bucketed separately below
            polys = pd.concat([polys, nodes], ignore_index=True)
    out = gpd.GeoDataFrame(polys, crs="EPSG:4326")
    if covered is not None:
        # Centroid in an equal-area CRS, not in lat/lon: a 200 m substation's 4326 centroid
        # is close enough, but the containment test is cheap to do correctly.
        cov_eq = gpd.GeoSeries([covered], crs="EPSG:4326").to_crs(EQ).iloc[0]
        out = out[out.to_crs(EQ).geometry.centroid.within(cov_eq).values].reset_index(drop=True)
    return out


def field_eval(run_name: str, prob_dir: Path, labels_aoi: str, lo=0.2, hi=0.4,
               min_cand_area_m2: float = MIN_CAND_AREA_M2) -> dict:
    """`min_cand_area_m2` is the smallest component kept as a candidate. It used to be
    `settings.min_sub_area_m2`, which is the LABEL floor (20,000 m2 in configs/aoi.yaml) --
    two different jobs sharing one constant. At 20,000 m2 a candidate had to be as large as
    the biggest ground-truth bucket to count at all, so a substation smaller than that could
    only ever be detected via an over-predicted footprint. Measured on sindh_test v12mean,
    that filter discarded 20 correctly-detected substations whose components carried a
    median peak probability of 0.53. The two constants are now independent.
    """
    settings = Settings.load()  # noqa: F841 -- kept for callers that patch label settings
    v2 = polygonize_chips_v2(prob_dir, lo=lo, hi=hi)
    v2["area_m2"] = [geodesic_area_m2(g) for g in v2.geometry]
    v2f = v2[v2.area_m2 >= min_cand_area_m2].reset_index(drop=True)

    covered = raster_coverage(prob_dir)
    region = prob_dir.parent.name if prob_dir.name == "prob" else prob_dir.name
    row = {"run_name": run_name, "region": region, "eval_version": EVAL_VERSION,
           "min_cand_area_m2": min_cand_area_m2, "n_candidates": len(v2f)}
    if v2f.empty:
        row.update(auc=float("nan"), p_at_20=float("nan"), p_at_50=float("nan"),
                    n_hits=0, n_fp_proxy=0)
        return row

    # Bound the ground truth by the inferred footprint, not by where candidates happened to
    # land -- the latter varies run to run, which silently made recall denominators differ
    # between rows of this very CSV (n_20000-inf has been both 19 and 21 for the same region).
    bounds = (gpd.GeoSeries([covered], crs="EPSG:4326").total_bounds if covered is not None
              else gpd.GeoSeries(v2f.geometry, crs="EPSG:4326").total_bounds)
    gt_union, gt_desc = _ground_truth(labels_aoi, bounds)
    v2f["hit"] = (v2f.to_crs(EQ).intersects(gt_union).values
                  if gt_union is not None else False)

    y, scores = v2f.hit.values, v2f.conf_max.values
    row.update(
        auc=round(auc(scores, y), 4),
        p_at_20=round(precision_at(scores, y, 20), 4),
        p_at_50=round(precision_at(scores, y, 50), 4),
        n_hits=int(y.sum()),
        n_fp_proxy=int((~y).sum()),
        gt_desc=gt_desc,
    )

    # installation-level recall bucketed by area: does ANY candidate hit this real sub?
    labels = _label_polys(labels_aoi, bounds, covered=covered)
    row["n_gt_uncovered"] = len(_label_polys(labels_aoi, bounds)) - len(labels)
    v2_eq = v2f.to_crs(EQ)
    for lo_a, hi_a in AREA_BUCKETS:
        sel = labels[(labels.area_m2 >= lo_a) & (labels.area_m2 < hi_a)]
        if sel.empty:
            row[f"recall_{lo_a}-{hi_a if hi_a != np.inf else 'inf'}"] = float("nan")
            continue
        hit = sel.to_crs(EQ).geometry.apply(lambda g: v2_eq.intersects(g).any())
        row[f"recall_{lo_a}-{hi_a if hi_a != np.inf else 'inf'}"] = round(float(hit.mean()), 3)
        row[f"n_{lo_a}-{hi_a if hi_a != np.inf else 'inf'}"] = len(sel)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--prob-dir", required=True)
    ap.add_argument("--labels", default=None, help="AOI under data/labels/ (default: Overpass)")
    ap.add_argument("--min-cand-area-m2", type=float, default=MIN_CAND_AREA_M2,
                    help="Smallest component kept as a candidate (decoupled from the label "
                         "floor as of eval_version 2)")
    a = ap.parse_args()

    row = field_eval(a.run_name, ROOT / a.prob_dir, a.labels,
                     min_cand_area_m2=a.min_cand_area_m2)
    print(pd.Series(row).to_string())

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    prior = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else pd.DataFrame()
    # Key on eval_version too, so the v1 history stays intact alongside the v2 re-baseline
    # instead of being overwritten by it. Rows predating the bump have no column -> v1.
    if len(prior):
        if "eval_version" not in prior.columns:
            prior["eval_version"] = 1
        prior = prior[~((prior.get("run_name") == a.run_name)
                        & (prior.get("region") == row["region"])
                        & (prior.get("eval_version") == row["eval_version"]))]
    updated = pd.concat([prior, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(RESULTS_CSV, index=False)
    print(f"\nwrote {RESULTS_CSV} ({len(updated)} rows)")


if __name__ == "__main__":
    main()
