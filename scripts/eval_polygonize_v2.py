"""Compare polygonize scoring variants on the Yunnan pilot, zero new data.

Ground truth is free: OSM substations (Overpass) over the pilot footprint. A component
intersecting a mapped substation (node centers buffered to a disc) is a verified true
positive; the rest are "unknown, mostly FP". For each scoring variant we rank all
components and measure AUC + precision@k for hitting mapped substations, plus how much
of the 0.5 fusion plateau (S1-confident / S2-silent, where conf_max is unrankable)
each variant can actually spread out.

Known bias: mapped substations skew large/obvious, so any size-correlated metric gets
flattered -- n_pixels is therefore reported as a diagnostic row, not proposed as a score.

Usage:
  pixi run -e ml python scripts/eval_polygonize_v2.py                      # yunnan/Overpass
  pixi run -e ml python scripts/eval_polygonize_v2.py \
      --prob-dir data/osmose_regions/sindh_test/prob --labels pakistan     # local polygons
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
from subdetect.evaluate import auc, precision_at  # noqa: E402
from subdetect.postprocess import polygonize_chips, polygonize_chips_v2  # noqa: E402
from osmose_detect import fetch_substations  # noqa: E402

EQ = "EPSG:6933"
NODE_DISC_M = 100.0  # Overpass returns centers; a disc gives polygons a chance to hit
PLATEAU = (0.49, 0.51)


def _ground_truth(labels_aoi: str | None, bounds) -> tuple[object, str]:
    """Union geometry of known substations over `bounds`; local polygon labels when an
    AOI is given (polygons as-is + nodes as discs), else Overpass node discs."""
    b = bounds
    if labels_aoi:
        polys = gpd.read_parquet(ROOT / "data" / "labels" / labels_aoi / "substations_poly.parquet")
        polys = polys.cx[b[0]:b[2], b[1]:b[3]]
        parts = [polys.to_crs(EQ).geometry]
        node_p = ROOT / "data" / "labels" / labels_aoi / "substations_node.parquet"
        n_nodes = 0
        if node_p.exists():
            nodes = gpd.read_parquet(node_p).cx[b[0]:b[2], b[1]:b[3]]
            n_nodes = len(nodes)
            if n_nodes:
                parts.append(nodes.to_crs(EQ).buffer(NODE_DISC_M))
        union = gpd.GeoSeries(pd.concat(parts, ignore_index=True), crs=EQ).union_all()
        return union, f"{len(polys)} label polygons + {n_nodes} node discs ({labels_aoi})"
    subs = fetch_substations((b[0] - 0.05, b[1] - 0.05, b[2] + 0.05, b[3] + 0.05))
    if subs.empty:
        return None, "no OSM substations in footprint"
    return (subs.to_crs(EQ).buffer(NODE_DISC_M).union_all(),
            f"{len(subs)} Overpass substations ({NODE_DISC_M:.0f} m discs)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob-dir", default="data/osmose_regions/yunnan/prob")
    ap.add_argument("--labels", default=None,
                    help="AOI name under data/labels/ for polygon ground truth "
                         "(e.g. pakistan); default: Overpass")
    ap.add_argument("--out", default=None, help="Optional GeoJSON path for v2 components")
    a = ap.parse_args()

    settings = Settings.load()
    prob_dir = ROOT / a.prob_dir

    v1 = polygonize_chips(prob_dir, 0.3)
    v2 = polygonize_chips_v2(prob_dir)  # lo=0.2, hi=0.4
    for df in (v1, v2):
        df["area_m2"] = [geodesic_area_m2(g) for g in df.geometry]
    v1f = v1[v1.area_m2 >= settings.min_sub_area_m2].reset_index(drop=True)
    v2f = v2[v2.area_m2 >= settings.min_sub_area_m2].reset_index(drop=True)
    print(f"v1 (single 0.3 threshold):    {len(v1)} components, {len(v1f)} after area floor")
    print(f"v2 (hysteresis 0.4 -> 0.2):   {len(v2)} components, {len(v2f)} after area floor")

    b = gpd.GeoSeries([*v1f.geometry, *v2f.geometry], crs="EPSG:4326").total_bounds
    subs_buf, gt_desc = _ground_truth(a.labels, b)
    if subs_buf is None:
        print(gt_desc + "; cannot score")
        return
    print(f"ground truth: {gt_desc}\n")

    v1f["hit"] = v1f.to_crs(EQ).intersects(subs_buf).values
    v2f["hit"] = v2f.to_crs(EQ).intersects(subs_buf).values

    header = f"{'variant / score':<28} {'n':>5} {'hits':>5} {'AUC':>6} {'P@20':>6} {'P@50':>6}"
    print(header)
    print("-" * len(header))
    rows = [
        ("v1 confidence (max, today)", v1f, v1f.confidence.values),
        ("v2 conf_max", v2f, v2f.conf_max.values),
        ("v2 conf_p90 (proposed)", v2f, v2f.conf_p90.values),
        ("v2 conf_mean", v2f, v2f.conf_mean.values),
        ("v2 n_pixels (diagnostic)", v2f, v2f.n_pixels.values.astype(float)),
    ]
    for name, df, scores in rows:
        y = df.hit.values
        print(f"{name:<28} {len(df):>5} {int(y.sum()):>5} "
              f"{auc(scores, y):>6.3f} {precision_at(scores, y, 20):>6.2f} "
              f"{precision_at(scores, y, 50):>6.2f}")

    lo_p, hi_p = PLATEAU
    p1 = v1f[(v1f.confidence >= lo_p) & (v1f.confidence <= hi_p)]
    p2 = v2f[(v2f.conf_max >= lo_p) & (v2f.conf_max <= hi_p)]
    print(f"\nplateau (conf_max in [{lo_p}, {hi_p}]):")
    print(f"  v1: {len(p1)}/{len(v1f)} components, "
          f"{p1.confidence.nunique()} distinct confidence values -> unrankable")
    if len(p2):
        print(f"  v2: {len(p2)}/{len(v2f)} components, conf_p90 spans "
              f"{p2.conf_p90.min():.3f}-{p2.conf_p90.max():.3f} "
              f"({p2.conf_p90.nunique()} distinct values), "
              f"AUC within plateau {auc(p2.conf_p90.values, p2.hit.values):.3f}")

    if a.out:
        out = ROOT / a.out
        v2f.sort_values("conf_max", ascending=False).reset_index(drop=True).to_file(
            out, driver="GeoJSON")
        print(f"\nwrote {out}: {len(v2f)} v2 components for side-by-side review")


if __name__ == "__main__":
    main()
