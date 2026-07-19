"""Do mapped power=tower features improve candidate ranking? Tested on sindh_test.

Physics: transmission towers form regularly-spaced chains along corridors, and
substations are the nodes where several chains converge/terminate. Villages can mimic
a substation's local texture but not converging tower chains. This script tests the
cheap (OSM-mapped-tower) version of that signal before investing in S1
persistent-scatterer chain detection (see the GitHub issue "Corridor evidence from
Sentinel-1 tower chains").

Per candidate (v9_mean sindh_test rasters, same polygonization as field_eval.py):
  tower_dist_m    distance to nearest power=tower node
  n_towers_1500   towers within 1.5 km
  n_bearings      corridor convergence: tower-tower links (< 500 m) among towers
                  within 1.5 km, link bearings folded mod 180 deg into 30-deg bins;
                  count of bins holding >= 2 links
Ranking variants scored with component-level AUC / P@20 / P@50 against the same
fixed OSM ground truth as field_eval.py:
  conf            conf_max alone (the field_eval baseline)
  conf*tower      conf_max * (0.5 + 0.5 * exp(-tower_dist / 2000))   [line-prior shape]
  conf*conv       conf_max * (0.75 + 0.25 * min(n_bearings, 3))      [convergence]
  conf*line       conf_max * mapped-line prior (production _rank formula, reference)

Usage:
  pixi run -e ml python scripts/tower_features_eval.py \
      [--prob-dir data/osmose_regions/sindh_test_v9mean/prob]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from subdetect.config import Settings, geodesic_area_m2  # noqa: E402
from subdetect.evaluate import auc, precision_at  # noqa: E402
from subdetect.postprocess import polygonize_chips_v2  # noqa: E402
from eval_polygonize_v2 import _ground_truth  # noqa: E402

EQ = "EPSG:6933"


def load_towers(seq_path: Path, bounds) -> gpd.GeoDataFrame:
    pts = []
    with seq_path.open() as f:
        for line in f:
            line = line.strip().lstrip("\x1e")
            if not line:
                continue
            try:
                feat = json.loads(line)
            except json.JSONDecodeError:
                continue
            g = feat.get("geometry") or {}
            if g.get("type") != "Point":
                continue
            lon, lat = g["coordinates"][:2]
            if bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]:
                pts.append(Point(lon, lat))
    return gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326")


def tower_features(cands_eq: gpd.GeoDataFrame, towers_eq: gpd.GeoDataFrame) -> pd.DataFrame:
    tower_pts = list(towers_eq.geometry.values)
    tree = STRtree(tower_pts)
    xy = np.array([[p.x, p.y] for p in tower_pts])
    feats = []
    for geom in cands_eq.geometry:
        c = geom.centroid
        near_i = tree.query(c.buffer(1500.0), predicate="intersects")
        if len(xy):
            d_all = np.hypot(xy[:, 0] - c.x, xy[:, 1] - c.y)
            tower_dist = float(d_all.min())
        else:
            tower_dist = 1e6
        n_bearings = 0
        if len(near_i) >= 2:
            pts = xy[near_i]
            bins = set()
            for i in range(len(pts)):
                d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
                for j in np.where((d > 1.0) & (d < 500.0))[0]:
                    if j <= i:
                        continue
                    b = np.degrees(np.arctan2(pts[j, 1] - pts[i, 1],
                                              pts[j, 0] - pts[i, 0])) % 180.0
                    bins.add(int(b // 30))
            n_bearings = len(bins)
        feats.append({"tower_dist_m": tower_dist, "n_towers_1500": len(near_i),
                      "n_bearings": n_bearings})
    return pd.DataFrame(feats)


def line_prior(cands: gpd.GeoDataFrame, labels_aoi: str) -> np.ndarray:
    lines = gpd.read_parquet(ROOT / "data/labels" / labels_aoi / "lines.parquet").to_crs(EQ)
    tree = STRtree(list(lines.geometry.values))
    dists = []
    for geom in cands.to_crs(EQ).geometry:
        c = geom.centroid
        i = tree.nearest(c)
        dists.append(float(lines.geometry.values[i].distance(c)))
    d = np.array(dists)
    return 0.5 + 0.5 * np.exp(-d / 2000.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob-dir", default="data/osmose_regions/sindh_test_v9mean/prob")
    ap.add_argument("--towers", default="data/osm/pakistan_towers.geojsonseq")
    ap.add_argument("--labels", default="pakistan")
    a = ap.parse_args()

    settings = Settings.load()
    v2 = polygonize_chips_v2(ROOT / a.prob_dir, lo=0.2, hi=0.4)
    v2["area_m2"] = [geodesic_area_m2(g) for g in v2.geometry]
    cands = v2[v2.area_m2 >= settings.min_sub_area_m2].reset_index(drop=True)
    bounds = gpd.GeoSeries(cands.geometry, crs="EPSG:4326").total_bounds
    gt_union, gt_desc = _ground_truth(a.labels, bounds)
    cands["hit"] = cands.to_crs(EQ).intersects(gt_union).values
    print(f"{len(cands)} candidates, {int(cands.hit.sum())} hits ({gt_desc})")

    margin = 0.05
    towers = load_towers(ROOT / a.towers,
                         (bounds[0] - margin, bounds[1] - margin,
                          bounds[2] + margin, bounds[3] + margin))
    print(f"{len(towers)} mapped towers in region")
    feats = tower_features(cands.to_crs(EQ), towers.to_crs(EQ))
    cands = pd.concat([cands, feats], axis=1)

    y = cands.hit.values
    variants = {
        "conf": cands.conf_max.values,
        "conf*tower": cands.conf_max.values
            * (0.5 + 0.5 * np.exp(-cands.tower_dist_m.values / 2000.0)),
        "conf*conv": cands.conf_max.values
            * (0.75 + 0.25 * np.minimum(cands.n_bearings.values, 3)),
        "conf*line": cands.conf_max.values * line_prior(cands, a.labels),
        "conf*tower*conv": cands.conf_max.values
            * (0.5 + 0.5 * np.exp(-cands.tower_dist_m.values / 2000.0))
            * (0.75 + 0.25 * np.minimum(cands.n_bearings.values, 3)),
    }
    print(f"\n{'variant':>16}  {'AUC':>6}  {'P@20':>5}  {'P@50':>5}")
    for name, s in variants.items():
        print(f"{name:>16}  {auc(s, y):6.3f}  {precision_at(s, y, 20):5.2f}  "
              f"{precision_at(s, y, 50):5.2f}")

    # feature-alone separation (does the physics show up at all?)
    print("\nfeature-alone AUC (hit vs non-hit):")
    for f in ["tower_dist_m", "n_towers_1500", "n_bearings"]:
        s = -cands[f].values if f == "tower_dist_m" else cands[f].values
        print(f"  {f:>14}: {auc(s.astype(float), y):.3f}")

    out = ROOT / "data/eval_results/tower_features_sindh_test.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    cands.drop(columns="geometry").assign(
        centroid_lon=cands.geometry.centroid.x, centroid_lat=cands.geometry.centroid.y
    ).to_parquet(out)
    print(f"\nwrote per-candidate features -> {out}")


if __name__ == "__main__":
    main()
