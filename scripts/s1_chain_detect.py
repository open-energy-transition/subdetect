"""S1 tower-chain detector prototype + validation on sindh_test.

Feasibility context (measured 2026-07-19, this repo): mapped tower locations are
only weakly separable as individual S1 bright points (3x3-max VH dB AUC 0.702 vs
background) -- so this detector leans on the *chain geometry*: many weak points,
collinear over kilometres with regular spacing, are collectively unambiguous where
any single point is not. See docs/issues/s1-tower-chain-corridor-evidence.md.

Per cell (data/osmose_regions/sindh_test/composites/<cell>/composite_s1.tif):
  1. dB-convert VV+VH, local contrast = dB - 31px local mean; point score =
     max(VV, VH) contrast at 3x3 local maxima above --contrast-db.
  2. RANSAC chains: random point pairs seed lines; inliers within --corridor-m of
     the line; keep chains with >= --min-points inliers spanning >= --min-span-m
     and median inlier spacing in [120, 700] m. Greedy: accept best, drop its
     inliers, repeat.
  3. Validate against mapped OSM lines: point precision (chain points within 100 m
     of a mapped line) and line coverage (mapped-line samples within 150 m of a
     chain).
  4. Re-rank the sindh_test v9_mean candidates by conf * chain-distance prior and
     score AUC/P@20/P@50 -- including a variant where the detector's output is the
     ONLY corridor evidence (simulating unmapped regions).

Usage:
  pixi run -e ml python scripts/s1_chain_detect.py [--contrast-db 3.5] [--corridor-m 60]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.warp
from scipy import ndimage
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from subdetect.config import Settings, geodesic_area_m2  # noqa: E402
from subdetect.evaluate import auc, precision_at  # noqa: E402
from subdetect.postprocess import polygonize_chips_v2  # noqa: E402
from eval_polygonize_v2 import _ground_truth  # noqa: E402

EQ = "EPSG:6933"
PX_M = 10.0


def bright_points(s1_tif: Path, contrast_db: float) -> np.ndarray:
    """(N,2) EPSG:6933 coords of local-maxima bright points."""
    with rasterio.open(s1_tif) as src:
        arr = src.read().astype("float32")
        valid = arr[0] > 0
        db = arr / 500.0 - 50.0
        contrast = np.full(db.shape[1:], -99.0, "float32")
        for b in range(2):
            band = np.where(valid, db[b], np.nan)
            local = ndimage.uniform_filter(np.nan_to_num(band), 31)
            norm = ndimage.uniform_filter(valid.astype("float32"), 31)
            local = np.where(norm > 0.3, local / np.maximum(norm, 1e-6), np.nan)
            c = band - local
            contrast = np.fmax(contrast, np.nan_to_num(c, nan=-99.0))
        is_max = contrast == ndimage.maximum_filter(contrast, 3)
        rows, cols = np.where(is_max & (contrast >= contrast_db) & valid)
        if not len(rows):
            return np.empty((0, 2))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        ex, ey = rasterio.warp.transform(src.crs, EQ, xs, ys)
    return np.column_stack([ex, ey])


def fit_chains(pts: np.ndarray, corridor_m: float, min_points: int, min_span_m: float,
               rng: np.random.Generator, iters: int = 4000, max_chains: int = 12) -> list[dict]:
    chains = []
    active = np.ones(len(pts), bool)
    for _ in range(max_chains):
        idx = np.where(active)[0]
        if len(idx) < min_points:
            break
        best = None
        p = pts[idx]
        for _ in range(iters):
            i, j = rng.choice(len(p), 2, replace=False)
            d = p[j] - p[i]
            span = np.hypot(*d)
            if not 500.0 <= span <= 4000.0:
                continue
            u = d / span
            rel = p - p[i]
            t = rel @ u
            perp = np.abs(rel @ np.array([-u[1], u[0]]))
            inl = perp <= corridor_m
            if inl.sum() < min_points:
                continue
            ti = np.sort(t[inl])
            span_i = ti[-1] - ti[0]
            if span_i < min_span_m:
                continue
            gaps = np.diff(ti)
            gaps = gaps[gaps > 1.0]
            if len(gaps) == 0 or not 120.0 <= np.median(gaps) <= 700.0 or gaps.max() > 1200.0:
                continue
            score = inl.sum() * min(span_i / min_span_m, 3.0)
            if best is None or score > best[0]:
                best = (score, inl, p[i], u, ti)
        if best is None:
            break
        _, inl, origin, u, ti = best
        a, b = origin + u * ti[0], origin + u * ti[-1]
        chains.append({"geometry": LineString([a, b]), "n_points": int(inl.sum()),
                       "span_m": float(ti[-1] - ti[0]),
                       "points": pts[idx[inl]]})
        active[idx[inl]] = False
    return chains


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast-db", type=float, default=5.0)
    ap.add_argument("--corridor-m", type=float, default=60.0)
    ap.add_argument("--min-points", type=int, default=6)
    ap.add_argument("--min-span-m", type=float, default=1500.0)
    ap.add_argument("--max-neighbors", type=int, default=3,
                    help="Drop bright points with more than this many neighbors within "
                         "250 m: towers are isolated (chain spacing 250-400 m), village/"
                         "town clutter clusters densely. <0 disables.")
    ap.add_argument("--prob-dir", default="data/osmose_regions/sindh_test_v9mean/prob")
    a = ap.parse_args()
    rng = np.random.default_rng(7)

    from scipy.spatial import cKDTree

    cells = sorted(Path(ROOT / "data/osmose_regions/sindh_test/composites").glob("*/composite_s1.tif"))
    all_chains, n_pts_total, n_pts_kept = [], 0, 0
    for tif in cells:
        pts = bright_points(tif, a.contrast_db)
        n_pts_total += len(pts)
        if a.max_neighbors >= 0 and len(pts) > 1:
            tree = cKDTree(pts)
            n_nb = np.array([len(nb) - 1 for nb in tree.query_ball_point(pts, 250.0)])
            pts = pts[n_nb <= a.max_neighbors]
        n_pts_kept += len(pts)
        all_chains += fit_chains(pts, a.corridor_m, a.min_points, a.min_span_m, rng)
    print(f"{len(cells)} cells: {n_pts_total} bright points, {n_pts_kept} after "
          f"isolation filter -> {len(all_chains)} chains "
          f"({sum(c['n_points'] for c in all_chains)} chained points)")
    if not all_chains:
        print("NO CHAINS FOUND — detector does not fire at these thresholds.")
        return

    chains_gdf = gpd.GeoDataFrame(
        [{"n_points": c["n_points"], "span_m": c["span_m"], "geometry": c["geometry"]}
         for c in all_chains], crs=EQ)
    out = ROOT / "data/eval_results/s1_chains_sindh_test.geoparquet"
    chains_gdf.to_crs("EPSG:4326").to_parquet(out)
    print(f"wrote chains -> {out}")

    # --- validation vs mapped lines
    lines = gpd.read_parquet(ROOT / "data/labels/pakistan/lines.parquet").to_crs(EQ)
    bounds4326 = chains_gdf.to_crs("EPSG:4326").total_bounds
    lines4326 = gpd.read_parquet(ROOT / "data/labels/pakistan/lines.parquet")
    lines_reg = lines4326.cx[bounds4326[0]:bounds4326[2], bounds4326[1]:bounds4326[3]].to_crs(EQ)
    line_tree = STRtree(list(lines_reg.geometry.values))

    pt_hits = pt_total = 0
    for c in all_chains:
        for x, y in c["points"]:
            pt_total += 1
            i = line_tree.nearest(Point(x, y))
            if lines_reg.geometry.values[i].distance(Point(x, y)) <= 100.0:
                pt_hits += 1
    print(f"chain-point precision vs mapped lines (<=100 m): {pt_hits}/{pt_total} "
          f"= {pt_hits / max(pt_total, 1):.2f}")

    chain_tree = STRtree(list(chains_gdf.geometry.values))
    covered = total = 0
    for geom in lines_reg.geometry:
        for d in np.arange(0.0, geom.length, 200.0):
            total += 1
            p = geom.interpolate(d)
            i = chain_tree.nearest(p)
            if chains_gdf.geometry.values[i].distance(p) <= 150.0:
                covered += 1
    print(f"mapped-line coverage (<=150 m): {covered}/{total} = {covered / max(total, 1):.2f}")

    # --- re-rank test on the v9_mean candidates
    settings = Settings.load()
    v2 = polygonize_chips_v2(ROOT / a.prob_dir, lo=0.2, hi=0.4)
    v2["area_m2"] = [geodesic_area_m2(g) for g in v2.geometry]
    cands = v2[v2.area_m2 >= settings.min_sub_area_m2].reset_index(drop=True)
    cb = gpd.GeoSeries(cands.geometry, crs="EPSG:4326").total_bounds
    gt_union, gt_desc = _ground_truth("pakistan", cb)
    y = cands.to_crs(EQ).intersects(gt_union).values
    cents = cands.to_crs(EQ).geometry.centroid

    def dist_prior(tree, geoms, decay=2000.0):
        pr = []
        for c in cents:
            i = tree.nearest(c)
            pr.append(0.5 + 0.5 * np.exp(-geoms[i].distance(c) / decay))
        return np.array(pr)

    conf = cands.conf_max.values
    chain_pr = dist_prior(chain_tree, chains_gdf.geometry.values)
    line_pr = dist_prior(line_tree, lines_reg.geometry.values)
    print(f"\n{len(cands)} candidates, {int(y.sum())} hits ({gt_desc})")
    print(f"{'variant':>12}  {'AUC':>6}  {'P@20':>5}  {'P@50':>5}")
    for name, s in [("conf", conf), ("conf*chain", conf * chain_pr),
                    ("conf*line", conf * line_pr)]:
        print(f"{name:>12}  {auc(s, y):6.3f}  {precision_at(s, y, 20):5.2f}  "
              f"{precision_at(s, y, 50):5.2f}")


if __name__ == "__main__":
    main()
