"""Lead ranking by isolated S1 bright points ("tower prior from imagery, not OSM").

Follows the EHV detectability findings (scripts/ehv_detectability.py, issue doc
docs/issues/s1-tower-chain-corridor-evidence.md): transmission towers are
individually detectable in Sentinel-1 VH as bright local-contrast points (AUC 0.909
for >=400 kV), while Sentinel-2 cannot see them at 10 m (AUC ~ chance) -- so this
prior is S1-only by design. The anti-village trick from the chain detector carries
over: tower points are ISOLATED (<= --max-neighbors bright points within 250 m),
village clutter clusters densely.

Per candidate: count isolated bright points in a 150-1000 m annulus around the
centroid (the incoming spans + anchor towers around a substation yard, excluding
the yard's own structures) and the distance to the nearest such point.

Modes:
  --eval   score ranking variants on sindh_test v9_mean candidates against the
           fixed OSM ground truth (like field_eval.py)
  --apply  add features + rank_score_s1tower to a leads/candidates parquet
           (composites dir must follow <dir>/<cell>/composite_s1.tif layout)

Usage:
  pixi run -e ml python scripts/isolated_tower_prior.py --eval
  pixi run -e ml python scripts/isolated_tower_prior.py --apply \
      --candidates data/predictions_v9_mean/new_leads_pakistan_india_meanfusion.geoparquet \
      --composites data/composites/pakistan/composites data/composites/india_pilot/composites
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

EQ = "EPSG:6933"
ANNULUS_IN_M, ANNULUS_OUT_M = 150.0, 1000.0


def cell_bright_points(s1_tif: Path, contrast_db: float, max_neighbors: int) -> np.ndarray:
    """(N,2) EQ coords of isolated bright VH local maxima (tower signature)."""
    with rasterio.open(s1_tif) as src:
        arr = src.read(2).astype("float32")
        valid = arr > 0
        db = arr / 500.0 - 50.0
        band = np.where(valid, db, np.nan)
        local = ndimage.uniform_filter(np.nan_to_num(band), 31)
        norm = ndimage.uniform_filter(valid.astype("float32"), 31)
        local = np.where(norm > 0.3, local / np.maximum(norm, 1e-6), np.nan)
        contrast = np.nan_to_num(band - local, nan=-99.0)
        is_max = contrast == ndimage.maximum_filter(contrast, 3)
        rows, cols = np.where(is_max & (contrast >= contrast_db) & valid)
        if not len(rows):
            return np.empty((0, 2))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        ex, ey = rasterio.warp.transform(src.crs, EQ, xs, ys)
    pts = np.column_stack([ex, ey])
    if max_neighbors >= 0 and len(pts) > 1:
        tree = cKDTree(pts)
        n_nb = np.array([len(nb) - 1 for nb in tree.query_ball_point(pts, 250.0)])
        pts = pts[n_nb <= max_neighbors]
    return pts


def collect_points(comp_dirs: list[Path], cands: gpd.GeoDataFrame,
                   contrast_db: float, max_neighbors: int) -> np.ndarray:
    """Isolated bright points from every cell within 1.2 km of any candidate."""
    hull = cands.to_crs(EQ).buffer(1200.0).union_all()
    parts = []
    n_cells = 0
    for comp_dir in comp_dirs:
        for tif in sorted(comp_dir.glob("*/composite_s1.tif")):
            with rasterio.open(tif) as src:
                b = rasterio.warp.transform_bounds(src.crs, EQ, *src.bounds)
            if not box(*b).intersects(hull):
                continue
            n_cells += 1
            pts = cell_bright_points(tif, contrast_db, max_neighbors)
            if len(pts):
                parts.append(pts)
    print(f"scanned {n_cells} cells -> "
          f"{sum(len(p) for p in parts)} isolated bright points")
    return np.vstack(parts) if parts else np.empty((0, 2))


def features(cands: gpd.GeoDataFrame, pts: np.ndarray) -> pd.DataFrame:
    cents = cands.to_crs(EQ).geometry.centroid
    if not len(pts):
        return pd.DataFrame({"iso_towers_1km": 0, "iso_tower_dist_m": 1e6},
                            index=cands.index)
    tree = cKDTree(pts)
    n, dist = [], []
    for c in cents:
        idx = tree.query_ball_point([c.x, c.y], ANNULUS_OUT_M)
        d = np.hypot(pts[idx, 0] - c.x, pts[idx, 1] - c.y) if idx else np.array([])
        n.append(int(((d >= ANNULUS_IN_M) & (d <= ANNULUS_OUT_M)).sum()))
        dd, _ = tree.query([c.x, c.y])
        dist.append(float(dd))
    return pd.DataFrame({"iso_towers_1km": n, "iso_tower_dist_m": np.round(dist, 1)},
                        index=cands.index)


def iso_prior(f: pd.DataFrame) -> np.ndarray:
    return 0.75 + 0.25 * np.minimum(f.iso_towers_1km.values, 4) / 4.0


def run_eval(a) -> None:
    from subdetect.config import Settings, geodesic_area_m2
    from subdetect.evaluate import auc, precision_at
    from subdetect.postprocess import polygonize_chips_v2
    from eval_polygonize_v2 import _ground_truth
    from shapely.strtree import STRtree
    import json
    from shapely.geometry import Point

    settings = Settings.load()
    v2 = polygonize_chips_v2(ROOT / "data/osmose_regions/sindh_test_v9mean/prob", lo=0.2, hi=0.4)
    v2["area_m2"] = [geodesic_area_m2(g) for g in v2.geometry]
    cands = v2[v2.area_m2 >= settings.min_sub_area_m2].reset_index(drop=True)
    bounds = gpd.GeoSeries(cands.geometry, crs="EPSG:4326").total_bounds
    gt_union, gt_desc = _ground_truth("pakistan", bounds)
    y = cands.to_crs(EQ).intersects(gt_union).values

    pts = collect_points([ROOT / "data/osmose_regions/sindh_test/composites"],
                         cands, a.contrast_db, a.max_neighbors)
    f = features(cands, pts)

    # mapped-tower prior for comparison
    towers = []
    with (ROOT / "data/osm/pakistan_towers.geojsonseq").open() as fh:
        for line in fh:
            line = line.strip().lstrip("\x1e")
            if not line:
                continue
            try:
                g = json.loads(line)["geometry"]
            except Exception:
                continue
            if g and g["type"] == "Point":
                towers.append(Point(g["coordinates"][:2]))
    towers_eq = gpd.GeoDataFrame(geometry=towers, crs="EPSG:4326").to_crs(EQ)
    ttree = STRtree(list(towers_eq.geometry.values))
    tg = towers_eq.geometry.values
    tdist = np.array([float(tg[ttree.nearest(c)].distance(c))
                      for c in cands.to_crs(EQ).geometry.centroid])
    mapped_prior = 0.5 + 0.5 * np.exp(-tdist / 2000.0)

    conf = cands.conf_max.values
    ip = iso_prior(f)
    print(f"\n{len(cands)} candidates, {int(y.sum())} hits ({gt_desc})")
    print(f"{'variant':>22}  {'AUC':>6}  {'P@20':>5}  {'P@50':>5}")
    for name, s in [("conf", conf), ("conf*isoS1", conf * ip),
                    ("conf*mappedTower", conf * mapped_prior),
                    ("conf*isoS1*mapped", conf * ip * mapped_prior)]:
        print(f"{name:>22}  {auc(s, y):6.3f}  {precision_at(s, y, 20):5.2f}  "
              f"{precision_at(s, y, 50):5.2f}")
    print("\nfeature-alone AUC:")
    print(f"  iso_towers_1km: {auc(f.iso_towers_1km.values.astype(float), y):.3f}")
    print(f"  -iso_tower_dist: {auc(-f.iso_tower_dist_m.values, y):.3f}")


def run_apply(a) -> None:
    cands = gpd.read_parquet(ROOT / a.candidates)
    pts = collect_points([ROOT / d for d in a.composites], cands,
                         a.contrast_db, a.max_neighbors)
    f = features(cands, pts)
    cands["iso_towers_1km"] = f.iso_towers_1km.values
    cands["iso_tower_dist_m"] = f.iso_tower_dist_m.values
    cands["iso_tower_prior"] = np.round(iso_prior(f), 4)
    base = cands["rank_score_tower"] if "rank_score_tower" in cands else cands["rank_score"]
    cands["rank_score_s1tower"] = np.round(base * cands.iso_tower_prior, 4)
    cands = cands.sort_values("rank_score_s1tower", ascending=False).reset_index(drop=True)
    out = ROOT / a.candidates
    cands.to_parquet(out)
    if out.suffix != ".geojson":
        cands.to_file(out.with_suffix("").with_suffix(".geojson"), driver="GeoJSON")
    print(f"updated {out} ({len(cands)} rows; iso_towers_1km median "
          f"{int(cands.iso_towers_1km.median())})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--candidates")
    ap.add_argument("--composites", nargs="+", default=[])
    ap.add_argument("--contrast-db", type=float, default=5.0)
    ap.add_argument("--max-neighbors", type=int, default=3)
    a = ap.parse_args()
    if a.eval:
        run_eval(a)
    if a.apply:
        run_apply(a)


if __name__ == "__main__":
    main()
