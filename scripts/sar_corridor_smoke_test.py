"""Smoke test: can the SAR (S1) tower/line corridor signal detect substations ON ITS
OWN -- independent of the segmentation model's own probability output?

Everything built so far (tower_features_eval.py, isolated_tower_prior.py) used tower
evidence to RE-RANK candidates the U-Net had already found. This asks a different,
more basic question: if you strip the model out entirely and score raw locations
purely from S1 tower/line reflectivity and corridor-convergence geometry, does that
alone separate real substations from background?

Per location (real substation centroids vs random background points, sindh_test):
  yard_contrast    local S1 VH/VV contrast AT the point itself (corner-reflector
                    signature of the substation's own gantries/transformers)
  n_towers_1km     count of isolated S1 bright points (tower signature: <=3
                    neighbors/250 m, >=5 dB local contrast) in a 150-1000 m annulus
  n_bearings       corridor CONVERGENCE: those isolated points' pairwise links
                    (<500 m apart) binned into 30-deg bearing bins mod 180 -- >=2
                    distinct bearings means chains from different directions meet
                    here, which is what "towers and lines next to a substation"
                    physically looks like from orbit.
Combined score = mean rank of the three features. Reports AUC, precision/recall at
threshold quantiles, and a comparison against the segmentation model's own conf_max
on the same locations (is the SAR signal partially independent evidence, or just a
noisier copy of what the model already sees?).

Usage:
  pixi run -e ml python scripts/sar_corridor_smoke_test.py [--n-negative 3]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from subdetect.evaluate import auc, precision_at  # noqa: E402

EQ = "EPSG:6933"
REGION = ROOT / "data/osmose_regions/sindh_test"
ANNULUS_IN_M, ANNULUS_OUT_M = 150.0, 1000.0
NEG_CLEARANCE_M = 500.0


def cell_index() -> gpd.GeoDataFrame:
    from shapely.geometry import box
    rows = []
    for tif in sorted((REGION / "composites").glob("*/composite_s1.tif")):
        with rasterio.open(tif) as src:
            b = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        rows.append({"cell": tif.parent.name, "path": str(tif), "geometry": box(*b)})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def ground_truth_points() -> gpd.GeoDataFrame:
    """Real substation centroids (polys) + node points, sindh_test extent."""
    cells = cell_index()
    b = cells.total_bounds
    polys = gpd.read_parquet(ROOT / "data/labels/pakistan/substations_poly.parquet")
    polys = polys.cx[b[0]:b[2], b[1]:b[3]]
    pts = [g.centroid for g in polys.geometry]
    node_p = ROOT / "data/labels/pakistan/substations_node.parquet"
    if node_p.exists():
        nodes = gpd.read_parquet(node_p).cx[b[0]:b[2], b[1]:b[3]]
        pts += list(nodes.geometry)
    return gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326"), cells


def random_negatives(cells: gpd.GeoDataFrame, positives_eq: gpd.GeoSeries,
                     n: int, rng: np.random.Generator) -> gpd.GeoDataFrame:
    b = cells.to_crs(EQ).total_bounds
    tree = cKDTree(np.column_stack([positives_eq.x, positives_eq.y]))
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 40:
        attempts += 1
        x = rng.uniform(b[0], b[2])
        y = rng.uniform(b[1], b[3])
        d, _ = tree.query([x, y])
        if d >= NEG_CLEARANCE_M:
            pts.append((x, y))
    from shapely.geometry import Point
    gdf_eq = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in pts], crs=EQ)
    return gdf_eq.to_crs("EPSG:4326")


def local_contrast_map(band_db: np.ndarray, valid: np.ndarray) -> np.ndarray:
    band = np.where(valid, band_db, np.nan)
    local = ndimage.uniform_filter(np.nan_to_num(band), 31)
    norm = ndimage.uniform_filter(valid.astype("float32"), 31)
    local = np.where(norm > 0.3, local / np.maximum(norm, 1e-6), np.nan)
    return np.nan_to_num(band - local, nan=-99.0)


def isolated_bright_points(s1_tif: Path, contrast_db: float, max_neighbors: int) -> np.ndarray:
    with rasterio.open(s1_tif) as src:
        arr = src.read(2).astype("float32")  # VH
        valid = arr > 0
        db = arr / 500.0 - 50.0
        contrast = local_contrast_map(db, valid)
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


def bearing_diversity(center: np.ndarray, pts: np.ndarray) -> int:
    """Distinct 30-deg bearing bins among pts-within-500m pairwise links near center."""
    if len(pts) < 2:
        return 0
    bins = set()
    for i in range(len(pts)):
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        for j in np.where((d > 1.0) & (d < 500.0))[0]:
            if j <= i:
                continue
            b = np.degrees(np.arctan2(pts[j, 1] - pts[i, 1], pts[j, 0] - pts[i, 0])) % 180.0
            bins.add(int(b // 30))
    return len(bins)


def yard_contrast_at(s1_tif: Path, lon: float, lat: float) -> float:
    with rasterio.open(s1_tif) as src:
        x, y = rasterio.warp.transform("EPSG:4326", src.crs, [lon], [lat])
        c, r = ~src.transform * (x[0], y[0])
        r, c = int(round(r)), int(round(c))
        arr = src.read(2).astype("float32")
        valid = arr > 0
        if not (2 <= r < src.height - 2 and 2 <= c < src.width - 2 and valid[r, c]):
            return np.nan
        db = arr / 500.0 - 50.0
        contrast = local_contrast_map(db, valid)
        return float(contrast[r-1:r+2, c-1:c+2].max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast-db", type=float, default=5.0)
    ap.add_argument("--max-neighbors", type=int, default=3)
    ap.add_argument("--n-negative", type=int, default=3, help="negatives per positive")
    a = ap.parse_args()
    rng = np.random.default_rng(17)

    pos, cells = ground_truth_points()
    print(f"{len(pos)} real substation locations in sindh_test")
    neg = random_negatives(cells, pos.to_crs(EQ).geometry, len(pos) * a.n_negative, rng)
    print(f"{len(neg)} random background locations (>= {NEG_CLEARANCE_M:.0f} m from any real substation)")

    locs = gpd.GeoDataFrame(
        {"y": np.r_[np.ones(len(pos)), np.zeros(len(neg))]},
        geometry=list(pos.geometry) + list(neg.geometry), crs="EPSG:4326")
    joined = gpd.sjoin(locs, cells, how="inner", predicate="within")
    print(f"{len(joined)}/{len(locs)} locations fall inside a composite cell")

    # cache isolated points + s1 raster info per cell (many locations share a cell)
    cell_pts, cell_paths = {}, {}
    yard_c, n_tw, n_br = [], [], []
    for path, grp in joined.groupby("path"):
        pts = isolated_bright_points(Path(path), a.contrast_db, a.max_neighbors)
        pts_eq = gpd.GeoSeries(gpd.points_from_xy([0], [0])).to_crs(EQ)  # dummy, unused
        for idx, row in grp.iterrows():
            lonlat = row.geometry
            c = gpd.GeoSeries([lonlat], crs="EPSG:4326").to_crs(EQ).iloc[0]
            yard_c.append(yard_contrast_at(Path(path), lonlat.x, lonlat.y))
            if len(pts):
                d = np.hypot(pts[:, 0] - c.x, pts[:, 1] - c.y)
                ann = pts[(d >= ANNULUS_IN_M) & (d <= ANNULUS_OUT_M)]
                n_tw.append(len(ann))
                n_br.append(bearing_diversity(np.array([c.x, c.y]), ann))
            else:
                n_tw.append(0); n_br.append(0)

    joined = joined.reset_index(drop=True)
    joined["yard_contrast"] = yard_c
    joined["n_towers_1km"] = n_tw
    joined["n_bearings"] = n_br
    joined = joined.dropna(subset=["yard_contrast"]).reset_index(drop=True)
    y = joined.y.values.astype(bool)
    print(f"\n{len(joined)} usable locations ({int(y.sum())} real, {int((~y).sum())} background)")

    def rank01(v):
        return pd.Series(v).rank(pct=True).values

    combined = (rank01(joined.yard_contrast) + rank01(joined.n_towers_1km)
               + rank01(joined.n_bearings)) / 3.0

    print(f"\n{'signal':>16}  {'AUC':>6}  {'P@20':>5}  {'P@50':>5}")
    for name, s in [("yard_contrast", joined.yard_contrast.values),
                    ("n_towers_1km", joined.n_towers_1km.values.astype(float)),
                    ("n_bearings", joined.n_bearings.values.astype(float)),
                    ("combined (all 3)", combined)]:
        print(f"{name:>16}  {auc(s, y):6.3f}  {precision_at(s, y, 20):5.2f}  "
              f"{precision_at(s, y, 50):5.2f}")

    print(f"\nmedian by class:")
    print(f"  yard_contrast   real {joined[y].yard_contrast.median():6.2f} dB  "
          f"bg {joined[~y].yard_contrast.median():6.2f} dB")
    print(f"  n_towers_1km    real {joined[y].n_towers_1km.median():6.1f}     "
          f"bg {joined[~y].n_towers_1km.median():6.1f}")
    print(f"  n_bearings      real {joined[y].n_bearings.median():6.1f}     "
          f"bg {joined[~y].n_bearings.median():6.1f}")

    out = ROOT / "data/eval_results/sar_corridor_smoke_test.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    joined.drop(columns="geometry").to_parquet(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
