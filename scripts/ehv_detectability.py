"""Are >=400 kV transmission towers/lines detectable in S1 and/or S2 (or their fusion)?

Voltage-stratified version of the tower photometry test (see
docs/issues/s1-tower-chain-corridor-evidence.md): EHV towers (500/660 kV in Sindh)
are 50-90 m lattice structures with far larger radar cross-sections than the
132/220 kV average, so per-point S1 separability should rise with voltage; S2 at
10 m sees a tower only as a sub-pixel steel+shadow anomaly, expected ~invisible.

Tests on sindh_test (145 cells, composites on disk):
  A. Tower photometry per voltage class (tower -> nearest mapped line <=150 m):
     - S1: 3x3-max local contrast (dB vs 31 px local mean), VV and VH
     - S2: 3x3-max visible brightness, 3x3-min NDVI
     - AUC vs common background points (>=300 m from any tower/line)
     - rank fusion: mean of S1-contrast rank and S2-brightness rank
  B. Corridor enrichment per voltage class: fraction of 100 m samples along mapped
     lines with an S1 bright point (>= --contrast-db local max) within 60 m,
     against the same fraction for a control corridor offset 1 km perpendicular.
     Detecting the *line* only needs this enrichment to be high and specific.

Usage:
  pixi run -e ml python scripts/ehv_detectability.py [--contrast-db 5.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
from scipy import ndimage
from shapely.geometry import Point
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EQ = "EPSG:6933"
VCLASS = [(400_000, np.inf, ">=400kV"), (220_000, 400_000, "220kV"),
          (100_000, 220_000, "132kV"), (0, 100_000, "<=66kV")]


def vclass(v: float) -> str | None:
    if not np.isfinite(v):
        return None
    for lo, hi, name in VCLASS:
        if lo <= v < hi:
            return name
    return None


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    sc = np.r_[pos, neg]
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = sc.argsort()
    ranks = np.empty(len(sc))
    ranks[order] = np.arange(len(sc))
    return float((ranks[lab == 1].mean() - (lab.sum() - 1) / 2) / max((lab == 0).sum(), 1))


def local_contrast(band_db: np.ndarray, valid: np.ndarray) -> np.ndarray:
    band = np.where(valid, band_db, np.nan)
    local = ndimage.uniform_filter(np.nan_to_num(band), 31)
    norm = ndimage.uniform_filter(valid.astype("float32"), 31)
    local = np.where(norm > 0.3, local / np.maximum(norm, 1e-6), np.nan)
    return np.nan_to_num(band - local, nan=-99.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast-db", type=float, default=5.0)
    ap.add_argument("--cell-step", type=int, default=1, help="use every Nth cell")
    a = ap.parse_args()
    rng = np.random.default_rng(3)

    towers = []
    with (ROOT / "data/osm/pakistan_towers.geojsonseq").open() as f:
        for line in f:
            line = line.strip().lstrip("\x1e")
            if not line:
                continue
            try:
                g = json.loads(line)["geometry"]
            except Exception:
                continue
            if g and g["type"] == "Point":
                towers.append(g["coordinates"][:2])
    towers = gpd.GeoDataFrame(geometry=[Point(*c) for c in towers], crs="EPSG:4326")

    lines = gpd.read_parquet(ROOT / "data/labels/pakistan/lines.parquet")
    lines = lines.cx[66.5:70.0, 23.5:27.0].reset_index(drop=True)
    lines_eq = lines.to_crs(EQ)
    line_tree = STRtree(list(lines_eq.geometry.values))

    towers_eq = towers.to_crs(EQ)
    tclass = []
    for p in towers_eq.geometry:
        i = line_tree.nearest(p)
        if lines_eq.geometry.values[i].distance(p) <= 150.0:
            tclass.append(vclass(lines.voltage_v.values[i]))
        else:
            tclass.append(None)
    towers["cls"] = towers_eq["cls"] = tclass

    cells = sorted((ROOT / "data/osmose_regions/sindh_test/composites").glob("*"))[::a.cell_step]
    samples = []          # per-tower photometry rows
    bg = []               # background photometry rows
    corridor_hits = {}    # cls -> [hits, total, ctrl_hits, ctrl_total]
    tower_tree = STRtree(list(towers_eq.geometry.values))

    for cdir in cells:
        s1p, s2p = cdir / "composite_s1.tif", cdir / "composite_0.tif"
        if not (s1p.exists() and s2p.exists()):
            continue
        with rasterio.open(s1p) as s1, rasterio.open(s2p) as s2:
            arr1 = s1.read().astype("float32")
            valid = arr1[0] > 0
            db = arr1 / 500.0 - 50.0
            con_vv = local_contrast(db[0], valid)
            con_vh = local_contrast(db[1], valid)
            s2a = s2.read().astype("float32")
            bright = s2a[:3].mean(axis=0)
            ndvi = (s2a[6] - s2a[2]) / np.maximum(s2a[6] + s2a[2], 1.0)
            H, W = con_vv.shape
            b4326 = rasterio.warp.transform_bounds(s1.crs, "EPSG:4326", *s1.bounds)

            def px_feats(r: int, c: int) -> dict:
                sl = (slice(max(r-1, 0), r+2), slice(max(c-1, 0), c+2))
                return {"vv": float(con_vv[sl].max()), "vh": float(con_vh[sl].max()),
                        "bright": float(bright[sl].max()), "ndvi": float(ndvi[sl].min())}

            sel = towers.cx[b4326[0]:b4326[2], b4326[1]:b4326[3]]
            t_px = []
            for geom, cls in zip(sel.geometry, sel.cls):
                x, y = rasterio.warp.transform("EPSG:4326", s1.crs, [geom.x], [geom.y])
                c, r = ~s1.transform * (x[0], y[0])
                r, c = int(round(r)), int(round(c))
                if 2 <= r < H - 2 and 2 <= c < W - 2 and valid[r, c]:
                    t_px.append((r, c))
                    if cls:
                        samples.append({"cls": cls, **px_feats(r, c)})
            if t_px:
                tp = np.array(t_px)
                for _ in range(len(t_px)):
                    r = int(rng.integers(2, H - 2)); c = int(rng.integers(2, W - 2))
                    if not valid[r, c] or np.hypot(tp[:, 0]-r, tp[:, 1]-c).min() < 30:
                        continue
                    bg.append(px_feats(r, c))

            # corridor enrichment: bright local maxima -> distance test along lines
            is_max = con_vh == ndimage.maximum_filter(con_vh, 3)
            rows_, cols_ = np.where(is_max & (con_vh >= a.contrast_db) & valid)
            if len(rows_):
                xs, ys = rasterio.transform.xy(s1.transform, rows_, cols_)
                ex, ey = rasterio.warp.transform(s1.crs, EQ, xs, ys)
                pt_tree = STRtree([Point(x, y) for x, y in zip(ex, ey)])
                pts_xy = np.column_stack([ex, ey])
            else:
                pt_tree, pts_xy = None, None
            cell_box = gpd.GeoSeries([Point((b4326[0]+b4326[2])/2, (b4326[1]+b4326[3])/2)],
                                     crs="EPSG:4326").to_crs(EQ).buffer(6000).iloc[0]
            for i in line_tree.query(cell_box, predicate="intersects"):
                cls = vclass(lines.voltage_v.values[i])
                if cls is None:
                    continue
                geom = lines_eq.geometry.values[i].intersection(cell_box)
                if geom.is_empty:
                    continue
                corr = corridor_hits.setdefault(cls, [0, 0, 0, 0])
                for gpart in getattr(geom, "geoms", [geom]):
                    L = gpart.length
                    for d in np.arange(0.0, L, 100.0):
                        p = gpart.interpolate(d)
                        # control point: 1 km perpendicular offset
                        p2 = gpart.interpolate(min(d + 100.0, L))
                        dx, dy = p2.x - p.x, p2.y - p.y
                        n = np.hypot(dx, dy)
                        if n < 1.0:
                            continue
                        ctrl = Point(p.x - dy / n * 1000.0, p.y + dx / n * 1000.0)
                        if pts_xy is not None and len(pts_xy):
                            dmin = np.hypot(pts_xy[:, 0]-p.x, pts_xy[:, 1]-p.y).min()
                            corr[0] += int(dmin <= 60.0); corr[1] += 1
                            dmin_c = np.hypot(pts_xy[:, 0]-ctrl.x, pts_xy[:, 1]-ctrl.y).min()
                            corr[2] += int(dmin_c <= 60.0); corr[3] += 1
                        else:
                            corr[1] += 1; corr[3] += 1

    S = pd.DataFrame(samples)
    B = pd.DataFrame(bg)
    print(f"\n{len(S)} tower samples ({S.cls.value_counts().to_dict()}), {len(B)} background")
    print(f"\n{'class':>9} {'n':>5}  {'VHmed':>6} {'aucVH':>6} {'aucVV':>6} "
          f"{'aucS2br':>7} {'aucNDVI':>7} {'aucFUSE':>7}")
    bg_rank_src = {k: B[k].values for k in ["vv", "vh", "bright", "ndvi"]}
    for cls in [">=400kV", "220kV", "132kV", "<=66kV"]:
        sel = S[S.cls == cls]
        if len(sel) < 20:
            continue
        aucs = {}
        for k, flip in [("vh", 1), ("vv", 1), ("bright", 1), ("ndvi", -1)]:
            aucs[k] = rank_auc(flip * sel[k].values, flip * bg_rank_src[k])
        # rank fusion S1 VH + S2 brightness
        allv = np.r_[sel.vh.values, B.vh.values]
        allb = np.r_[sel.bright.values, B.bright.values]
        rv = pd.Series(allv).rank().values; rb = pd.Series(allb).rank().values
        fused = (rv + rb) / 2
        auc_f = rank_auc(fused[:len(sel)], fused[len(sel):])
        print(f"{cls:>9} {len(sel):>5}  {sel.vh.median():6.1f} {aucs['vh']:6.3f} "
              f"{aucs['vv']:6.3f} {aucs['bright']:7.3f} {aucs['ndvi']:7.3f} {auc_f:7.3f}")

    print(f"\ncorridor enrichment (S1 bright point >= {a.contrast_db} dB within 60 m):")
    print(f"{'class':>9}  {'on-line':>8}  {'control':>8}  {'lift':>5}")
    for cls in [">=400kV", "220kV", "132kV", "<=66kV"]:
        if cls not in corridor_hits:
            continue
        h, t, hc, tc = corridor_hits[cls]
        on = h / max(t, 1); ct = hc / max(tc, 1)
        print(f"{cls:>9}  {on:8.3f}  {ct:8.3f}  {on / max(ct, 1e-4):5.1f}x  (n={t})")


if __name__ == "__main__":
    main()
