"""Settlement/construction hard-negative chips for Pakistan + India, mined from OSM.

Field validation of the v9 leads (2026-07-19) found the dominant false-positive
classes are small villages, construction sites and bare land -- none of which the
94%-positive v9 chip index ever shows the model as labeled background. This script
mines exactly those confusers from the cached Geofabrik PBFs (data/osm/) and cuts
all-background chips from the composites already on disk, so no imagery download is
needed (unlike scripts/build_global_hardneg_chips.py, which composes via STAC).

Classes mined: place=village|hamlet (nodes+polys), landuse=residential,
landuse=construction. Safety against "negative" chips containing real-but-unmapped
substations (this is an under-mapped region -- the reason the global script stuck to
well-mapped cities): candidate centers must be >= 1600 m from every OSM substation
(poly or node, bigger than a chip half-diagonal) AND >= 500 m from every mapped
power line (unmapped substations sit on lines). Candidates are deduped on a 1.5 km
grid so chips barely overlap.

Usage:
  pixi run -e ml python scripts/build_settlement_hardneg_chips.py \
      [--out data/chips_settlement_hardneg] [--per-class-cap village=1800,residential=900,construction=500]
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
from rasterio.windows import Window
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subdetect.chips import _chip_id  # noqa: E402
from subdetect.config import CHIP_SIZE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("settlement_hardneg")

EQ = "EPSG:6933"
SUB_CLEARANCE_M = 1600.0   # > chip half-diagonal (224 px * 10 m -> 1583 m)
LINE_CLEARANCE_M = 500.0   # unmapped substations sit on mapped lines
DEDUP_GRID_M = 1500.0
VAL_FRAC = 0.1

AOIS = {
    "pakistan": ROOT / "data/osm/pakistan-latest.osm.pbf",
    "india_pilot": ROOT / "data/osm/india_pilot_all.osm.pbf",
}
TAG_CLASSES = {  # OSM tag -> chip class
    ("place", "village"): "village",
    ("place", "hamlet"): "village",
    ("landuse", "residential"): "residential",
    ("landuse", "construction"): "construction",
}


def extract_settlements(aoi: str, pbf: Path) -> gpd.GeoDataFrame:
    """osmium tags-filter + export -> one point per settlement/landuse feature."""
    osm_dir = ROOT / "data/osm"
    filt = osm_dir / f"{aoi}_settlements.osm.pbf"
    seq = osm_dir / f"{aoi}_settlements.geojsonseq"
    if not filt.exists():
        subprocess.run(["osmium", "tags-filter", "-o", str(filt), "--overwrite", str(pbf),
                        "nwr/place=village,hamlet", "nwr/landuse=residential,construction"],
                       check=True)
    if not seq.exists():
        cfg = osm_dir / "settlement_export_config.json"
        cfg.write_text(json.dumps({"attributes": {"type": "@type", "id": "@id"},
                                   "linear_tags": False, "area_tags": True,
                                   "include_tags": ["place", "landuse"]}))
        subprocess.run(["osmium", "export", str(filt), "-f", "geojsonseq", "--overwrite",
                        "-c", str(cfg), "-o", str(seq)], check=True)
    rows = []
    with seq.open() as f:
        for line in f:
            line = line.strip().lstrip("\x1e")
            if not line:
                continue
            try:
                feat = json.loads(line)
            except json.JSONDecodeError:
                continue
            props = feat.get("properties", {})
            cls = None
            for (k, v), c in TAG_CLASSES.items():
                if props.get(k) == v:
                    cls = c
                    break
            if cls is None:
                continue
            try:
                geom = shape(feat["geometry"])
            except Exception:  # noqa: BLE001 — skip broken geometries
                continue
            pt = geom if geom.geom_type == "Point" else geom.representative_point()
            rows.append({"cls": cls, "geometry": pt})
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    log.info("%s: %d settlement features (%s)", aoi, len(gdf),
             gdf.cls.value_counts().to_dict() if len(gdf) else {})
    return gdf


def composite_cells(aoi: str) -> gpd.GeoDataFrame:
    """WGS84 bounds polygon per composite cell (chip must fit fully inside)."""
    from shapely.geometry import box
    rows = []
    for tif in sorted((ROOT / "data/composites" / aoi / "composites").glob("*/composite_0.tif")):
        with rasterio.open(tif) as src:
            b = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        rows.append({"cell": tif.parent.name, "path": str(tif), "geometry": box(*b)})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def clearance_filter(aoi: str, pts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    labels = ROOT / "data/labels" / aoi
    subs = gpd.read_parquet(labels / "substations_poly.parquet")[["geometry"]]
    nodes_p = labels / "substations_node.parquet"
    if nodes_p.exists():
        subs = pd.concat([subs, gpd.read_parquet(nodes_p)[["geometry"]]], ignore_index=True)
    subs_eq = gpd.GeoDataFrame(subs, crs="EPSG:4326").to_crs(EQ)
    sub_tree = STRtree(list(subs_eq.geometry.values))
    lines_eq = gpd.read_parquet(labels / "lines.parquet").to_crs(EQ)
    line_tree = STRtree(list(lines_eq.geometry.values))

    pts_eq = pts.to_crs(EQ)
    keep = []
    for geom in pts_eq.geometry:
        near_sub = sub_tree.query(geom.buffer(SUB_CLEARANCE_M), predicate="intersects")
        near_line = line_tree.query(geom.buffer(LINE_CLEARANCE_M), predicate="intersects")
        keep.append(len(near_sub) == 0 and len(near_line) == 0)
    out = pts[np.array(keep)].reset_index(drop=True)
    log.info("%s: clearance kept %d/%d (%s)", aoi, len(out), len(pts),
             out.cls.value_counts().to_dict() if len(out) else {})
    return out


def dedup_and_cap(pts: gpd.GeoDataFrame, caps: dict, rng: np.random.Generator) -> gpd.GeoDataFrame:
    pts_eq = pts.to_crs(EQ)
    key = (pts_eq.geometry.x // DEDUP_GRID_M).astype(int).astype(str) + "_" + \
          (pts_eq.geometry.y // DEDUP_GRID_M).astype(int).astype(str)
    pts = pts.assign(_grid=key.values).drop_duplicates("_grid").drop(columns="_grid")
    parts = []
    for cls, cap in caps.items():
        sel = pts[pts.cls == cls]
        if len(sel) > cap:
            sel = sel.sample(cap, random_state=int(rng.integers(1 << 31)))
        parts.append(sel)
    return pd.concat(parts, ignore_index=True) if parts else pts.iloc[:0]


def cut_chips(aoi: str, pts: gpd.GeoDataFrame, cells: gpd.GeoDataFrame,
              out_dir: Path, rng: np.random.Generator) -> list[dict]:
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    joined = gpd.sjoin(pts, cells, how="inner", predicate="within")
    log.info("%s: %d candidates fall inside composite cells", aoi, len(joined))
    records, seen = [], set()
    by_path = {}
    for _, row in joined.iterrows():
        by_path.setdefault(row.path, []).append(row)
    for path, rows in by_path.items():
        with rasterio.open(path) as src:
            H, W = src.height, src.width
            for row in rows:
                x, y = rasterio.warp.transform("EPSG:4326", src.crs,
                                               [row.geometry.x], [row.geometry.y])
                col, r = ~src.transform * (x[0], y[0])
                r0 = int(np.clip(round(r) - CHIP_SIZE // 2, 0, max(H - CHIP_SIZE, 0)))
                c0 = int(np.clip(round(col) - CHIP_SIZE // 2, 0, max(W - CHIP_SIZE, 0)))
                if H < CHIP_SIZE or W < CHIP_SIZE:
                    continue
                arr = src.read(window=Window(c0, r0, CHIP_SIZE, CHIP_SIZE)).astype("uint16")[:10]
                if (arr > 0).mean() < 0.5:
                    continue
                cid = _chip_id(row.geometry.x, row.geometry.y)
                if cid in seen:
                    continue
                seen.add(cid)
                transform = src.window_transform(Window(c0, r0, CHIP_SIZE, CHIP_SIZE))
                img_p = out_dir / "images" / f"{cid}.tif"
                mask_p = out_dir / "masks" / f"{cid}.tif"
                prof = dict(driver="GTiff", width=CHIP_SIZE, height=CHIP_SIZE,
                            crs=src.crs, transform=transform, compress="deflate", predictor=2)
                with rasterio.open(img_p, "w", count=10, dtype="uint16", **prof) as dst:
                    dst.write(arr)
                with rasterio.open(mask_p, "w", count=1, dtype="int16", **prof) as dst:
                    dst.write(np.zeros((1, CHIP_SIZE, CHIP_SIZE), "int16"))
                records.append({
                    "chip_id": cid, "lon": row.geometry.x, "lat": row.geometry.y,
                    "kind": "hard_negative", "tile": f"{aoi}_{row.cls}",
                    "split": "val" if rng.random() < VAL_FRAC else "train",
                    "sub_pixels": 0, "image": str(img_p.relative_to(ROOT)), "s1": None,
                    "mask": str(mask_p.relative_to(ROOT)), "aoi": "settlement_hardneg",
                })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/chips_settlement_hardneg")
    ap.add_argument("--per-class-cap", default="village=1800,residential=900,construction=500")
    a = ap.parse_args()
    caps = {k: int(v) for k, v in (kv.split("=") for kv in a.per_class_cap.split(","))}
    out_dir = ROOT / a.out
    rng = np.random.default_rng(42)

    all_records = []
    for aoi, pbf in AOIS.items():
        pts = extract_settlements(aoi, pbf)
        cells = composite_cells(aoi)
        pts = pts[pts.within(cells.union_all())].reset_index(drop=True)
        log.info("%s: %d features within composite coverage", aoi, len(pts))
        pts = clearance_filter(aoi, pts)
        pts = dedup_and_cap(pts, caps, rng)
        log.info("%s: %d after dedup+cap", aoi, len(pts))
        all_records += cut_chips(aoi, gpd.GeoDataFrame(pts, crs="EPSG:4326"), cells, out_dir, rng)

    idx = pd.DataFrame(all_records)
    idx.to_parquet(out_dir / "index.parquet")
    log.info("wrote %d chips -> %s (%s; val=%d)", len(idx), out_dir / "index.parquet",
             idx.tile.value_counts().to_dict(), int((idx.split == "val").sum()))


if __name__ == "__main__":
    main()
