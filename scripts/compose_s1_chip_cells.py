"""Composite S1 for exactly the cells that contain training chips (both AOIs).

Same output contract as `compose --sensor s1` (composite_s1.tif pinned to the cell's
S2 geobox, skip-if-exists) so it cooperates with any full-ROI compose run.
Priority order: pakistan val-split cells first (smallest, unblocks evaluation),
then pakistan train + hardneg, then india_pilot.

Usage: pixi run python scripts/compose_s1_chip_cells.py [--workers 6]
"""
import argparse, logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import rasterio

import sys
sys.path.insert(0, "src")
from subdetect.imagery import s1_composite
from subdetect.config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("s1chip")

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=6)
a = ap.parse_args()

settings = Settings.load()
s1_window = tuple(getattr(settings, "s1_window", ("2025-11-01", "2026-03-15")))

# (aoi, index, split-priority)
jobs = []
for aoi, idx_path in (("pakistan", "data/chips/pakistan/index.parquet"),
                      ("pakistan", "data/chips/pakistan_hardneg/index.parquet"),
                      ("india_pilot", "data/chips/india_pilot/index.parquet")):
    idx = pd.read_parquet(idx_path)
    for tile, split in idx.groupby("tile")["split"].first().items():
        jobs.append((0 if (aoi == "pakistan" and split == "val") else 1, aoi, tile))
seen = set()
ordered = []
for prio, aoi, tile in sorted(jobs):
    if (aoi, tile) in seen:
        continue
    seen.add((aoi, tile))
    ordered.append((aoi, tile))
log.info("%d unique chip cells", len(ordered))

def one(job):
    aoi, tile = job
    cell_dir = Path("data/composites") / aoi / "composites" / tile
    out = cell_dir / "composite_s1.tif"
    if out.exists():
        return "skip"
    base = cell_dir / "composite_0.tif"
    if not base.exists():
        return "no_s2"
    from odc.geo.geobox import GeoBox
    with rasterio.open(base) as b:
        gbox = GeoBox((b.height, b.width), b.transform, b.crs)
        bounds = rasterio.warp.transform_bounds(b.crs, "EPSG:4326", *b.bounds)
    try:
        res = s1_composite(tuple(bounds), date_range=s1_window, geobox=gbox)
    except Exception as e:  # noqa: BLE001
        log.warning("cell %s failed: %s", tile, str(e)[:100])
        return "err"
    if res is None:
        return "empty"
    arr, transform, crs = res
    tmp = out.with_suffix(".tif.tmp")
    with rasterio.open(tmp, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                       count=arr.shape[0], dtype="uint16", crs=crs, transform=transform,
                       compress="deflate", predictor=2) as dst:
        dst.write(arr)
    tmp.rename(out)
    return "ok"

import rasterio.warp
done = 0
from collections import Counter
stats = Counter()
with ThreadPoolExecutor(a.workers) as ex:
    for r in ex.map(one, ordered):
        stats[r] += 1
        done += 1
        if done % 25 == 0:
            log.info("%d/%d cells (%s)", done, len(ordered), dict(stats))
log.info("DONE: %s", dict(stats))
