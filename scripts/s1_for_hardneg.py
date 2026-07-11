"""Add co-registered S1 chips to the hard-negative chip set.

The hardneg chips were built by mine_hard_negatives.py (not the chips CLI), so
`chips --s1` can't refresh them. Mirrors chips.py's S1 block: reads a
composite_s1.tif window at each chip's bbox from the pakistan composites,
writes s1/<chip_id>.tif, and fills the index's `s1` column. Resumable.
"""
import sys, logging
from pathlib import Path
import pandas as pd
import rasterio

sys.path.insert(0, "src")
from subdetect.chips import _chip_bbox, _crop, _write_tif, CHIP_SIZE
from subdetect.config import S1_BANDS
from subdetect.local_source import CompositeIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("s1hardneg")

chip_dir = Path("data/chips/pakistan_hardneg")
(chip_dir / "s1").mkdir(exist_ok=True)
idx = pd.read_parquet(chip_dir / "index.parquet")
comp_idx = CompositeIndex(Path("data/composites/pakistan"))

s1_col = []
n_ok = n_skip = n_miss = 0
for row in idx.itertuples():
    s1_path = chip_dir / "s1" / f"{row.chip_id}.tif"
    rel = str(s1_path)
    if s1_path.exists():
        s1_col.append(rel); n_skip += 1
        continue
    try:
        res = comp_idx.read_window(_chip_bbox(row.lon, row.lat), "composite_s1.tif")
    except FileNotFoundError:
        res = None
    if res is None:
        s1_col.append(None); n_miss += 1
        continue
    arr, transform, crs = res
    arr = arr[: len(S1_BANDS)]
    arr, ox, oy = _crop(arr, CHIP_SIZE)
    transform = transform * rasterio.Affine.translation(ox, oy)
    _write_tif(s1_path, arr, transform, crs, "uint16")
    s1_col.append(rel); n_ok += 1

idx["s1"] = s1_col
n_before = len(idx)
idx = idx[idx.s1.notna()].reset_index(drop=True)  # a null s1 path would crash the dual loader
idx.to_parquet(chip_dir / "index.parquet")
log.info("hardneg S1 chips: %d written, %d existing, %d missing composites", n_ok, n_skip, n_miss)
if n_miss:
    log.warning("dropped %d/%d hardneg chips lacking S1 composites from the index", n_before - len(idx), n_before)
