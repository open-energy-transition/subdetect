"""Stage 0 of the ESRI core-label study: inventory what has to be looked at.

One row per *substation group* per positive chip of the source corpus (default: the
v16_honestburn index, i.e. v11b with OSM chips re-burned centre-in). A group is a set of
positive mask pixels whose 50 m dilations touch -- one substation, or a few adjacent
polygons that make up one site. Working from the chip MASK rather than from OSM polygons
is deliberate: 26.5k of the corpus's 29.5k positive chips are TorchGeo chips whose source
polygons are not on disk, and the mask is what the model is actually trained on.

Each row carries the crop window (NORTH-UP chip pixel coordinates, positives + context
margin) that the fetch stage turns into an Esri image on the chip's own UTM grid. TorchGeo
chips are stored south-up under a north-up transform (see subdetect.core_labels), so their
masks are flipped before grouping and the row is marked `flipped=True`.

Usage:
  pixi run -e ml python scripts/esri_core_targets.py
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import ndimage
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subdetect.core_labels import is_flipped  # noqa: E402
GROUP_DILATE_PX = 5      # 50 m: polygons closer than this are one site
MARGIN_MIN_PX = 8        # >= 80 m of context round every group
MARGIN_FRAC = 0.5        # ... or half the group's extent, whichever is larger


def _groups(mask_path: str, flipped: bool) -> list[dict]:
    """Groups in NORTH-UP chip pixel coordinates (TorchGeo chips are stored south-up)."""
    with rasterio.open(mask_path) as src:
        m = src.read(1)
        tr, crs = src.transform, src.crs
    if flipped:
        m = m[::-1]
    pos = m == 1
    if not pos.any():
        return []
    lab, n = ndimage.label(ndimage.binary_dilation(pos, iterations=GROUP_DILATE_PX),
                           structure=np.ones((3, 3)))
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    H, W = m.shape
    out = []
    for g in range(1, n + 1):
        rr, cc = np.nonzero(pos & (lab == g))
        if rr.size == 0:
            continue
        r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
        ext = max(r1 - r0, c1 - c0)
        mg = max(MARGIN_MIN_PX, int(np.ceil(MARGIN_FRAC * ext)))
        cy, cx = tr * ((c0 + c1) / 2, (r0 + r1) / 2)
        lon, lat = to_ll.transform(cy, cx)
        out.append(dict(
            group=g, n_px=int(rr.size), extent_px=int(ext),
            pr0=int(r0), pr1=int(r1), pc0=int(c0), pc1=int(c1),
            r0=int(max(r0 - mg, 0)), r1=int(min(r1 + mg, H)),
            c0=int(max(c0 - mg, 0)), c1=int(min(c1 + mg, W)),
            lon=float(lon), lat=float(lat), crs=str(crs),
        ))
    return out


def _one(args):
    aoi, chip_id, mask, image, split = args
    try:
        rows = _groups(mask, is_flipped(aoi))
    except Exception as e:  # noqa: BLE001
        print(f"WARN {chip_id}: {e}")
        return []
    for r in rows:
        r.update(aoi=aoi, chip_id=chip_id, mask=mask, image=image, split=split,
                 flipped=is_flipped(aoi),
                 target_id=f"{chip_id}_{r['group']}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path,
                    default=ROOT / "data/chips_v16_honestburn/combined/index.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "data/esri_core/targets.parquet")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    idx = pd.read_parquet(a.index)
    chips = idx[idx.sub_pixels > 0].drop_duplicates(["aoi", "chip_id"])
    print(f"{len(chips)} positive chips:\n{chips.aoi.value_counts().to_string()}")
    jobs = list(zip(chips.aoi, chips.chip_id, chips["mask"], chips.image, chips.split))
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in tqdm(ex.map(_one, jobs, chunksize=64), total=len(jobs)):
            rows.extend(r)
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)
    print(f"{len(df)} groups -> {a.out}")
    print(df.groupby("aoi").agg(groups=("target_id", "size"), med_px=("n_px", "median"),
                                med_ext=("extent_px", "median")).to_string())
    print("groups/chip:", df.groupby("chip_id").size().value_counts().head(6).to_dict())


if __name__ == "__main__":
    main()
