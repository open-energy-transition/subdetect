"""Filter-only arm of the ESRI core-label study: drop bad OSM labels, keep good ones whole.

The core rewrite (esri_core_labels.py) trimmed every polygon to its Esri-visible core. This
script uses the same per-target core measurement only as a FILTER:

  core share = core_px / orig_px     (from esri_core_labels.py's core_stats.parquet)
  core share <  --min-core-share     -> the whole target becomes ignore (-1)
  core share >= --min-core-share     -> the target keeps its original OSM burn, untouched
  status no_imagery                  -> kept (nothing to judge it by)

"no_core" targets have core share 0, so they are always dropped. Everything else in the
chip (background, other targets, existing ignore pixels) is untouched, so the only
difference from the control arm is which targets are ignored. Masks are written in the
SOURCE chip orientation, and an index is written alongside that repoints only the chips
that changed.

Usage:
  pixi run -e ml python scripts/esri_core_filter_labels.py \
      --index data/chips_v16_honestburn/combined/index.parquet --out-root data/chips_v17_filter
  pixi run -e ml python scripts/esri_core_filter_labels.py \
      --index data/chips_v5_honestburn/combined/index.parquet --out-root data/chips_v5_filter \
      --aoi pakistan india_pilot pakistan_hardneg
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from esri_core_targets import GROUP_DILATE_PX  # noqa: E402
from subdetect.core_labels import is_flipped, write_label  # noqa: E402

CORE = ROOT / "data/esri_core"


def _chip(job) -> dict:
    aoi, chip_id, mask_path, groups, out_root = job
    fl = is_flipped(aoi)
    with rasterio.open(ROOT / mask_path) as src:
        m = src.read(1)
    m = m[::-1].copy() if fl else m.copy()
    pos = m == 1
    # same grouping as esri_core_targets.py / esri_core_labels.py, so group ids match
    lab, _ = ndimage.label(ndimage.binary_dilation(pos, iterations=GROUP_DILATE_PX),
                           structure=np.ones((3, 3)))
    drop = pos & np.isin(lab, groups)
    new = m.copy()
    new[drop] = -1
    out = Path(out_root) / "masks" / aoi / f"{chip_id}.tif"
    write_label(ROOT / mask_path, out, new, fl)
    return {"aoi": aoi, "chip_id": chip_id, "dropped_px": int(drop.sum()),
            "mask": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
            "sub_pixels": int((new == 1).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--stats", type=Path, default=ROOT / "data/chips_v17_core/core_stats.parquet",
                    help="per-target core stats from esri_core_labels.py (thr 0.76, dilate 0)")
    ap.add_argument("--min-core-share", type=float, default=0.10)
    ap.add_argument("--aoi", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    t = pd.read_parquet(CORE / "targets.parquet")
    st = pd.read_parquet(a.stats)[["target_id", "status", "orig_px", "core_px"]]
    t = t.merge(st, on="target_id", how="left", validate="one_to_one")
    if a.aoi:
        t = t[t.aoi.isin(a.aoi)]
    t["core_share"] = t.core_px / t.orig_px.clip(lower=1)
    t["drop"] = (t.status != "no_imagery") & t.status.notna() & (t.core_share < a.min_core_share)

    jobs = [(aoi, cid, mp, g.group.astype(int).tolist(), str(a.out_root))
            for (aoi, cid, mp), g in t[t["drop"]].groupby(["aoi", "chip_id", "mask"])]
    print(f"{int(t['drop'].sum())} of {len(t)} targets dropped (core share < {a.min_core_share}), "
          f"{len(jobs)} chips rewritten")
    remap = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for r in tqdm(ex.map(_chip, jobs, chunksize=16), total=len(jobs), mininterval=20):
            remap[(r["aoi"], r["chip_id"])] = r
    a.out_root.mkdir(parents=True, exist_ok=True)
    t.drop(columns=["crs"], errors="ignore").to_parquet(a.out_root / "filter_stats.parquet")

    idx = pd.read_parquet(a.index)
    keys = list(zip(idx.aoi, idx.chip_id))
    hit = np.array([k in remap for k in keys])
    idx.loc[hit, "mask"] = [remap[k]["mask"] for k, h in zip(keys, hit) if h]
    idx.loc[hit, "sub_pixels"] = [remap[k]["sub_pixels"] for k, h in zip(keys, hit) if h]
    (a.out_root / "combined").mkdir(parents=True, exist_ok=True)
    idx.to_parquet(a.out_root / "combined" / "index.parquet")
    missing = len(set(remap) - set(keys))  # the index repeats upsampled chips, so count keys
    if missing:
        print(f"WARNING: {missing} rewritten chips are not in {a.index}")

    dropped_px = sum(r["dropped_px"] for r in remap.values())
    for aoi, g in t.groupby("aoi"):
        d = g[g["drop"]]
        print(f"{aoi:22s} targets {len(g):6d}  dropped {len(d):6d} ({100 * len(d) / len(g):4.1f}%)  "
              f"positive px dropped {100 * d.orig_px.sum() / max(g.orig_px.sum(), 1):4.1f}%")
    print(f"pixels set to ignore: {dropped_px:,} of {int(t.orig_px.sum()):,} target px "
          f"({100 * dropped_px / max(t.orig_px.sum(), 1):.1f}%); chips repointed in index: {int(hit.sum())}")


if __name__ == "__main__":
    main()
