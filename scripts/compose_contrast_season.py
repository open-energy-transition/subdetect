"""Build a post-monsoon contrast-season S2 composite (composite_1.tif) for the
Pakistan cells our substation training/val chips actually use, to test whether a
two-season [dry, post-monsoon] stack helps distinguish substations from bare-land
false positives (hypothesis: substation yards stay bare year-round, natural bare
land may green up post-monsoon; earthpv found the opposite for PV -- see its
README "Two-season stacking experiment (negative result)" -- but that was a
different target class and thinner data, so worth testing directly for subs).

Pins composite_1 to composite_0's exact geobox (same pattern as subdetect's S1
compose and earthpv's own contrast-season compose), so the two layers are
pixel-aligned for channel-stacking. Reuses earthpv's already-composited
composite_1.tif via hardlink where available (same window); STAC-composites the
rest. Resumable (skip cells that already have composite_1.tif).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import rasterio
import rasterio.warp
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subdetect.config import LOCAL_BANDS  # noqa: E402
from subdetect.imagery import annual_composite  # noqa: E402

AOI = "pakistan"
CONTRAST_WINDOW = ("2025-09-20", "2025-11-15")  # matches earthpv's own Pakistan/Punjab stack_window
EARTHPV_SOURCES = [
    Path("/run/media/tobi/aidisc/earthpv/data/composites/pakistan/composites"),
    Path("/run/media/tobi/aidisc/earthpv/data/composites/punjab/composites"),
]


# Scaled-down scope (2026-07-09): a full 505-cell run measured at ~104s/cell fresh
# STAC compose (cloudy post-monsoon window -> heavy scene scanning), i.e. 10+ hours
# for the full set -- too expensive given earthpv's own same-backbone/same-terrain
# prior result was already negative. This selects all 19 val tiles (needed for any
# comparison at all) plus a capped sample of positive/hard-negative tiles, sized for
# a directionally-informative (not fully powered) ~1.5-3h test.
TARGET_POS = 60
TARGET_HN = 40


def target_cells() -> set[str]:
    pos = pd.read_parquet(ROOT / "data/chips/pakistan/index.parquet")
    hn = pd.read_parquet(ROOT / "data/chips/pakistan_hardneg/index.parquet")
    val_tiles = set(pos[pos.split == "val"].tile.dropna())
    pos_tiles = set(pos[pos.sub_pixels > 0].tile.dropna())
    hn_tiles = set(hn.tile.dropna())

    selected = set(val_tiles)
    pos_pool = sorted(pos_tiles - val_tiles)
    selected |= set(pos_pool[:TARGET_POS])
    hn_pool = sorted(hn_tiles - selected)
    selected |= set(hn_pool[:TARGET_HN])
    return selected


def main(workers: int = 4) -> None:
    comp_dir = ROOT / "data/composites" / AOI / "composites"
    cells = sorted(target_cells())
    print(f"Target cells: {len(cells)}")

    linked = fresh = skipped = failed = 0

    def _one(name: str) -> str:
        nonlocal linked, fresh, skipped, failed
        cell_dir = comp_dir / name
        dst = cell_dir / "composite_1.tif"
        base = cell_dir / "composite_0.tif"
        if dst.exists():
            return "skip"
        if not base.exists():
            return "fail"
        for src_root in EARTHPV_SOURCES:
            src = src_root / name / "composite_1.tif"
            if src.exists():
                try:
                    os.link(src, dst)
                    return "link"
                except OSError:
                    pass  # fall through to fresh compose
        try:
            with rasterio.open(base) as b:
                bounds4326 = rasterio.warp.transform_bounds(b.crs, "EPSG:4326", *b.bounds)
                from odc.geo.geobox import GeoBox

                gbox = GeoBox((b.height, b.width), b.transform, b.crs)
            res = annual_composite(bounds4326, date_range=CONTRAST_WINDOW, geobox=gbox, max_cloud=60)
        except Exception as e:  # noqa: BLE001 -- one bad cell must not kill the run
            print(f"  {name} failed: {e}")
            return "fail"
        if res is None:
            return "fail"
        arr, transform, crs = res
        tmp = dst.with_suffix(".tif.tmp")
        with rasterio.open(
            tmp, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
            dtype="uint16", crs=crs, transform=transform, compress="deflate", predictor=2,
        ) as out:
            out.write(arr)
            out.descriptions = tuple(LOCAL_BANDS)
        tmp.rename(dst)
        return "fresh"

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for r in tqdm(ex.map(_one, cells), total=len(cells), desc="contrast-season"):
                if r == "link":
                    linked += 1
                elif r == "fresh":
                    fresh += 1
                elif r == "skip":
                    skipped += 1
                else:
                    failed += 1
    else:
        for name in tqdm(cells, desc="contrast-season"):
            r = _one(name)
            if r == "link":
                linked += 1
            elif r == "fresh":
                fresh += 1
            elif r == "skip":
                skipped += 1
            else:
                failed += 1

    print(f"linked {linked}, fresh-composed {fresh}, already present {skipped}, failed {failed}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
