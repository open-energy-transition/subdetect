"""Mine hard-negative training chips from confirmed false-positive candidates.

Manual review (2026-07-08) of the top-ranked v2_india "new" candidates found the
overwhelming majority were false positives -- mostly bare/exposed natural land
confused for a substation's gravel yard, not genuine unmapped substations. Since
none of these locations carry a substation label, chips built there get an
all-background (or -ignore, if they happen to graze another labelled feature)
mask automatically via the same `_burn_mask` production logic -- no special-
cased "negative" mask is needed. This directly teaches the model the exact
mistakes it is currently making, at zero new-imagery cost.

Usage: python scripts/mine_hard_negatives.py [n]  (default n=900, ~1x positive
chip count, per user decision to avoid overcorrecting into a recall collapse).
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subdetect.chips import (  # noqa: E402
    _bbox_poly,
    _burn_mask,
    _chip_bbox,
    _chip_id,
    _crop,
    _split_of,
    _tile_of,
    _write_tif,
)
from subdetect.config import CHIP_SIZE, MODEL_BANDS, Settings, resolve_aoi  # noqa: E402
from subdetect.local_source import CompositeIndex, load_substation_labels  # noqa: E402

AOI = "pakistan"
DEDUP_DEG = 0.02  # ~2 km, matches the positive-chip dedup grid in chips.py


def main(n_target: int) -> None:
    settings = Settings.load()
    _, cfg = resolve_aoi(AOI, settings)

    cands = gpd.read_parquet(f"data/predictions_v2_india/{AOI}/candidates.parquet")
    fp = cands[cands.status == "new"].sort_values("confidence", ascending=False).copy()
    fp["lon"] = fp.geometry.centroid.x
    fp["lat"] = fp.geometry.centroid.y
    cell = ((fp.lon / DEDUP_DEG).round().astype(int).astype(str) + "_"
            + (fp.lat / DEDUP_DEG).round().astype(int).astype(str))
    fp = fp.loc[~cell.duplicated()].reset_index(drop=True).head(n_target)
    print(f"Mining {len(fp)} hard-negative chips "
          f"(deduped from {(cands.status == 'new').sum()} false positives, "
          f"confidence range {fp.confidence.min():.3f}-{fp.confidence.max():.3f})")

    out_dir = ROOT / "data" / "chips" / "pakistan_hardneg"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    comp_idx = CompositeIndex(ROOT / "data" / "composites" / AOI)
    labels = load_substation_labels(ROOT / "data" / "labels" / AOI, min_area_m2=settings.min_sub_area_m2)
    n_bands = len(MODEL_BANDS)

    records = []
    for _, row in tqdm(list(fp.iterrows()), total=len(fp), desc="hard_negatives"):
        lon, lat = float(row.lon), float(row.lat)
        cid = _chip_id(lon, lat)
        img_path = out_dir / "images" / f"{cid}.tif"
        mask_path = out_dir / "masks" / f"{cid}.tif"
        tile = _tile_of(lon, lat, comp_idx)
        try:
            if not (img_path.exists() and mask_path.exists()):
                res = comp_idx.read_window(_chip_bbox(lon, lat))
                if res is None:
                    continue
                arr, transform, crs = res
                arr = arr[:n_bands]
                arr, ox, oy = _crop(arr, CHIP_SIZE)
                transform = transform * rasterio.Affine.translation(ox, oy)
                win_labels = labels[labels.geometry.intersects(_bbox_poly(lon, lat))]
                mask = _burn_mask(win_labels, transform, crs, arr.shape[-2:],
                                  settings.min_sub_area_m2, settings.node_ignore_radius_m)
                _write_tif(img_path, arr, transform, crs, "uint16")
                _write_tif(mask_path, mask.astype("int16"), transform, crs, "int16")
        except Exception as e:  # noqa: BLE001 -- one bad chip must not kill the run
            print(f"chip {cid} failed: {e}")
            continue
        with rasterio.open(mask_path) as m:
            band = m.read(1)
        records.append(dict(chip_id=cid, lon=lon, lat=lat, kind="hard_negative", tile=tile,
                             split=_split_of(lon, lat, cfg), sub_pixels=int((band == 1).sum()),
                             image=str(img_path), s1=None, mask=str(mask_path)))

    index = pd.DataFrame(records)
    index_path = out_dir / "index.parquet"
    index.to_parquet(index_path)
    n_actually_positive = int((index.sub_pixels > 0).sum()) if len(index) else 0
    print(f"Wrote {len(index)} hard-negative chips -> {index_path}")
    if n_actually_positive:
        print(f"  note: {n_actually_positive} accidentally overlap a real substation label "
              "(burned correctly as positive, harmless but not a true hard negative)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 900)
