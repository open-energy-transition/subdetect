"""Mine hard-negative training chips from the Yunnan osmose pilot's low-confidence leads.

Unlike `mine_hard_negatives.py` (Pakistan), these candidates have NOT been manually
reviewed -- the pilot is fresh, unlabeled territory. Treating the whole lead set as
false positives would risk teaching the model to suppress genuine unmapped substations,
which is the entire point of the osmose lead-generation run. Instead this mines only the
bottom half of the pilot's `rank_score` (low confidence and/or far from any osmose
endpoint) as the hard-negative pool -- the detections a reviewer would deprioritize
anyway -- and still checks each chip against real OSM substations (fetched fresh over
the pilot's coverage) so a false positive that happens to sit on a genuine mapped
substation is burned "ignore", not "background".

Usage: python scripts/mine_hard_negatives_yunnan.py [n]  (default n=300, ~half the
pilot's 594 leads, capped to the low-rank half regardless of n)
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from subdetect.chips import _bbox_poly, _burn_mask, _chip_bbox, _chip_id, _crop, _write_tif  # noqa: E402
from subdetect.config import CHIP_SIZE, MODEL_BANDS, S1_BANDS, Settings  # noqa: E402
from subdetect.local_source import CompositeIndex  # noqa: E402
from osmose_detect import fetch_substations  # noqa: E402

REGION = "yunnan"
DEDUP_DEG = 0.02  # ~2 km, matches the positive-chip dedup grid in chips.py


def _osm_labels_as_ignore(region_dir: Path) -> gpd.GeoDataFrame:
    """Real OSM substations over the pilot's coverage, shaped like load_substation_labels'
    output (role/area_m2 columns) so _burn_mask can reuse it directly. Points only (no
    polygon geometry from Overpass `out center`), so treated as role="node" -> ignore disc,
    same convention chips.py already uses for node-mapped substations."""
    comp_idx = CompositeIndex(region_dir)
    b = comp_idx.coverage.bounds
    subs = fetch_substations((b[0] - 0.05, b[1] - 0.05, b[2] + 0.05, b[3] + 0.05))
    subs = subs.assign(area_m2=0.0, voltage_v=float("nan"), role="node")
    return subs


def main(n_target: int) -> None:
    settings = Settings.load()
    region_dir = ROOT / "data" / "osmose_regions" / REGION

    leads = gpd.read_file(region_dir / "leads_pilot.geojson")
    leads = leads.sort_values("rank_score", ascending=False).reset_index(drop=True)
    low_half = leads.iloc[len(leads) // 2:].copy()  # bottom half by rank_score only
    low_half["lon"] = low_half.geometry.centroid.x
    low_half["lat"] = low_half.geometry.centroid.y
    cell = ((low_half.lon / DEDUP_DEG).round().astype(int).astype(str) + "_"
            + (low_half.lat / DEDUP_DEG).round().astype(int).astype(str))
    low_half = low_half.loc[~cell.duplicated()].reset_index(drop=True).head(n_target)
    print(f"Mining {len(low_half)} hard-negative chips from the bottom half of "
          f"{len(leads)} pilot leads by rank_score (rank_score range "
          f"{low_half.rank_score.min():.4f}-{low_half.rank_score.max():.4f}, "
          f"confidence range {low_half.confidence.min():.3f}-{low_half.confidence.max():.3f})")

    out_dir = ROOT / "data" / "chips" / "yunnan_hardneg"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "s1").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    comp_idx = CompositeIndex(region_dir)
    labels = _osm_labels_as_ignore(region_dir)
    print(f"Loaded {len(labels)} real OSM substations over the pilot coverage as an "
          f"ignore safety net")
    n_bands = len(MODEL_BANDS)

    records = []
    for _, row in tqdm(list(low_half.iterrows()), total=len(low_half), desc="hard_negatives"):
        lon, lat = float(row.lon), float(row.lat)
        cid = _chip_id(lon, lat)
        img_path = out_dir / "images" / f"{cid}.tif"
        s1_path = out_dir / "s1" / f"{cid}.tif"
        mask_path = out_dir / "masks" / f"{cid}.tif"
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
            if not s1_path.exists():
                s1res = comp_idx.read_window(_chip_bbox(lon, lat), "composite_s1.tif")
                if s1res is None:
                    continue
                s1arr, s1t, s1crs = s1res
                s1arr = s1arr[: len(S1_BANDS)]
                s1arr, ox, oy = _crop(s1arr, CHIP_SIZE)
                s1t = s1t * rasterio.Affine.translation(ox, oy)
                _write_tif(s1_path, s1arr, s1t, s1crs, "uint16")
        except Exception as e:  # noqa: BLE001 -- one bad chip must not kill the run
            print(f"chip {cid} failed: {e}")
            continue
        with rasterio.open(mask_path) as m:
            band = m.read(1)
        records.append(dict(chip_id=cid, lon=lon, lat=lat, kind="hard_negative", tile=None,
                             split="train", sub_pixels=int((band == 1).sum()),
                             image=str(img_path), s1=str(s1_path), mask=str(mask_path)))

    index = pd.DataFrame(records)
    index_path = out_dir / "index.parquet"
    index.to_parquet(index_path)
    n_actually_positive = int((index.sub_pixels > 0).sum()) if len(index) else 0
    print(f"Wrote {len(index)} hard-negative chips -> {index_path}")
    if n_actually_positive:
        print(f"  note: {n_actually_positive} accidentally overlap a real substation label "
              "(burned correctly as ignore, harmless but not a true hard negative)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
