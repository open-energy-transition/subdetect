"""Random global hard-negative chips from well-mapped OSM regions (cities, large water
bodies): true "definitely not a substation" examples, sourced from areas where OSM
completeness is high enough that its `power=substation` data is trustworthy ground truth
-- unlike the model's usual training domain (grid corridors in under-mapped regions),
where "no OSM substation here" might just mean "unmapped".

Rationale: merging the TorchGeo Substation dataset (scripts/build_chips_from_substation_ds.py)
pushed data/chips_v9/combined to 94.5% positive chips (every TorchGeo location was curated
around a real substation, with no accompanying hard-negative sampling) -- see memory
subdetect-improvement-levers. Cities give urban/rooftop confusers (the kind of thing that
drove FPs before -- see label_refine.py's building-density guard); water gives a clean,
spectrally distinct negative and a check against reflective-surface false positives.

NOTE dense European/Asian cities have thousands of real (mostly small distribution)
substations -- an early version of this script rejected an entire city if ANY substation
fell within its whole 0.1-deg cell, which threw out every major city outright (Paris:
5339, Amsterdam: 816). Fixed to check clearance per CANDIDATE CHIP CENTER instead (>= 1.6
km from the nearest known substation -- bigger than a chip's half-diagonal, so the
substation itself can't appear in frame), so cities still contribute clean chips from
their parks/rivers/residential fringes while explicitly avoiding their many real
substations.

Pipeline per location:
  1. One Overpass power=substation fetch for the whole cell (+ a small margin) -- reused
     for clearance-filtering every candidate below, not a single go/no-go per location.
  2. Compose one annual S2 median for the containing 0.1-deg cell (full calendar year,
     not a seasonal window -- these span every hemisphere/season).
  3. Sample up to --chips-per-cell candidate 224x224 windows (target center first, then
     random), keeping only those >= CLEARANCE_M from every known substation.
  4. Write image (10-band uint16) + an all-zero mask (pure background) per kept chip.

Usage:
  pixi run -e ml python scripts/build_global_hardneg_chips.py \
      --out data/chips_global_hardneg [--limit-locations N] [--chips-per-cell 6] [--workers 6]
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subdetect.chips import _chip_id  # noqa: E402
from subdetect.config import CHIP_SIZE  # noqa: E402
from subdetect.imagery import annual_composite  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from osmose_detect import fetch_substations  # noqa: E402

log = logging.getLogger(__name__)
CELL_DEG = 0.1
CELL_MARGIN_DEG = 0.02  # extra Overpass fetch margin around the cell
CLEARANCE_M = 1600.0  # > a chip's half-diagonal (2240m side -> 1583m), so a substation
                      # point can never fall inside a chip whose center clears this
YEAR_WINDOW = ("2025-01-01", "2025-12-31")  # global locations span every season
_OVERPASS_LOCK = threading.Semaphore(1)  # Overpass rate-limits hard (429s); serialize

# Well-mapped cities (dense OSM tagging -> substation data here is trustworthy),
# spread across continents for domain diversity in the urban/rooftop confuser class.
CITIES = [
    ("berlin", 52.5200, 13.4050), ("amsterdam", 52.3676, 4.9041), ("paris", 48.8566, 2.3522),
    ("london", 51.5074, -0.1278), ("zurich", 47.3769, 8.5417), ("vienna", 48.2082, 16.3738),
    ("copenhagen", 55.6761, 12.5683), ("stockholm", 59.3293, 18.0686), ("madrid", 40.4168, -3.7038),
    ("rome", 41.9028, 12.4964), ("warsaw", 52.2297, 21.0122), ("prague", 50.0755, 14.4378),
    ("nyc", 40.7128, -74.0060), ("los_angeles", 34.0522, -118.2437), ("chicago", 41.8781, -87.6298),
    ("toronto", 43.6532, -79.3832), ("mexico_city", 19.4326, -99.1332), ("tokyo", 35.6762, 139.6503),
    ("osaka", 34.6937, 135.5023), ("seoul", 37.5665, 126.9780), ("sydney", -33.8688, 151.2093),
    ("melbourne", -37.8136, 144.9631), ("singapore", 1.3521, 103.8198), ("hong_kong", 22.3193, 114.1694),
    ("dubai", 25.2048, 55.2708), ("cape_town", -33.9249, 18.4241), ("nairobi", -1.2921, 36.8219),
    ("sao_paulo", -23.5505, -46.6333), ("buenos_aires", -34.6037, -58.3816), ("santiago", -33.4489, -70.6693),
    ("bogota", 4.7110, -74.0721), ("cairo", 30.0444, 31.2357), ("istanbul", 41.0082, 28.9784),
    ("moscow", 55.7558, 37.6173), ("beijing", 39.9042, 116.4074), ("shanghai", 31.2304, 121.4737),
    ("mumbai", 19.0760, 72.8777), ("bangkok", 13.7563, 100.5018), ("jakarta", -6.2088, 106.8456),
    ("auckland", -36.8485, 174.7633), ("helsinki", 60.1699, 24.9384), ("brussels", 50.8503, 4.3517),
    ("lisbon", 38.7223, -9.1393), ("dublin", 53.3498, -6.2603), ("oslo", 59.9139, 10.7522),
]

# Large, open-water centers (away from shorelines -- avoid mixed land/water pixels).
WATER = [
    ("lake_superior", 47.7, -87.5), ("lake_huron", 44.8, -82.4), ("lake_michigan", 44.0, -87.0),
    ("lake_baikal", 53.5, 108.0), ("caspian_sea", 41.5, 50.5), ("lake_victoria", -1.0, 33.0),
    ("lake_tanganyika", -6.5, 29.5), ("great_bear_lake", 66.0, -120.6), ("great_slave_lake", 61.5, -114.0),
    ("lake_ladoga", 60.85, 31.5), ("lake_onega", 61.7, 35.5), ("vanern", 58.9, 13.3),
    ("lake_titicaca", -15.8, -69.3), ("lake_balkhash", 46.0, 74.5), ("lake_nicaragua", 11.6, -85.4),
    ("lake_winnipeg", 52.0, -97.5), ("lake_erie", 42.2, -81.2), ("lake_ontario", 43.7, -77.9),
    ("aral_sea_n", 45.6, 59.9), ("lake_malawi", -12.0, 34.5),
]


def _cell_bbox(lon: float, lat: float) -> tuple[float, float, float, float]:
    ix, iy = int(np.floor(lon / CELL_DEG)), int(np.floor(lat / CELL_DEG))
    return (ix * CELL_DEG, iy * CELL_DEG, (ix + 1) * CELL_DEG, (iy + 1) * CELL_DEG)


def _fetch_substations_retry(bbox, attempts: int = 4):
    with _OVERPASS_LOCK:
        for i in range(attempts):
            try:
                return fetch_substations(bbox)
            except Exception as e:  # noqa: BLE001 — retry on rate limit / transient errors
                if i == attempts - 1:
                    raise
                wait = 8 * (i + 1)
                log.warning("Overpass error (%s); retrying in %ds", e, wait)
                time.sleep(wait)


def _write_tif(path: Path, arr: np.ndarray, transform, crs, dtype: str) -> None:
    arr = arr if arr.ndim == 3 else arr[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                       count=arr.shape[0], dtype=dtype, crs=crs, transform=transform,
                       compress="deflate", predictor=2) as dst:
        dst.write(arr)


def _process_location(name: str, lat: float, lon: float, out_dir: Path,
                       chips_per_cell: int, rng: np.random.Generator) -> list[dict]:
    cbox = _cell_bbox(lon, lat)
    check_bbox = (cbox[0] - CELL_MARGIN_DEG, cbox[1] - CELL_MARGIN_DEG,
                  cbox[2] + CELL_MARGIN_DEG, cbox[3] + CELL_MARGIN_DEG)
    try:
        subs = _fetch_substations_retry(check_bbox)
    except Exception as e:  # noqa: BLE001 — Overpass hiccup must not kill the whole run
        log.warning("%s: Overpass check failed (%s); skipping (can't confirm negatives)", name, e)
        return []

    res = annual_composite(cbox, date_range=YEAR_WINDOW, max_cloud=30, max_items=8)
    if res is None:
        log.warning("%s: no usable S2 scenes; skipping", name)
        return []
    arr, transform, crs = res
    H, W = arr.shape[1], arr.shape[2]
    if H < CHIP_SIZE or W < CHIP_SIZE:
        log.warning("%s: composite too small (%dx%d); skipping", name, H, W)
        return []

    to_local = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    if len(subs):
        sx, sy = to_local.transform(subs.geometry.x.values, subs.geometry.y.values)
        sub_xy = np.column_stack([sx, sy])
    else:
        sub_xy = np.empty((0, 2))
    log.info("%s: %d OSM substation(s) in/near cell -- clearance-filtering candidates",
             name, len(sub_xy))

    def clear(x: float, y: float) -> bool:
        if len(sub_xy) == 0:
            return True
        return float(np.hypot(sub_xy[:, 0] - x, sub_xy[:, 1] - y).min()) >= CLEARANCE_M

    cx, cy = to_local.transform(lon, lat)
    margin_px = CHIP_SIZE // 2 + 2
    candidates = [(cx, cy)] if clear(cx, cy) else []
    attempts, max_attempts = 0, chips_per_cell * 15
    while len(candidates) < chips_per_cell and attempts < max_attempts:
        attempts += 1
        r = int(rng.integers(margin_px, max(H - margin_px, margin_px + 1)))
        c = int(rng.integers(margin_px, max(W - margin_px, margin_px + 1)))
        x, y = transform * (c, r)
        if clear(x, y):
            candidates.append((x, y))
    if not candidates:
        log.info("%s: no clearance-passing candidates (too dense); skipping", name)
        return []

    inv = ~transform
    records = []
    for x, y in candidates:
        col, row = inv * (x, y)
        r0 = int(np.clip(int(row) - CHIP_SIZE // 2, 0, max(H - CHIP_SIZE, 0)))
        c0 = int(np.clip(int(col) - CHIP_SIZE // 2, 0, max(W - CHIP_SIZE, 0)))
        chip = arr[:, r0:r0 + CHIP_SIZE, c0:c0 + CHIP_SIZE]
        if chip.shape[1] != CHIP_SIZE or chip.shape[2] != CHIP_SIZE:
            continue
        chip_transform = transform * rasterio.Affine.translation(c0, r0)
        wx, wy = chip_transform * (CHIP_SIZE / 2, CHIP_SIZE / 2)
        lonlat = to_wgs84.transform(wx, wy)
        cid = _chip_id(lonlat[0], lonlat[1])
        img_path = out_dir / "images" / f"{cid}.tif"
        mask_path = out_dir / "masks" / f"{cid}.tif"
        if not (img_path.exists() and mask_path.exists()):
            _write_tif(img_path, chip.astype("uint16"), chip_transform, crs, "uint16")
            mask = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype="int16")
            _write_tif(mask_path, mask, chip_transform, crs, "int16")
        # Train-only: global cities/water have no relation to the Pakistan/Sindh
        # deployment domain -- val should come only from the geographic holdout
        # (val_bbox in configs/aoi.yaml), not a hash split of unrelated locations.
        split = "train"
        records.append(dict(chip_id=cid, lon=lonlat[0], lat=lonlat[1], kind="hard_negative",
                            tile=name, split=split, sub_pixels=0,
                            image=str(img_path), s1=None, mask=str(mask_path)))
    log.info("%s: %d/%d candidates written", name, len(records), len(candidates))
    return records


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/chips_global_hardneg")
    ap.add_argument("--chips-per-cell", type=int, default=6)
    ap.add_argument("--limit-locations", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    locations = CITIES + WATER
    if a.limit_locations:
        locations = locations[: a.limit_locations]

    all_records = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_process_location, name, lat, lon, out_dir, a.chips_per_cell,
                          np.random.default_rng(a.seed + i)): name
                for i, (name, lat, lon) in enumerate(locations)}
        for fut in as_completed(futs):
            all_records.extend(fut.result())

    index = pd.DataFrame(all_records)
    index_path = out_dir / "index.parquet"
    index.to_parquet(index_path)
    log.info("Wrote %d hard-negative chips from %d/%d locations (%d val) -> %s",
             len(index), index.tile.nunique() if len(index) else 0, len(locations),
             int((index.split == "val").sum()) if len(index) else 0, index_path)


if __name__ == "__main__":
    main()
