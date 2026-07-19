"""Convert the TransitionZero/TorchGeo `Substation` dataset (27k global S2 chips + binary
masks; https://github.com/Lindsay-Lab/substation-seg, torchgeo.datasets.Substation) into
subdetect's chip index format, to grow the data-starved S2-only arm (val IoU_1 0.111 at
v5 -- see memory subdetect-improvement-levers).

Source layout expected under --src:
  image_stack/lat_<LAT>_lon_<LON>.npz   arr_0: (revisits, 13, 228, 228) float64
  mask.tar.gz                           mask/lat_<LAT>_lon_<LON>.npz -> arr_0: (228, 228) uint8

Band order is standard Sentinel-2 (confirmed against torchgeo's source: rgb_bands=(3,2,1)
i.e. B04,B03,B02 at those indices -> B01..B12 at indices 0..12). We select the same 10
bands/order as subdetect.config.LOCAL_BANDS (indices [1,2,3,4,5,6,7,8,11,12]), so the
output GeoTIFFs are byte-compatible with the existing S2-only training pipeline.

Mask class 3 = substation (confirmed against torchgeo's Substation.__getitem__:
`mask[mask != 3] = 0; mask[mask == 3] = 1`); this repo's README does not document it.

Revisits are median-composited into one image per location (matching subdetect's single
dry-season-composite convention). The paper's better-performing approach -- fusing
revisits in the model's latent space (arXiv:2409.17363) -- is an architecture change,
not a data-conversion one; tracked separately, not done here.

No Sentinel-1 companion exists for this source, so `s1` is always None -- these chips
only feed S2-only (or S2 arm of fusion) training.

Usage:
  pixi run -e ml python scripts/build_chips_from_substation_ds.py \
      --src /run/media/tobi/b20cbdca-5a92-4cc0-a21d-48d30839b238 \
      --out data/chips_torchgeo/substation_global [--limit 200] [--workers 8]
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import tarfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio import Affine
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subdetect.chips import _chip_id  # noqa: E402
from subdetect.config import CHIP_SIZE  # noqa: E402

log = logging.getLogger(__name__)

# Standard Sentinel-2 13-band order (matches torchgeo.datasets.Substation.rgb_bands=(3,2,1)):
# B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12
S13_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
LOCAL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
BAND_IDX = [S13_ORDER.index(b) for b in LOCAL_BANDS]  # [1,2,3,4,5,6,7,8,11,12]

NAME_RE = re.compile(r"lat_(-?[\d.]+)_lon_(-?[\d.]+)\.npz$")


def _utm_epsg(lon: float, lat: float) -> int:
    return (32600 if lat >= 0 else 32700) + int((lon + 180) / 6) + 1


def _center_crop_or_pad(arr: np.ndarray, size: int) -> np.ndarray:
    *lead, y, x = arr.shape
    oy, ox = max((y - size) // 2, 0), max((x - size) // 2, 0)
    arr = arr[..., oy: oy + size, ox: ox + size]
    if arr.shape[-2] < size or arr.shape[-1] < size:
        pad = [(0, 0)] * (arr.ndim - 2) + [(0, size - arr.shape[-2]), (0, size - arr.shape[-1])]
        arr = np.pad(arr, pad)
    return arr


def _write_tif(path: Path, arr: np.ndarray, transform, crs, dtype: str) -> None:
    arr = arr if arr.ndim == 3 else arr[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                       count=arr.shape[0], dtype=dtype, crs=crs, transform=transform,
                       compress="deflate", predictor=2) as dst:
        dst.write(arr)


def _convert_one(args: tuple) -> dict | None:
    img_path, mask_path, out_dir = args
    m = NAME_RE.search(img_path.name)
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    cid = _chip_id(lon, lat)
    out_img = out_dir / "images" / f"{cid}.tif"
    out_mask = out_dir / "masks" / f"{cid}.tif"
    try:
        if not (out_img.exists() and out_mask.exists()):
            image = np.load(img_path)["arr_0"]  # (T, 13, H, W)
            image = image[:, BAND_IDX, :, :]
            image = np.median(image, axis=0)  # (10, H, W)
            image = _center_crop_or_pad(image, CHIP_SIZE)
            image = np.clip(image, 0, 65535).astype("uint16")

            mask = np.load(mask_path)["arr_0"]  # (H, W) uint8, class 3 = substation
            mask = np.where(mask == 3, 1, 0).astype("int16")
            mask = _center_crop_or_pad(mask, CHIP_SIZE)

            epsg = _utm_epsg(lon, lat)
            from pyproj import Transformer

            tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
            cx, cy = tr.transform(lon, lat)
            half = CHIP_SIZE * 10.0 / 2
            transform = Affine(10.0, 0.0, cx - half, 0.0, -10.0, cy + half)
            crs = f"EPSG:{epsg}"

            _write_tif(out_img, image, transform, crs, "uint16")
            _write_tif(out_mask, mask, transform, crs, "int16")
        else:
            with rasterio.open(out_mask) as src:
                mask = src.read(1)
    except Exception as e:  # noqa: BLE001 — one bad chip must not kill the run
        log.warning("chip %s failed: %s", img_path.name, e)
        return None

    sub_pixels = int((mask == 1).sum())
    split = "val" if int(cid[-2:], 16) < 26 else "train"  # deterministic ~10% val
    return dict(chip_id=cid, lon=lon, lat=lat, kind="positive" if sub_pixels else "background",
                tile="torchgeo_substation", split=split, sub_pixels=sub_pixels,
                image=str(out_img), s1=None, mask=str(out_mask))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Dir containing image_stack/ and mask.tar.gz")
    ap.add_argument("--out", required=True, help="Output dir (index.parquet + images/masks)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    src = Path(a.src)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = out_dir / "_mask_src"
    if not mask_dir.exists():
        log.info("extracting mask.tar.gz -> %s", mask_dir)
        with tarfile.open(src / "mask.tar.gz") as tf:
            tf.extractall(mask_dir)  # noqa: S202 — trusted local archive

    images = sorted((src / "image_stack").glob("*.npz"))
    if a.limit:
        images = images[: a.limit]
    tasks = []
    for img_path in images:
        mask_path = mask_dir / "mask" / img_path.name
        if mask_path.exists():
            tasks.append((img_path, mask_path, out_dir))
    log.info("%d/%d image files have a matching mask; converting", len(tasks), len(images))

    records = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_convert_one, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc="convert"):
            r = f.result()
            if r:
                records.append(r)

    index = pd.DataFrame(records)
    index_path = out_dir / "index.parquet"
    index.to_parquet(index_path)
    log.info("Wrote %d chips (%d with substation, %d val) -> %s", len(index),
             int((index.sub_pixels > 0).sum()) if len(index) else 0,
             int((index.split == "val").sum()) if len(index) else 0, index_path)


if __name__ == "__main__":
    main()
