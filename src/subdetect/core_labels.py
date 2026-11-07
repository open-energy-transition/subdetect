"""Shared chip-geometry helpers for the ESRI core-label study.

The one non-obvious fact this module exists to encode: TorchGeo chips
(`aoi == "torchgeo_substation"`, written by scripts/build_chips_from_substation_ds.py) are
stored SOUTH-UP under a north-up geotransform -- image and mask rows are vertically
mirrored relative to the transform. Training never noticed (image and mask flip together
and the datamodule augments with flips), but anything that pairs a chip with imagery fetched
by its transform does. Every reader here returns NORTH-UP arrays, and `write_label`
flips back so the corpus stays internally consistent with its own images.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

FLIPPED_AOIS = {"torchgeo_substation"}


def is_flipped(aoi: str) -> bool:
    return aoi in FLIPPED_AOIS


def read_north_up(path: str | Path, flipped: bool, r0: int = 0, r1: int | None = None,
                  c0: int = 0, c1: int | None = None, bands=None) -> np.ndarray:
    """Read a window given in NORTH-UP pixel coordinates of the chip grid."""
    with rasterio.open(path) as src:
        H, W = src.height, src.width
        r1 = H if r1 is None else r1
        c1 = W if c1 is None else c1
        sr0, sr1 = (H - r1, H - r0) if flipped else (r0, r1)
        arr = src.read(bands, window=Window(c0, sr0, c1 - c0, sr1 - sr0))
    return arr[..., ::-1, :] if flipped else arr


def write_label(src_mask: str | Path, out_path: str | Path, label_north_up: np.ndarray,
                flipped: bool) -> None:
    """Write an int16 label in the SOURCE chip's orientation, transform and CRS."""
    from subdetect.chips import _write_tif

    with rasterio.open(src_mask) as src:
        transform, crs = src.transform, src.crs
    arr = label_north_up[::-1] if flipped else label_north_up
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tif(out_path, np.ascontiguousarray(arr).astype("int16"), transform, crs, "int16")
