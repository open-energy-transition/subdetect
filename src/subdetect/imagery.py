"""Sentinel-2 L2A + Sentinel-1 RTC median composites from Microsoft Planetary Computer.

`annual_composite` (S2) is earthpv's, unchanged: a cloud-masked median of the 10 local
bands at 10 m. `s1_composite` (new) builds a co-registered VV/VH dry-season median in dB,
encoded to uint16, on the S2 cell's exact grid so the two modalities stack pixel-perfectly.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import odc.stac
import planetary_computer
import pystac_client

from subdetect.config import S1_OFFSET_DB, S1_SCALE

log = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# SCL classes to keep: 4 vegetation, 5 bare, 6 water, 7 unclassified.
_SCL_VALID = (4, 5, 6, 7)
MAX_CLOUD = 60

_SEARCH_LOCK = __import__("threading").Lock()


@lru_cache(maxsize=1)
def _catalog() -> pystac_client.Client:
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def annual_composite(
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str] = ("2025-11-01", "2026-03-15"),
    max_cloud: int = 30,
    max_items: int = 12,
    geobox=None,
) -> tuple[np.ndarray, object, object] | None:
    """Cloud-masked median over the 10 local S2 bands (B02..B12), 10 m.

    Uses the ~`max_items` least-cloudy scenes in `date_range` to bound the download
    while keeping a clean median. Returns (array[10,H,W] uint16, transform, crs) in the
    bbox's UTM zone, or None if no usable scenes. `geobox` pins the output to an exact
    existing grid; the STAC search still uses `bbox`. The catalog search is serialized
    (pystac-client is not thread-safe); the COG reads run concurrently fine.
    """
    from rasterio.crs import CRS

    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    with _SEARCH_LOCK:
        search = _catalog().search(
            collections=["sentinel-2-l2a"], bbox=bbox,
            datetime=f"{date_range[0]}/{date_range[1]}",
            query={"eo:cloud_cover": {"lt": max_cloud}},
        )
        items = sorted(search.items(), key=lambda it: it.properties.get("eo:cloud_cover", 100))
    if not items:
        return None
    items = items[:max_items]
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) / 6) + 1
    grid = dict(geobox=geobox) if geobox is not None else dict(
        bbox=bbox, resolution=10, crs=CRS.from_epsg(epsg)
    )
    ds = odc.stac.load(
        items, bands=[*bands, "SCL"], groupby="solar_day",
        chunks={"x": 2048, "y": 2048}, fail_on_error=False, **grid,
    )
    valid = ds["SCL"].isin(_SCL_VALID)
    masked = ds[bands].where(valid)
    med = masked.median(dim="time", skipna=True).fillna(0).astype("uint16").compute()
    arr = np.stack([med[b].values for b in bands], axis=0)
    transform = med.odc.transform if hasattr(med, "odc") else med.rio.transform()
    crs = geobox.crs if geobox is not None else CRS.from_epsg(epsg)
    return arr, transform, crs


def s1_composite(
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str] = ("2025-11-01", "2026-03-15"),
    max_items: int = 16,
    geobox=None,
) -> tuple[np.ndarray, object, object] | None:
    """Sentinel-1 RTC (gamma0, terrain-corrected) VV/VH dry-season median, encoded uint16.

    Median is taken in linear power then converted to dB, so it is robust to speckle;
    `DN = clip((dB + S1_OFFSET_DB) * S1_SCALE, 1, 65535)`, 0 reserved for nodata. Pass
    `geobox` = the cell's S2 GeoBox so VV/VH land on the identical grid as composite_0.
    Returns (array[2,H,W] uint16, transform, crs) or None if no scenes.
    """
    from rasterio.crs import CRS

    assets = ["vv", "vh"]
    with _SEARCH_LOCK:
        search = _catalog().search(
            collections=["sentinel-1-rtc"], bbox=bbox,
            datetime=f"{date_range[0]}/{date_range[1]}",
        )
        items = list(search.items())
    if not items:
        return None
    items = items[:max_items]
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) / 6) + 1
    grid = dict(geobox=geobox) if geobox is not None else dict(
        bbox=bbox, resolution=10, crs=CRS.from_epsg(epsg)
    )
    ds = odc.stac.load(
        items, bands=assets, groupby="solar_day",
        chunks={"x": 2048, "y": 2048}, fail_on_error=False, **grid,
    )
    # RTC assets are linear power (gamma0); mask non-positive/no-data, median, -> dB.
    lin = ds[assets].where(lambda d: d > 0)
    med = lin.median(dim="time", skipna=True).compute()
    out = []
    for a in assets:
        vals = med[a].values.astype("float32")
        db = 10.0 * np.log10(np.where(vals > 0, vals, np.nan))
        dn = np.clip((db + S1_OFFSET_DB) * S1_SCALE, 1, 65535)
        dn[~np.isfinite(db)] = 0  # nodata
        out.append(dn.astype("uint16"))
    arr = np.stack(out, axis=0)
    transform = med.odc.transform if hasattr(med, "odc") else med.rio.transform()
    crs = geobox.crs if geobox is not None else CRS.from_epsg(epsg)
    return arr, transform, crs
