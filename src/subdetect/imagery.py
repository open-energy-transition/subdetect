"""Sentinel-2 L2A + Sentinel-1 RTC median composites from Microsoft Planetary Computer.

`annual_composite` (S2) is earthpv's, unchanged: a cloud-masked median of the 10 local
bands at 10 m, falling back to Element84's Earth Search catalog (AWS Open Data, same L2A
scenes as COGs) when Planetary Computer errors out — PC is faster when healthy but has
multi-hour 503 storms and SAS-token expiries under sustained load; Earth Search needs no
auth/tokens and lives in a different failure domain. `s1_composite` (new) builds a
co-registered VV/VH dry-season median in dB, encoded to uint16, on the S2 cell's exact
grid so the two modalities stack pixel-perfectly.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
import odc.stac
import planetary_computer
import pystac_client
import xarray as xr

from subdetect.config import S1_OFFSET_DB, S1_SCALE

log = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
ES_STAC_URL = "https://earth-search.aws.element84.com/v1"
# SCL classes to keep: 4 vegetation, 5 bare, 6 water, 7 unclassified.
_SCL_VALID = (4, 5, 6, 7)
MAX_CLOUD = 60

# A healthy PC read logs ~0 of these; a 503-storm or expired-SAS-token cell
# logs dozens within seconds (odc/GDAL retry the same bad URL repeatedly
# without re-signing, so the read "succeeds" via fail_on_error=False but with
# degraded/slow coverage). Substring match, not logger name, since the
# messages come from GDAL's C error stack via multiple rasterio/odc loggers.
_GDAL_FAILURE_MARKERS = (
    "GDAL signalled an error", "CPLE_AppDefined", "Ignoring read failure",
    "TIFFReadEncodedTile", "response_code=4", "response_code=5",
)
MAX_GDAL_WARNINGS = 20


class _GdalFailureCounter(logging.Handler):
    """Counts GDAL read-failure log records from this thread only.

    Attached to the root logger since the messages originate from several
    rasterio/GDAL loggers (name varies by version); thread filtering is what
    keeps counts scoped to this call when other cells compose concurrently.
    """

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.count = 0
        self._thread_id = threading.get_ident()

    def emit(self, record):
        if record.thread != self._thread_id:
            return
        msg = record.getMessage()
        if any(marker in msg for marker in _GDAL_FAILURE_MARKERS):
            self.count += 1


@contextmanager
def _count_gdal_failures():
    counter = _GdalFailureCounter()
    root = logging.getLogger()
    root.addHandler(counter)
    try:
        yield counter
    finally:
        root.removeHandler(counter)


# Circuit breaker: once PC racks up PC_TRIP_THRESHOLD consecutive failures
# (network errors, empty-turned-exception, or the GDAL-warning trip above),
# skip it entirely for PC_COOLDOWN_S and go straight to Earth Search — a
# systemic PC outage means every subsequent cell would otherwise pay the same
# slow doomed-retry cost before falling back. A success (from any thread)
# resets the streak; the cooldown then just lapses and PC gets tried fresh.
PC_TRIP_THRESHOLD = 3
PC_COOLDOWN_S = 600
_PC_LOCK = threading.Lock()
_pc_breaker = {"fails": 0, "open_until": 0.0}


def _pc_should_skip() -> bool:
    with _PC_LOCK:
        return time.monotonic() < _pc_breaker["open_until"]


def _pc_record(ok: bool) -> None:
    with _PC_LOCK:
        if ok:
            _pc_breaker["fails"] = 0
            return
        _pc_breaker["fails"] += 1
        if _pc_breaker["fails"] >= PC_TRIP_THRESHOLD:
            _pc_breaker["open_until"] = time.monotonic() + PC_COOLDOWN_S
            _pc_breaker["fails"] = 0
            log.warning(
                "Planetary Computer circuit breaker OPEN for %ds after %d consecutive failures",
                PC_COOLDOWN_S, PC_TRIP_THRESHOLD,
            )


# Earth Search serves the same COG assets keyed by common band name; the rest of
# the pipeline (and the trained model's radiometry) speaks PC's B-names.
_ES_BAND_FOR = {
    "B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1",
    "B06": "rededge2", "B07": "rededge3", "B08": "nir", "B8A": "nir08",
    "B11": "swir16", "B12": "swir22", "SCL": "scl",
}

_SEARCH_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _catalog() -> pystac_client.Client:
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


@lru_cache(maxsize=1)
def _es_catalog() -> pystac_client.Client:
    return pystac_client.Client.open(ES_STAC_URL)


def _annual_composite_via(
    provider: str,
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str],
    max_cloud: int,
    max_items: int,
    geobox=None,
) -> tuple[np.ndarray, object, object] | None:
    """One provider attempt of annual_composite; `provider` is
    "planetary-computer" or "earth-search"."""
    from rasterio.crs import CRS

    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    es = provider == "earth-search"
    catalog = _es_catalog() if es else _catalog()
    with _SEARCH_LOCK:
        search = catalog.search(
            collections=["sentinel-2-l2a"], bbox=bbox,
            datetime=f"{date_range[0]}/{date_range[1]}",
            query={"eo:cloud_cover": {"lt": max_cloud}},
        )
        items = sorted(search.items(), key=lambda it: it.properties.get("eo:cloud_cover", 100))
    if not items:
        return None
    items = items[:max_items]
    load_bands = [_ES_BAND_FOR[b] for b in [*bands, "SCL"]] if es else [*bands, "SCL"]
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) / 6) + 1
    grid = dict(geobox=geobox) if geobox is not None else dict(
        bbox=bbox, resolution=10, crs=CRS.from_epsg(epsg)
    )
    with _count_gdal_failures() as failures:
        ds = odc.stac.load(
            items, bands=load_bands, groupby="solar_day",
            chunks={"x": 2048, "y": 2048}, fail_on_error=False, **grid,
        )
        if es:
            ds = ds.rename({_ES_BAND_FOR[b]: b for b in [*bands, "SCL"]})
        valid = ds["SCL"].isin(_SCL_VALID)
        masked = ds[bands].where(valid)
        if es:
            # Earth Search bakes the baseline->=04.00 BOA offset into its COGs
            # (earthsearch:boa_offset_applied); PC serves raw DNs, which is what the
            # model is calibrated to. Add the offset back per solar day so fallback
            # cells are radiometrically identical to PC ones. Pre-2022 baselines
            # carry no offset.
            per_day = {
                it.datetime.date(): 1000 if it.properties.get("earthsearch:boa_offset_applied") else 0
                for it in items
            }
            offs = [per_day.get(d, 1000) for d in ds["time"].dt.date.values]
            masked = masked + xr.DataArray(offs, coords={"time": ds["time"]}, dims="time")
        med = masked.median(dim="time", skipna=True).fillna(0).astype("uint16").compute()
    if not es and failures.count > MAX_GDAL_WARNINGS:
        # Reads "succeeded" (fail_on_error=False swallows per-tile errors into
        # NaN) but this many GDAL failures means a chunk of scenes/bands didn't
        # actually land — a quietly degraded composite, not just a slow one.
        # Only checked for PC: Earth Search has no fallback beyond it.
        raise RuntimeError(
            f"{provider}: {failures.count} GDAL read failures for bbox={bbox} "
            "(likely 503 storm / SAS token expiry); treating as failed"
        )
    arr = np.stack([med[b].values for b in bands], axis=0)
    if (arr != 0).mean() < 0.01:
        # fail_on_error=False degrades read failures to NaN -> 0, so a total 503
        # storm can yield a "successful" all-zero composite that resume-skipping
        # then never repairs. Cells are land around power infrastructure:
        # (near-)all-zero means the reads failed wholesale, not dark ground.
        raise RuntimeError(
            f"{provider}: composite {100 * (arr == 0).mean():.0f}% empty for bbox={bbox}"
        )
    transform = med.odc.transform if hasattr(med, "odc") else med.rio.transform()
    crs = geobox.crs if geobox is not None else CRS.from_epsg(epsg)
    return arr, transform, crs


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

    Tries Planetary Computer first (fastest when healthy), and retries the whole
    cell via Earth Search (AWS) if PC errors out or comes back empty — different
    hosting, so PC's recurring 503 storms don't take the cell down. After
    PC_TRIP_THRESHOLD consecutive PC failures, a circuit breaker skips PC
    entirely (straight to Earth Search) for PC_COOLDOWN_S, so a systemic outage
    doesn't make every remaining cell pay for a doomed PC attempt first.
    """
    if _pc_should_skip():
        log.info(
            "Planetary Computer circuit breaker open; using Earth Search directly for bbox=%s",
            bbox,
        )
        return _annual_composite_via(
            "earth-search", bbox, date_range, max_cloud, max_items, geobox
        )
    try:
        result = _annual_composite_via(
            "planetary-computer", bbox, date_range, max_cloud, max_items, geobox
        )
    except Exception as e:  # noqa: BLE001 — any PC failure is grounds for fallback
        _pc_record(False)
        log.warning(
            "Planetary Computer failed for bbox=%s (%s); falling back to Earth Search",
            bbox, e,
        )
        return _annual_composite_via(
            "earth-search", bbox, date_range, max_cloud, max_items, geobox
        )
    if result is None:
        # PC has no scenes for this window; ES mirrors the same ESA archive but
        # ingestion lags differ — cheap second opinion before declaring no-data.
        # Not a health signal either way, so it doesn't touch the breaker.
        log.info("Planetary Computer returned no scenes for bbox=%s; trying Earth Search", bbox)
        return _annual_composite_via(
            "earth-search", bbox, date_range, max_cloud, max_items, geobox
        )
    _pc_record(True)
    return result


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
