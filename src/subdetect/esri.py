"""Esri World Imagery access: cached XYZ tiles, stitched and reprojected onto a chip's grid.

Used by the ESRI core-label study (docs/issues/esri-core-labels.md) to look at a substation
at ~0.5 m and decide which part of an OSM `power=substation` polygon is the equipment yard
and which is the surrounding property. Nothing here is substation-specific.

Tiles are cached on disk exactly as received (JPEG bytes, `<cache>/<z>/<x>/<y>.jpg`), so a
re-run, or a second chip overlapping the same substation, costs no network. A missing tile
(HTTP 404/no data) is cached as a zero-byte file and read back as "no imagery" (alpha 0),
never as black pixels.

Imagery terms: Esri World Imagery is used here for research label refinement; respect
Esri's terms of use and attribute "Esri, Vantor, Earthstar Geographics" in anything
published from it.
"""
from __future__ import annotations

import io
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
import requests
from PIL import Image
from rasterio import Affine
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/"
            "tile/{z}/{y}/{x}")
IDENTIFY_URL = ("https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/"
                "MapServer/identify")
TILE = 256
HALF_WORLD = 20037508.342789244  # EPSG:3857 half-extent, metres


def tile_res(z: int) -> float:
    """EPSG:3857 metres per pixel at zoom z (true ground res is this x cos(lat))."""
    return 2 * HALF_WORLD / (TILE * 2 ** z)


def is_placeholder(rgb: np.ndarray) -> bool:
    """Esri's no-data tile: >=90% of pixels grey (spread <8) at brightness 195-215."""
    a = rgb.astype(np.int16)
    sat = a.max(-1) - a.min(-1)
    br = a.mean(-1)
    return bool(((sat < 8) & (br > 195) & (br < 215)).mean() > 0.9)


def zoom_for(ground_res_m: float, lat: float) -> int:
    """Coarsest zoom whose true ground resolution at `lat` is within 25% of the request.

    A tight tolerance is expensive: at 5% every site south of ~28 N asked for 0.5 m was
    sent to z19 (4x the tiles, and usually Esri's no-data placeholder there)."""
    c = math.cos(math.radians(lat))
    for z in range(12, 20):
        if tile_res(z) * c <= ground_res_m * 1.25:
            return z
    return 19


class EsriTiles:
    def __init__(self, cache_dir: Path, workers: int = 8, timeout: float = 30.0):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.timeout = timeout
        self._local = threading.local()

    def _session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers["User-Agent"] = "subdetect-research/0.2 (label refinement study)"
            retry = Retry(total=6, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
                          allowed_methods=("GET",))
            s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=self.workers * 2))
            self._local.s = s
        return s

    def _tile_bytes(self, z: int, x: int, y: int) -> bytes:
        p = self.cache / str(z) / str(x) / f"{y}.jpg"
        if p.exists():
            return p.read_bytes()
        r = self._session().get(TILE_URL.format(z=z, x=x, y=y), timeout=self.timeout)
        if r.status_code == 404:
            data = b""
        else:
            r.raise_for_status()
            data = r.content
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.stem}.{threading.get_ident()}.part")  # threads may race on a shared tile
        tmp.write_bytes(data)
        tmp.replace(p)
        return data

    def tile(self, z: int, x: int, y: int) -> np.ndarray:
        """(256, 256, 4) uint8 RGBA; alpha 0 where the server has no imagery.

        Beyond its native zoom Esri answers 200 with a "Map data not yet available" tile
        (flat grey ~205 with faint text) -- 10% of a 102-site pilot at z18. Those are
        returned as alpha 0 too, so callers can fall back to a coarser zoom.
        """
        data = self._tile_bytes(z, x, y)
        if not data:
            return np.zeros((TILE, TILE, 4), np.uint8)
        im = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        if is_placeholder(im):
            return np.zeros((TILE, TILE, 4), np.uint8)
        return np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])

    def mosaic_3857(self, bounds: tuple[float, float, float, float], z: int,
                    pool: ThreadPoolExecutor | None = None) -> tuple[np.ndarray, Affine]:
        """Stitch every tile intersecting EPSG:3857 `bounds` -> ((H, W, 4) RGBA, transform)."""
        minx, miny, maxx, maxy = bounds
        res = tile_res(z)
        n = 2 ** z
        x0 = max(int((minx + HALF_WORLD) // (TILE * res)), 0)
        x1 = min(int((maxx + HALF_WORLD) // (TILE * res)), n - 1)
        y0 = max(int((HALF_WORLD - maxy) // (TILE * res)), 0)
        y1 = min(int((HALF_WORLD - miny) // (TILE * res)), n - 1)
        keys = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
        if pool is not None:
            tiles = list(pool.map(lambda k: self.tile(z, *k), keys))
        else:
            tiles = [self.tile(z, *k) for k in keys]
        nx, ny = x1 - x0 + 1, y1 - y0 + 1
        out = np.zeros((ny * TILE, nx * TILE, 4), np.uint8)
        for (x, y), t in zip(keys, tiles):
            out[(y - y0) * TILE:(y - y0 + 1) * TILE, (x - x0) * TILE:(x - x0 + 1) * TILE] = t
        transform = Affine(res, 0, -HALF_WORLD + x0 * TILE * res,
                           0, -res, HALF_WORLD - y0 * TILE * res)
        return out, transform

    def read_grid(self, dst_transform: Affine, dst_crs, shape: tuple[int, int], z: int,
                  pool: ThreadPoolExecutor | None = None) -> np.ndarray:
        """Esri imagery resampled onto an arbitrary grid -> (H, W, 4) uint8 RGBA."""
        h, w = shape
        minx, maxy = dst_transform * (0, 0)
        maxx, miny = dst_transform * (w, h)
        b = rasterio.warp.transform_bounds(dst_crs, "EPSG:3857",
                                           min(minx, maxx), min(miny, maxy),
                                           max(minx, maxx), max(miny, maxy), densify_pts=21)
        pad = 4 * tile_res(z)
        src, src_tr = self.mosaic_3857((b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad), z, pool)
        out = np.zeros((4, h, w), np.uint8)
        rasterio.warp.reproject(
            np.moveaxis(src, -1, 0), out, src_transform=src_tr, src_crs="EPSG:3857",
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=rasterio.warp.Resampling.bilinear,
        )
        return np.moveaxis(out, 0, -1)

    def metadata(self, lon: float, lat: float) -> dict:
        """Acquisition date / resolution / stated accuracy of the top imagery layer."""
        d = 0.002
        try:
            r = self._session().get(IDENTIFY_URL, timeout=self.timeout, params=dict(
                geometry=f"{lon},{lat}", geometryType="esriGeometryPoint", sr=4326,
                layers="all", tolerance=1, mapExtent=f"{lon - d},{lat - d},{lon + d},{lat + d}",
                imageDisplay="400,400,96", returnGeometry="false", f="json"))
            res = r.json().get("results", [])
        except Exception as e:  # noqa: BLE001 -- metadata is advisory, never fatal
            log.debug("identify failed at %.5f,%.5f: %s", lon, lat, e)
            return {}
        for item in res:
            a = item.get("attributes", {})
            if item.get("layerId") == 0 or "DATE (YYYYMMDD)" in a:
                try:
                    res_m = float(a.get("RESOLUTION (M)"))
                except (TypeError, ValueError):
                    res_m = float("nan")
                try:
                    acc_m = float(a.get("ACCURACY (M)"))
                except (TypeError, ValueError):
                    acc_m = float("nan")
                return dict(img_date=str(a.get("DATE (YYYYMMDD)", "")), img_res_m=res_m,
                            img_acc_m=acc_m, img_source=str(a.get("SOURCE_INFO", "")))
        return {}
