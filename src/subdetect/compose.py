"""Build Sentinel-2 (and Sentinel-1) composites for an AOI's ROI cells via STAC.

Cells come from roi.select_cells (full power-corridor ROI for Pakistan, capped
substation cells for the India pilot), ordered so label cells composite first — the
training set becomes usable without waiting for the whole ROI. Output COGs mirror
earthpv's layout (`<cell>/composite_0.tif`, `<cell>/composite_s1.tif`) so CompositeIndex
and infer read them unchanged. Resumable (existing layers skipped), thread-parallel.
"""

from __future__ import annotations

import logging
from pathlib import Path

import rasterio
from tqdm import tqdm

from subdetect.config import LOCAL_BANDS, S1_BANDS, Settings, resolve_aoi
from subdetect.imagery import annual_composite, s1_composite
from subdetect.roi import CELL_DEG, select_cells


def run_compose(
    aoi: str,
    out_dir: Path,
    sensor: str = "s2",
    radius_km: float = 20.0,
    min_voltage: float = 0.0,
    limit: int = 0,
    workers: int = 1,
) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    settings = Settings.load()
    _, cfg = resolve_aoi(aoi, settings)
    region_dir = Path(out_dir) / aoi
    out_dir = region_dir / "composites"
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = select_cells(aoi, cfg, settings, radius_km, min_voltage)
    if limit:
        cells = cells.head(limit)
    s2_window = tuple(settings.raw.get("s2_window", ("2025-11-01", "2026-03-15")))
    s1_window = tuple(settings.raw.get("s1_window", ("2025-11-01", "2026-03-15")))
    layer = "composite_0.tif" if sensor == "s2" else "composite_s1.tif"
    bands = LOCAL_BANDS if sensor == "s2" else S1_BANDS
    log.info("Compositing %s for %d cells of %s (%d workers) -> %s",
             sensor.upper(), len(cells), aoi, workers, layer)

    def _one(cell) -> bool:
        cell_dir = out_dir / cell["name"]
        tif = cell_dir / layer
        if tif.exists():
            return False
        bbox = (cell.lon0, cell.lat0, cell.lon0 + CELL_DEG, cell.lat0 + CELL_DEG)
        try:
            if sensor == "s2":
                res = annual_composite(bbox, date_range=s2_window)
            else:
                # S1 pins to the cell's S2 grid so VV/VH align with composite_0 exactly.
                base = cell_dir / "composite_0.tif"
                if not base.exists():
                    return False  # S1 only where S2 exists (needs the geobox)
                from odc.geo.geobox import GeoBox

                with rasterio.open(base) as b:
                    gbox = GeoBox((b.height, b.width), b.transform, b.crs)
                res = s1_composite(bbox, date_range=s1_window, geobox=gbox)
        except Exception as e:  # noqa: BLE001 — one bad cell must not kill the run
            logging.getLogger(__name__).warning("cell %s failed: %s", cell["name"], e)
            return False
        if res is None:
            return False
        arr, transform, crs = res
        cell_dir.mkdir(parents=True, exist_ok=True)
        tmp = tif.with_suffix(".tif.tmp")
        with rasterio.open(
            tmp, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
            dtype="uint16", crs=crs, transform=transform, compress="deflate", predictor=2,
        ) as dst:
            dst.write(arr)
            dst.descriptions = tuple(bands)
        tmp.rename(tif)
        return True

    rows = [cell for _, cell in cells.iterrows()]
    done = 0
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for ok in tqdm(ex.map(_one, rows), total=len(rows), desc=f"compose-{sensor}"):
                done += int(ok)
    else:
        for cell in tqdm(rows, desc=f"compose-{sensor}"):
            done += int(_one(cell))
    log.info("Composited %d new %s cells -> %s", done, sensor.upper(), out_dir)
    return region_dir
