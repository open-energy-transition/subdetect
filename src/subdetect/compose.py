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

from subdetect.config import (
    DEM_BANDS, LOCAL_BANDS, S1_BANDS, TOWER_PRIOR_BANDS, Settings, resolve_aoi,
)
from subdetect.imagery import annual_composite, s1_composite
from subdetect.roi import CELL_DEG, _snapped_origin, select_cells


def run_compose(
    aoi: str,
    out_dir: Path,
    sensor: str = "s2",
    radius_km: float = 20.0,
    min_voltage: float = 0.0,
    limit: int = 0,
    workers: int = 1,
    bbox_filter: tuple[float, float, float, float] | None = None,
    cells_file: Path | None = None,
) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    settings = Settings.load()
    _, cfg = resolve_aoi(aoi, settings)
    region_dir = Path(out_dir) / aoi
    out_dir = region_dir / "composites"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cells = select_cells(aoi, cfg, settings, radius_km, min_voltage)
    except FileNotFoundError:
        # A feasibility AOI (e.g. the BESS study's gb_bess/ca_bess/tx_bess) carries no
        # substation or line labels of its own, so the ROI rule -- "within radius_km of a
        # mapped power line or substation" -- cannot be evaluated. That is fine precisely
        # when an explicit cell list was supplied, because the cells_file branch below
        # synthesises every cell from the AOI lattice anyway. Without a cell list this
        # stays a hard error: it means `osm --aoi` was never run.
        if cells_file is None:
            raise
        import pandas as pd

        cells = pd.DataFrame({
            "ix": pd.Series(dtype="int64"), "iy": pd.Series(dtype="int64"),
            "name": pd.Series(dtype="object"), "has_label": pd.Series(dtype="bool"),
            "line_km": pd.Series(dtype="float64"),
            "lon0": pd.Series(dtype="float64"), "lat0": pd.Series(dtype="float64"),
        })
    if cells_file:
        # Explicit cell list, bypassing ROI selection entirely. The ROI rule is "within
        # radius_km of a power line or substation", which structurally excludes targets
        # that are not grid-adjacent -- remote micro-hydro, for instance. Cells named here
        # that the ROI never selected are synthesised from the AOI's own lattice.
        import pandas as pd

        want = [ln.strip() for ln in Path(cells_file).read_text().split() if ln.strip()]
        minx, miny = _snapped_origin(tuple(cfg["bbox"]), cfg.get("grid_origin"))
        have = set(cells["name"]) if len(cells) else set()
        extra = [w for w in want if w not in have]
        rows = []
        for name in extra:
            ix, iy = int(name[:4]), int(name[5:])
            rows.append({"ix": ix, "iy": iy, "name": name, "has_label": False,
                         "line_km": 0.0, "lon0": minx + ix * CELL_DEG,
                         "lat0": miny + iy * CELL_DEG})
        cells = pd.concat([cells[cells["name"].isin(want)], pd.DataFrame(rows)],
                          ignore_index=True) if rows else cells[cells["name"].isin(want)]
        log.info("cells file %s: %d requested, %d from ROI, %d synthesised off-ROI",
                 cells_file, len(want), len(want) - len(extra), len(extra))
    if bbox_filter:
        # Restrict to a sub-window of the AOI. Needed because the composites volume has
        # little headroom, and a derived layer is rarely wanted for every ROI cell.
        x0, y0, x1, y1 = bbox_filter
        n0 = len(cells)
        cells = cells[(cells.lon0 >= x0) & (cells.lon0 <= x1)
                      & (cells.lat0 >= y0) & (cells.lat0 <= y1)]
        log.info("bbox filter %s: %d -> %d cells", bbox_filter, n0, len(cells))
    if limit:
        cells = cells.head(limit)
    # Per-AOI window overrides. The global dry-season window is tuned to Pakistan; a
    # Nov-Mar composite over Great Britain is cloud-wrecked and shot at ~15 deg sun
    # elevation. AOIs without their own key fall through to the global value unchanged.
    _dflt = ("2025-11-01", "2026-03-15")
    s2_window = tuple(cfg.get("s2_window", settings.raw.get("s2_window", _dflt)))
    s1_window = tuple(cfg.get("s1_window", settings.raw.get("s1_window", _dflt)))
    # Pre-construction window for --sensor s2pre. Per-AOI only: there is no sensible
    # global default for "before this class of thing was built".
    s2_pre_window = tuple(cfg.get("s2_pre_window") or ()) or None
    if sensor == "s2pre" and s2_pre_window is None:
        raise KeyError(f"AOI '{aoi}' has no s2_pre_window in configs/aoi.yaml")
    layer = {"s2": "composite_0.tif", "s1": "composite_s1.tif",
             "towerprior": "composite_tower.tif", "dem": "composite_dem.tif",
             "s2pre": "composite_pre.tif"}[sensor]
    bands = {"s2": LOCAL_BANDS, "s1": S1_BANDS,
             "towerprior": TOWER_PRIOR_BANDS, "dem": DEM_BANDS,
             "s2pre": LOCAL_BANDS}[sensor]
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
            elif sensor == "s1":
                # S1 pins to the cell's S2 grid so VV/VH align with composite_0 exactly.
                base = cell_dir / "composite_0.tif"
                if not base.exists():
                    return False  # S1 only where S2 exists (needs the geobox)
                from odc.geo.geobox import GeoBox

                with rasterio.open(base) as b:
                    gbox = GeoBox((b.height, b.width), b.transform, b.crs)
                res = s1_composite(bbox, date_range=s1_window, geobox=gbox)
            elif sensor == "s2pre":
                # Pre-construction S2 composite for the bi-temporal arm. Pinned to the
                # cell's current-epoch grid so the two dates difference pixel-for-pixel
                # with no resampling -- the same geobox trick as S1 and DEM. Deliberately
                # NOT composite_1.tif: that name already means "post-monsoon contrast
                # season" in the Pakistan tree and build_multidate_chips.py reads it.
                base = cell_dir / "composite_0.tif"
                if not base.exists():
                    return False
                from odc.geo.geobox import GeoBox

                with rasterio.open(base) as b:
                    gbox = GeoBox((b.height, b.width), b.transform, b.crs)
                res = annual_composite(bbox, date_range=s2_pre_window, geobox=gbox)
            elif sensor == "dem":
                # Pinned to the cell's S2 grid so terrain aligns with composite_0 exactly,
                # and derived on that metric grid (dx=dy=10 m) rather than on the DEM's
                # own 4326 tiles -- same geobox trick as S1.
                base = cell_dir / "composite_0.tif"
                if not base.exists():
                    return False
                from odc.geo.geobox import GeoBox

                from subdetect.terrain import dem_composite

                with rasterio.open(base) as b:
                    gbox = GeoBox((b.height, b.width), b.transform, b.crs)
                res = dem_composite(bbox, geobox=gbox)
            else:  # towerprior: pure post-processing of the already-composited S1 layer,
                # no STAC fetch — same grid as composite_s1.tif, so co-registration is free.
                base = cell_dir / "composite_s1.tif"
                if not base.exists():
                    return False
                # Imported here, not at module scope: tower_prior pulls in scipy, and a
                # module-level import made `subdetect.compose` unimportable in the data
                # env — breaking `compose` for *every* sensor, not just towerprior.
                from subdetect.tower_prior import tower_prior_dn

                with rasterio.open(base) as b:
                    s1_arr = b.read()
                    res = (tower_prior_dn(s1_arr), b.transform, b.crs)
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
