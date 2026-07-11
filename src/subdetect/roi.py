"""Region-of-interest cell selection from OSM power infrastructure.

The inference ROI is every 0.1 deg cell (on the Pakistan grid, anchored to earthpv's
`grid_origin` so composited cells are reusable) within a radius (default 20 km) of any
`power=line` corridor or substation. Substations rarely sit far from the transmission
grid, so this bounds the search space while staying recall-generous. Cells containing a
class-1 substation polygon are always included (they carry the training positives).

For India (a training-label source only, not an inference target), `training_cells`
selects just the cells containing large substation polygons, capped and ranked by summed
substation area, so compositing stays bounded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from subdetect.config import Settings, resolve_aoi

log = logging.getLogger(__name__)

CELL_DEG = 0.1
EQ_AREA = "EPSG:6933"  # equal-area for metric buffers


def _snapped_origin(bbox, grid_origin) -> tuple[float, float]:
    """Snap the AOI's (minx, miny) onto grid_origin's lattice (mod CELL_DEG) so cell
    names match another AOI's grid (Pakistan reuses earthpv's composited cells)."""
    minx, miny = bbox[0], bbox[1]
    if grid_origin:
        gx, gy = grid_origin
        minx = gx + np.floor((minx - gx) / CELL_DEG) * CELL_DEG
        miny = gy + np.floor((miny - gy) / CELL_DEG) * CELL_DEG
    return minx, miny


def cell_name(ix: int, iy: int) -> str:
    return f"{int(ix):04d}_{int(iy):04d}"


def _labels_dir(aoi: str) -> Path:
    return Path("data/labels") / aoi


def _read(aoi: str, name: str) -> gpd.GeoDataFrame:
    p = _labels_dir(aoi) / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run `subdetect osm --aoi {aoi}` first")
    return gpd.read_parquet(p)


def _cells_containing(points_or_polys: gpd.GeoDataFrame, minx: float, miny: float) -> pd.DataFrame:
    """(ix, iy) cell indices for each geometry's representative point."""
    pts = points_or_polys.geometry.representative_point()
    ix = np.floor((pts.x.values - minx) / CELL_DEG).astype(int)
    iy = np.floor((pts.y.values - miny) / CELL_DEG).astype(int)
    return pd.DataFrame({"ix": ix, "iy": iy})


def _substation_cells(aoi: str, settings: Settings, minx: float, miny: float) -> set[tuple[int, int]]:
    subs = _read(aoi, "substations_poly")
    big = subs[subs.area_m2 >= settings.min_sub_area_m2]
    if big.empty:
        return set()
    df = _cells_containing(big, minx, miny)
    return set(map(tuple, df.values.tolist()))


def roi_cells(
    aoi: str, cfg: dict, settings: Settings, radius_km: float, min_voltage: float = 0.0
) -> pd.DataFrame:
    """0.1 deg cells within `radius_km` of a power line or substation, on the AOI grid.

    Union of the metric buffer around lines+substations with the cells that contain a
    class-1 substation polygon. Ordered label-cells-first (training positives available
    without waiting for the whole ROI), then by line length per cell.
    """
    bbox = tuple(cfg["bbox"])
    minx, miny = _snapped_origin(bbox, cfg.get("grid_origin"))

    lines = _read(aoi, "lines")
    if min_voltage > 0 and "voltage_v" in lines.columns:
        lines = lines[(lines.voltage_v >= min_voltage) | lines.voltage_v.isna()]
    subs_poly = _read(aoi, "substations_poly")
    subs_node = _read(aoi, "substations_node")

    seeds = pd.concat(
        [lines[["geometry"]], subs_poly[["geometry"]], subs_node[["geometry"]]], ignore_index=True
    )
    seeds = gpd.GeoDataFrame(seeds, crs="EPSG:4326")
    log.info("Buffering %d line/substation seeds by %.0f km", len(seeds), radius_km)
    buf = seeds.to_crs(EQ_AREA).buffer(radius_km * 1000.0).union_all()
    buf = gpd.GeoSeries([buf], crs=EQ_AREA).to_crs("EPSG:4326").iloc[0]

    # Candidate cells: those whose box intersects the buffer's bounding box, tested
    # against the buffer geometry via a spatial index.
    bx0, by0, bx1, by1 = buf.bounds
    ix0 = int(np.floor((bx0 - minx) / CELL_DEG))
    ix1 = int(np.ceil((bx1 - minx) / CELL_DEG))
    iy0 = int(np.floor((by0 - miny) / CELL_DEG))
    iy1 = int(np.ceil((by1 - miny) / CELL_DEG))
    ixs, iys, boxes = [], [], []
    for ix in range(ix0, ix1 + 1):
        lon0 = minx + ix * CELL_DEG
        for iy in range(iy0, iy1 + 1):
            lat0 = miny + iy * CELL_DEG
            ixs.append(ix)
            iys.append(iy)
            boxes.append(box(lon0, lat0, lon0 + CELL_DEG, lat0 + CELL_DEG))
    grid = gpd.GeoDataFrame({"ix": ixs, "iy": iys}, geometry=boxes, crs="EPSG:4326")
    hit = grid[grid.intersects(buf)].copy()

    label_cells = _substation_cells(aoi, settings, minx, miny)
    keys = set(map(tuple, hit[["ix", "iy"]].values.tolist())) | label_cells
    cells = pd.DataFrame(sorted(keys), columns=["ix", "iy"])

    # Priority: label cells first, then by line length inside the cell.
    cells["has_label"] = [(ix, iy) in label_cells for ix, iy in zip(cells.ix, cells.iy)]
    lines_m = lines.to_crs(EQ_AREA)
    cell_boxes = gpd.GeoDataFrame(
        cells,
        geometry=[box(minx + ix * CELL_DEG, miny + iy * CELL_DEG,
                      minx + (ix + 1) * CELL_DEG, miny + (iy + 1) * CELL_DEG)
                  for ix, iy in zip(cells.ix, cells.iy)],
        crs="EPSG:4326",
    ).to_crs(EQ_AREA)
    joined = gpd.sjoin(lines_m[["geometry"]], cell_boxes.reset_index()[["index", "geometry"]],
                       how="inner", predicate="intersects")
    line_km = joined.groupby("index").apply(
        lambda g: g.geometry.length.sum() / 1000.0, include_groups=False
    ) if len(joined) else pd.Series(dtype=float)
    cells["line_km"] = cells.index.map(line_km).fillna(0.0)

    cells = cells.sort_values(["has_label", "line_km"], ascending=[False, False]).reset_index(drop=True)
    cells["name"] = [cell_name(ix, iy) for ix, iy in zip(cells.ix, cells.iy)]
    cells["lon0"] = minx + cells.ix * CELL_DEG
    cells["lat0"] = miny + cells.iy * CELL_DEG
    log.info("ROI: %d cells (%d contain substations), radius %.0f km, min_voltage %.0f",
             len(cells), int(cells.has_label.sum()), radius_km, min_voltage)
    return cells


def training_cells(aoi: str, cfg: dict, settings: Settings, max_cells: int = 0) -> pd.DataFrame:
    """Cells containing class-1 substation polygons, ranked by summed substation area,
    capped at `max_cells` (India pilot). Own grid (no grid_origin reuse)."""
    bbox = tuple(cfg["bbox"])
    minx, miny = _snapped_origin(bbox, cfg.get("grid_origin"))
    subs = _read(aoi, "substations_poly")
    big = subs[subs.area_m2 >= settings.min_sub_area_m2].copy()
    if big.empty:
        raise RuntimeError(f"No substation polygons >= {settings.min_sub_area_m2} m2 for {aoi}")
    df = _cells_containing(big, minx, miny)
    df["area_m2"] = big.area_m2.values
    agg = df.groupby(["ix", "iy"], as_index=False).agg(area_m2=("area_m2", "sum"),
                                                        n=("area_m2", "size"))
    agg = agg.sort_values("area_m2", ascending=False).reset_index(drop=True)
    max_cells = max_cells or int(cfg.get("max_train_cells", 0))
    if max_cells and len(agg) > max_cells:
        log.info("Capping %s training cells %d -> %d (by summed substation area)",
                 aoi, len(agg), max_cells)
        agg = agg.head(max_cells)
    agg["has_label"] = True
    agg["name"] = [cell_name(ix, iy) for ix, iy in zip(agg.ix, agg.iy)]
    agg["lon0"] = minx + agg.ix * CELL_DEG
    agg["lat0"] = miny + agg.iy * CELL_DEG
    log.info("%s training cells: %d (%d substations)", aoi, len(agg), int(agg.n.sum()))
    return agg


def select_cells(
    aoi: str, cfg: dict, settings: Settings, radius_km: float, min_voltage: float = 0.0
) -> pd.DataFrame:
    """Compose cell set for an AOI: full ROI for inference targets (Pakistan),
    capped training cells for label-only sources (India pilot with max_train_cells)."""
    if cfg.get("max_train_cells"):
        return training_cells(aoi, cfg, settings)
    return roi_cells(aoi, cfg, settings, radius_km, min_voltage)


def run_roi(aoi: str, radius_km: float = 20.0, min_voltage: float = 0.0, plot: bool = True) -> Path:
    """Compute + persist the ROI cell list (and a diagnostic PNG) for an AOI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings.load()
    _, cfg = resolve_aoi(aoi, settings)
    cells = select_cells(aoi, cfg, settings, radius_km, min_voltage)
    out_dir = Path("data/roi")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{aoi}_cells.parquet"
    cells.to_parquet(out)
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 10))
            lines = _read(aoi, "lines")
            lines.plot(ax=ax, color="0.6", linewidth=0.3)
            gpd.GeoDataFrame(
                cells,
                geometry=[box(x, y, x + CELL_DEG, y + CELL_DEG) for x, y in zip(cells.lon0, cells.lat0)],
                crs="EPSG:4326",
            ).boundary.plot(ax=ax, color="tab:blue", linewidth=0.2)
            _read(aoi, "substations_poly").plot(ax=ax, color="tab:red", markersize=2)
            ax.set_title(f"{aoi}: {len(cells)} ROI cells, radius {radius_km:.0f} km")
            fig.savefig(out_dir / f"{aoi}_roi.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # noqa: BLE001 — the PNG is a nicety, not required
            log.warning("ROI plot failed: %s", e)
    log.info("Wrote %d ROI cells -> %s", len(cells), out)
    return out
