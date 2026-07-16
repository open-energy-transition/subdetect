"""Read composited imagery + OSM substation labels from the local data tree.

CompositeIndex spatially indexes each cell's `composite_0.tif` (S2) and can read any
co-registered per-cell layer by name (e.g. `composite_s1.tif`), so S2 and S1 windows are
returned separately and keep their own scalings (no channel-concat, unlike earthpv's
two-season path).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.merge
import rasterio.warp
from rasterio.crs import CRS
from shapely.geometry import box

from subdetect.config import geodesic_area_m2

log = logging.getLogger(__name__)

Bbox = tuple[float, float, float, float]


class CompositeIndex:
    """Spatial index over `composites/<CELL>/composite_0.tif` COGs of one region."""

    def __init__(self, region_dir: Path):
        self.region_dir = Path(region_dir)
        rows = []
        for tif in sorted(self.region_dir.glob("composites/*/composite_0.tif")):
            try:
                with rasterio.open(tif) as src:
                    geom = box(*rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds))
                    rows.append({"path": str(tif), "crs": str(src.crs), "geometry": geom})
            except rasterio.errors.RasterioIOError as e:
                log.warning("skipping unreadable composite %s: %s", tif, e)
        if not rows:
            raise FileNotFoundError(f"No composites under {self.region_dir}/composites")
        self.index = gpd.GeoDataFrame(rows, crs="EPSG:4326")
        log.info("Indexed %d composite tiles under %s", len(self.index), self.region_dir)

    @property
    def coverage(self):
        return self.index.union_all()

    def read_window(
        self, bbox: Bbox, filename: str = "composite_0.tif"
    ) -> tuple[np.ndarray, rasterio.Affine, CRS] | None:
        """Read a 4326-bbox window from the named per-cell layer; mosaics across tiles.

        Returns None if the bbox is uncovered by the S2 index. Raises FileNotFoundError
        if a covering tile lacks the requested `filename` (e.g. S1 not composited yet).
        """
        hits = self.index[self.index.intersects(box(*bbox))]
        if hits.empty:
            return None
        full = hits[hits.covers(box(*bbox))]
        paths = [full.iloc[0].path] if not full.empty else list(hits.path)
        lpaths = [str(Path(p).with_name(filename)) for p in paths]
        missing = [p for p in lpaths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"missing layer '{filename}': {missing[:3]}")
        srcs = [rasterio.open(p) for p in lpaths]
        try:
            dst_crs = srcs[0].crs
            wb = rasterio.warp.transform_bounds("EPSG:4326", dst_crs, *bbox)
            arr, transform = rasterio.merge.merge(srcs, bounds=wb, nodata=0)
        finally:
            for s in srcs:
                s.close()
        return arr, transform, dst_crs

    def has_layer(self, tile_path: str, filename: str) -> bool:
        return (Path(tile_path).parent / filename).exists()


@lru_cache(maxsize=4)
def composite_index(region_dir: str) -> CompositeIndex:
    return CompositeIndex(Path(region_dir))


def load_substation_labels(labels_dir: Path, min_area_m2: float = 1000.0) -> gpd.GeoDataFrame:
    """Combined substation label set with a `role` column driving mask semantics:

    - role="pos"   substation polygon >= min_area_m2  -> class 1
    - role="small" substation polygon <  min_area_m2  -> ignore (-1)
    - role="node"  power=substation node              -> ignore disc (-1)
    - role="plant" power=plant polygon                -> ignore (-1)

    Prefers `substations_poly_refined.parquet` (see label_refine.py) over the raw
    `substations_poly.parquet` when present: refined geometries are tighter (S1+NDVI
    shrunk), and polygons flagged `status=no_signal` (no S1 backscatter evidence of
    real infrastructure anywhere inside them) are always demoted to role="small"
    regardless of area, since the OSM way likely doesn't cover real transmission-class
    equipment at all.
    """
    labels_dir = Path(labels_dir)
    parts = []

    refined_p = labels_dir / "substations_poly_refined.parquet"
    subs = gpd.read_parquet(refined_p if refined_p.exists() else labels_dir / "substations_poly.parquet")
    if "area_m2" not in subs.columns:
        subs["area_m2"] = [geodesic_area_m2(g) for g in subs.geometry]
    role = np.where(subs.area_m2 >= min_area_m2, "pos", "small")
    if "status" in subs.columns:
        role = np.where(subs.status == "no_signal", "small", role)
    subs = subs.assign(role=role)
    parts.append(subs[["geometry", "area_m2", "voltage_v", "role"]])

    node_p = labels_dir / "substations_node.parquet"
    if node_p.exists():
        nodes = gpd.read_parquet(node_p)
        if not nodes.empty:
            nodes = nodes.assign(area_m2=0.0, role="node")
            if "voltage_v" not in nodes.columns:
                nodes["voltage_v"] = float("nan")
            parts.append(nodes[["geometry", "area_m2", "voltage_v", "role"]])

    plant_p = labels_dir / "plants.parquet"
    if plant_p.exists():
        plants = gpd.read_parquet(plant_p)
        if not plants.empty:
            plants = plants.assign(area_m2=0.0, role="plant")
            if "voltage_v" not in plants.columns:
                plants["voltage_v"] = float("nan")
            parts.append(plants[["geometry", "area_m2", "voltage_v", "role"]])

    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")


def load_lines(labels_dir: Path) -> gpd.GeoDataFrame:
    return gpd.read_parquet(Path(labels_dir) / "lines.parquet")
