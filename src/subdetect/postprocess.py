"""Turn probability rasters into substation candidate polygons, ranked for triage.

Reuses earthpv's polygonize_chips. Replaces the building prior with a power-grid prior:
distance to the nearest transmission line (candidates hug the grid) and a known/new flag
(does the candidate coincide with an already-mapped OSM substation). Recall-first: nothing
is dropped below the area floor's re-rank; every candidate is kept and only re-ordered.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rio_features
from shapely.geometry import shape
from shapely.ops import unary_union
from tqdm import tqdm

from subdetect.config import Settings, geodesic_area_m2, resolve_aoi
from subdetect.local_source import load_lines, load_substation_labels

log = logging.getLogger(__name__)

EQ_AREA = "EPSG:6933"


def polygonize_chips(prob_dir: Path, threshold: float) -> gpd.GeoDataFrame:
    parts = []
    for tif in tqdm(sorted(prob_dir.glob("*.tif")), desc="polygonize"):
        geoms, confs = [], []
        with rasterio.open(tif) as src:
            prob = src.read(1).astype("float32") / 255.0
            hot = (prob >= threshold).astype("uint8")
            if hot.sum() == 0:
                continue
            for geom, _ in rio_features.shapes(hot, mask=hot.astype(bool), transform=src.transform):
                sel = rio_features.geometry_mask([geom], out_shape=prob.shape,
                                                 transform=src.transform, invert=True)
                geoms.append(shape(geom))
                confs.append(float(prob[sel].max()))
            crs = src.crs
        if geoms:
            parts.append(gpd.GeoDataFrame({"confidence": confs}, geometry=geoms, crs=crs)
                         .to_crs("EPSG:4326"))
    if not parts:
        return gpd.GeoDataFrame({"confidence": []}, geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    merged = gpd.GeoDataFrame(
        geometry=list(unary_union(gdf.geometry.values).geoms)
        if gdf.union_all().geom_type == "MultiPolygon" else [gdf.union_all()],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(merged, gdf, how="left", predicate="intersects")
    merged["confidence"] = joined.groupby(joined.index)["confidence"].max()
    return merged


def _grid_prior(aoi: str, cands: gpd.GeoDataFrame, settings: Settings) -> gpd.GeoDataFrame:
    labels_dir = Path("data/labels") / aoi
    lines = load_lines(labels_dir)
    subs = load_substation_labels(labels_dir, settings.min_sub_area_m2)
    cu = cands.to_crs(EQ_AREA).reset_index(drop=True)

    # Nearest transmission line: distance (m) + its voltage.
    lm = lines.to_crs(EQ_AREA).reset_index(drop=True)
    if not lm.empty:
        nearest = gpd.sjoin_nearest(cu, lm[["geometry", "voltage_v"]], how="left",
                                    distance_col="line_dist_m")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        cands["line_dist_m"] = nearest["line_dist_m"].round(1).values
        cands["line_voltage_v"] = nearest["voltage_v"].values
    else:
        cands["line_dist_m"] = -1.0
        cands["line_voltage_v"] = np.nan

    # known vs new: does the candidate coincide with an already-mapped OSM substation?
    known = np.zeros(len(cands), dtype=bool)
    poly = subs[subs.role.isin(["pos", "small"])]
    if not poly.empty:
        hit = gpd.sjoin(cands, poly[["geometry"]], how="left", predicate="intersects")
        known |= cands.index.isin(hit[~hit.index_right.isna()].index)
    nodes = subs[subs.role == "node"]
    if not nodes.empty:
        node_buf = nodes.to_crs(EQ_AREA).buffer(settings.node_ignore_radius_m).union_all()
        known |= cu.geometry.intersects(node_buf).values
    cands["status"] = np.where(known, "known", "new")
    return cands


def _rank(cands: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    dist = cands["line_dist_m"].to_numpy(float).copy()
    dist[dist < 0] = 1e6
    prior = 0.5 + 0.5 * np.exp(-dist / 2000.0)   # near a line -> ~1.0, far -> ~0.5
    conf = cands["confidence"].fillna(0.0).to_numpy(float)
    cands["rank_score"] = (conf * prior).round(4)
    return cands


def run_postprocess(aoi: str, pred_dir: Path, threshold: float = 0.3) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings.load()
    resolve_aoi(aoi, settings)  # validate AOI
    cands = polygonize_chips(Path(pred_dir) / aoi / "prob", threshold)
    log.info("Polygonized %d candidates at threshold %.2f", len(cands), threshold)
    if not cands.empty:
        cands["area_m2"] = [geodesic_area_m2(g) for g in cands.geometry]
        cands = cands[cands.area_m2 >= settings.min_sub_area_m2].reset_index(drop=True)
        cands = _grid_prior(aoi, cands, settings)
        cands = _rank(cands)
        cands = cands.sort_values("rank_score", ascending=False).reset_index(drop=True)
    out = Path(pred_dir) / aoi / "candidates.parquet"
    cands.to_parquet(out)
    log.info("Wrote %s (%s)", out,
             cands.status.value_counts().to_dict() if not cands.empty else "empty")
    return out
