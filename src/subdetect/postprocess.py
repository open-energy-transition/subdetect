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


def polygonize_chips_v2(prob_dir: Path, lo: float = 0.2, hi: float = 0.4) -> gpd.GeoDataFrame:
    """Hysteresis polygonization with robust per-component scores.

    Components are seeded at `hi` and grown through 4-adjacent pixels >= `lo`, so a
    yard whose interior dips below the old single threshold stays one component and
    single-pixel flukes below `hi` never seed one. `hi` must stay below 0.5: the
    S1x(0.5+0.5*S2) fusion caps S1-only detections at exactly 0.5, and a seed
    threshold at or above that plateau would delete them wholesale.

    Emits per component: conf_max (the legacy score, for comparison), conf_p90
    (top-decile mean -- robust to isolated hot pixels), conf_mean, n_pixels.
    Cross-cell merges take the max of conf_max and recombine the means weighted by
    pixel count (top-decile recombines only approximately).
    """
    from scipy import ndimage

    parts = []
    for tif in tqdm(sorted(prob_dir.glob("*.tif")), desc="polygonize_v2"):
        with rasterio.open(tif) as src:
            prob = src.read(1).astype("float32") / 255.0
            lab, n = ndimage.label(prob >= lo)
            if n == 0:
                continue
            seed_ids = np.unique(lab[prob >= hi])
            seed_ids = seed_ids[seed_ids != 0]
            if seed_ids.size == 0:
                continue
            keep = np.isin(lab, seed_ids)
            rows = []
            for geom, _ in rio_features.shapes(keep.astype("uint8"), mask=keep,
                                               transform=src.transform):
                sel = rio_features.geometry_mask([geom], out_shape=prob.shape,
                                                 transform=src.transform, invert=True)
                vals = prob[sel]
                p90 = float(vals[vals >= np.quantile(vals, 0.9)].mean())
                rows.append((shape(geom), float(vals.max()), p90,
                             float(vals.mean()), int(vals.size)))
            if rows:
                g, cmax, cp90, cmean, npx = zip(*rows)
                parts.append(gpd.GeoDataFrame(
                    {"conf_max": cmax, "conf_p90": cp90, "conf_mean": cmean,
                     "n_pixels": npx},
                    geometry=list(g), crs=src.crs).to_crs("EPSG:4326"))
    cols = ["conf_max", "conf_p90", "conf_mean", "n_pixels"]
    if not parts:
        return gpd.GeoDataFrame({c: [] for c in cols}, geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    u = gdf.union_all()
    merged = gpd.GeoDataFrame(
        geometry=list(u.geoms) if u.geom_type == "MultiPolygon" else [u],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(merged, gdf, how="left", predicate="intersects")
    w = joined["n_pixels"]
    agg = (joined.assign(_wp90=joined.conf_p90 * w, _wmean=joined.conf_mean * w)
           .groupby(joined.index)
           .agg(conf_max=("conf_max", "max"), n_pixels=("n_pixels", "sum"),
                _wp90=("_wp90", "sum"), _wmean=("_wmean", "sum")))
    merged["conf_max"] = agg["conf_max"]
    merged["n_pixels"] = agg["n_pixels"].astype(int)
    merged["conf_p90"] = (agg["_wp90"] / agg["n_pixels"]).round(4)
    merged["conf_mean"] = (agg["_wmean"] / agg["n_pixels"]).round(4)
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
