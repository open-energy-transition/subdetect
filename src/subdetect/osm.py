"""OSM power-infrastructure ingestion from Geofabrik PBF extracts via osmium-tool.

Overpass is unreliable at country scale and Overture S3 times out from this machine,
so labels come from Geofabrik country PBFs, filtered with `osmium tags-filter` and
converted to GeoJSONSeq with `osmium export`, then split into per-role GeoParquets:

    data/labels/<aoi>/
      substations_poly.parquet   power=substation polygons (+area_m2, voltage)  -> class 1 / ignore
      substations_node.parquet   power=substation nodes    (+voltage)           -> ignore discs
      plants.parquet             power=plant polygons                           -> ignore
      lines.parquet              power=line linestrings     (+voltage)          -> ROI + hard negatives

`build_labels` is idempotent per stage: it skips the download and osmium steps whose
outputs already exist, so a re-run only redoes the cheap Python split.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd

from subdetect.config import Settings, geodesic_area_m2, resolve_aoi

log = logging.getLogger(__name__)

GEOFABRIK = {
    "PK": "https://download.geofabrik.de/asia/pakistan-latest.osm.pbf",
    "IN": "https://download.geofabrik.de/asia/india-latest.osm.pbf",
}

# osmium export geometry config: keep the power tags we need, let osmium assemble
# closed power=substation/plant ways + multipolygon relations as polygons and
# power=line ways as linestrings (its default area detection follows the osm2pgsql
# polygon heuristic, under which `power` is an area key).
_EXPORT_CONFIG = {
    "attributes": {"type": "@type", "id": "@id"},
    "linear_tags": True,
    "area_tags": True,
    "include_tags": ["power", "voltage", "name", "substation", "operator"],
}


def _run(cmd: list[str]) -> None:
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _parse_voltage(v) -> float:
    """OSM voltage tag -> max volts as float (handles '220000', '132000;66000'). NaN on junk."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return float("nan")
    nums = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", str(v))]
    return max(nums) if nums else float("nan")


def _download(country: str, out_dir: Path) -> Path:
    url = GEOFABRIK[country]
    pbf = out_dir / Path(url).name
    if pbf.exists() and pbf.stat().st_size > 0:
        log.info("PBF present: %s (%.0f MB)", pbf, pbf.stat().st_size / 1e6)
        return pbf
    _run(["wget", "-c", "-q", "--show-progress", url, "-O", str(pbf)])
    return pbf


def build_labels(aoi: str, out_dir: Path) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings.load()
    bbox, cfg = resolve_aoi(aoi, settings)
    country = cfg["division"]["country"]
    osm_dir = Path("data/osm")
    osm_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir) / aoi
    out_dir.mkdir(parents=True, exist_ok=True)

    pbf = _download(country, osm_dir)

    # 1. Filter to power infrastructure (nodes+ways+relations for subs/plants, ways for lines).
    power = osm_dir / f"{aoi}_power.osm.pbf"
    if not power.exists():
        _run([
            "osmium", "tags-filter", "-o", str(power), "--overwrite", str(pbf),
            "nwr/power=substation", "nwr/power=plant", "w/power=line",
        ])

    # 2. Sub-country AOIs (e.g. india_pilot) clip to their bbox; a country extract does not.
    if cfg["division"].get("subtype") != "country":
        clipped = osm_dir / f"{aoi}_power_clip.osm.pbf"
        if not clipped.exists():
            _run([
                "osmium", "extract", "-s", "smart", "--overwrite",
                "-b", ",".join(str(x) for x in bbox), "-o", str(clipped), str(power),
            ])
        power = clipped

    # 3. Export to GeoJSONSeq with assembled geometries + stable ids.
    cfg_path = osm_dir / "export_config.json"
    cfg_path.write_text(json.dumps(_EXPORT_CONFIG))
    gjs = osm_dir / f"{aoi}_power.geojsonseq"
    _run([
        "osmium", "export", str(power), "-f", "geojsonseq", "--overwrite",
        "-c", str(cfg_path), "--add-unique-id=type_id", "-o", str(gjs),
    ])

    # 4. Read + split by role (GDAL GeoJSONSeq driver via pyogrio).
    gdf = gpd.read_file(gjs, engine="pyogrio")
    if "power" not in gdf.columns:
        gdf["power"] = None
    gdf["voltage_v"] = gdf["voltage"].map(_parse_voltage) if "voltage" in gdf.columns else float("nan")
    gt = gdf.geometry.geom_type
    is_poly = gt.isin(["Polygon", "MultiPolygon"])
    is_line = gt.isin(["LineString", "MultiLineString"])
    is_pt = gt == "Point"

    sub = gdf["power"] == "substation"
    plant = gdf["power"] == "plant"
    line = gdf["power"] == "line"

    keep_poly = ["geometry", "voltage_v", "name"] if "name" in gdf.columns else ["geometry", "voltage_v"]

    subs_poly = gdf.loc[sub & is_poly, keep_poly].copy()
    subs_poly["area_m2"] = [geodesic_area_m2(g) for g in subs_poly.geometry]
    subs_node = gdf.loc[sub & is_pt, [c for c in keep_poly if c != "geometry"] + ["geometry"]].copy()
    plants = gdf.loc[plant & is_poly, ["geometry", "voltage_v"]].copy()
    lines = gdf.loc[line & is_line, ["geometry", "voltage_v"]].copy()

    for name, part in [("substations_poly", subs_poly), ("substations_node", subs_node),
                       ("plants", plants), ("lines", lines)]:
        p = out_dir / f"{name}.parquet"
        gpd.GeoDataFrame(part, crs="EPSG:4326").reset_index(drop=True).to_parquet(p)

    n_big = int((subs_poly.area_m2 >= settings.min_sub_area_m2).sum()) if len(subs_poly) else 0
    log.info(
        "%s: %d substation polygons (%d >= %.0f m2), %d nodes, %d plants, %d lines -> %s",
        aoi, len(subs_poly), n_big, settings.min_sub_area_m2,
        len(subs_node), len(plants), len(lines), out_dir,
    )
    return out_dir
