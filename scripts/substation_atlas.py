"""Self-contained HTML atlas of new substation leads, styled like earthpv's
pakistan_pv_atlas.html (same night-lights choropleth, blue "grid" palette instead of
solar amber, leads instead of capacity).

Aggregates data/predictions_v9_mean/new_leads_pakistan_india_meanfusion.geoparquet
(or any leads file with the same schema: aoi, status, rank_score_final, geometry) to
a 0.1 deg grid, fetches ADM1 province/state polygons from geoBoundaries for the land
layer + region ranking, and renders scripts/templates/substation_atlas.html.

Usage:
  pixi run -e ml python scripts/substation_atlas.py \
      --leads data/predictions_v9_mean/new_leads_pakistan_india_meanfusion.geoparquet \
      --out docs/assets/substation_atlas.html
"""
from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape as shapely_shape

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts/templates/substation_atlas.html"
log = logging.getLogger("substation_atlas")

GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/ADM1/"
ISO3 = {"pakistan": "PAK", "india_pilot": "IND"}
CITIES = {
    "pakistan": [
        ["Karachi", 67.01, 24.86], ["Lahore", 74.34, 31.55], ["Islamabad", 73.05, 33.68],
        ["Faisalabad", 73.08, 31.42], ["Multan", 71.52, 30.2], ["Peshawar", 71.58, 34.01],
        ["Quetta", 66.98, 30.18], ["Hyderabad", 68.37, 25.4], ["Rawalpindi", 73.07, 33.6],
        ["Sukkur", 68.85, 27.7],
    ],
    "india_pilot": [
        ["Ludhiana", 75.86, 30.90], ["Amritsar", 74.87, 31.63], ["Chandigarh", 76.78, 30.73],
        ["Jalandhar", 75.58, 31.33], ["Patiala", 76.38, 30.34],
    ],
}


def fetch_admin1(iso3: str) -> gpd.GeoDataFrame | None:
    cache = ROOT / f"data/osm/admin1_{iso3}.geojson"
    if cache.exists():
        gj = json.loads(cache.read_text())
    else:
        try:
            meta = json.load(urllib.request.urlopen(GEOBOUNDARIES_API.format(iso3=iso3), timeout=30))
            gj = json.load(urllib.request.urlopen(meta["gjDownloadURL"], timeout=120))
            cache.write_text(json.dumps(gj))
        except Exception as e:  # noqa: BLE001 — degrade to no admin layer
            log.warning("geoBoundaries %s fetch failed: %s", iso3, e)
            return None
    rows = [{"name": f["properties"].get("shapeName"),
             "geometry": shapely_shape(f["geometry"])} for f in gj.get("features", [])]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _rings(geom, tolerance: float = 0.03) -> list:
    simple = geom.simplify(tolerance, preserve_topology=True)
    polys = getattr(simple, "geoms", [simple])
    return [[[round(x, 3), round(y, 3)] for x, y in p.exterior.coords]
            for p in polys if not p.is_empty]


def build_atlas(leads_path: Path, out: Path, title: str = "Pakistan + India") -> Path:
    leads = gpd.read_parquet(leads_path)
    score_col = "rank_score_final" if "rank_score_final" in leads else (
        "rank_score_s1tower" if "rank_score_s1tower" in leads else "rank_score")
    flag_cols = [c for c in ("flag_urban", "flag_building") if c in leads.columns]
    leads["_flagged"] = leads[flag_cols].any(axis=1) if flag_cols else False
    new = leads[leads.status == "new"] if "status" in leads else leads
    known = leads[leads.status == "known"] if "status" in leads else leads.iloc[0:0]
    new = new.set_geometry(new.geometry.centroid)
    if len(known):
        known = known.set_geometry(known.geometry.centroid)

    lon0 = np.floor(new.geometry.x / 0.1) * 0.1
    lat0 = np.floor(new.geometry.y / 0.1) * 0.1
    grid = (new.assign(lon0=lon0.round(1), lat0=lat0.round(1))
                .groupby(["lon0", "lat0"])
                .agg(n_new=("_flagged", "size"), n_flagged=("_flagged", "sum"),
                     top_score=(score_col, "max"))
                .reset_index())
    kx = (np.floor(known.geometry.x / 0.1) * 0.1).round(1) if len(known) else pd.Series([], dtype=float)
    ky = (np.floor(known.geometry.y / 0.1) * 0.1).round(1) if len(known) else pd.Series([], dtype=float)
    kgrid = (pd.DataFrame({"lon0": kx.values, "lat0": ky.values}).groupby(["lon0", "lat0"])
                .size().rename("n_known").reset_index()) if len(known) else pd.DataFrame(
                columns=["lon0", "lat0", "n_known"])
    grid = grid.merge(kgrid, on=["lon0", "lat0"], how="left")
    grid["n_known"] = grid.n_known.fillna(0).astype(int)

    bounds = [round(float(new.geometry.x.min()) - 0.05, 3), round(float(new.geometry.y.min()) - 0.05, 3),
              round(float(new.geometry.x.max()) + 0.15, 3), round(float(new.geometry.y.max()) + 0.15, 3)]
    cells = [[float(r.lon0), float(r.lat0), int(r.n_new), int(r.n_known), int(r.n_flagged),
              round(float(r.top_score), 3)] for r in grid.itertuples()]

    provinces, cities = [], []
    for aoi in leads.aoi.unique():
        iso3 = ISO3.get(aoi)
        if not iso3:
            continue
        cities += CITIES.get(aoi, [])
        adm = fetch_admin1(iso3)
        if adm is None:
            continue
        part_new = new[new.aoi == aoi]
        part_known = known[known.aoi == aoi] if len(known) else known
        joined = gpd.sjoin(part_new, adm, how="left", predicate="within")
        for _, prov in adm.iterrows():
            n_new = int((joined.name == prov["name"]).sum())
            if n_new == 0 and prov.geometry.area < 0.01:
                continue
            n_flagged = int(joined[joined.name == prov["name"]]._flagged.sum()) if n_new else 0
            n_known = 0
            if len(part_known):
                jk = gpd.sjoin(part_known, gpd.GeoDataFrame([prov], crs=adm.crs), how="inner",
                               predicate="within")
                n_known = len(jk)
            area_km2 = max(gpd.GeoSeries([prov.geometry], crs="EPSG:4326").to_crs("EPSG:6933")
                          .area.iloc[0] / 1e6, 1e-6)
            provinces.append({"name": str(prov["name"]), "n_new": n_new, "n_known": n_known,
                              "n_flagged": n_flagged, "dens": n_new / area_km2 * 1000.0,
                              "rings": _rings(prov.geometry)})
    provinces.sort(key=lambda p: -p["n_new"])

    data = {
        "bounds": bounds, "cells": cells, "provinces": provinces, "cities": cities,
        "totals": {"n_new": int(len(new)), "n_known": int(len(known)),
                  "n_flagged": int(new._flagged.sum()), "n_cells": int(len(grid))},
    }

    lede = (
        f"Two single-modality TerraMind models — one reading a year of Sentinel-1 radar, "
        f"one reading Sentinel-2 optical — each mark substation-shaped pixels across every "
        f"grid-corridor cell of {title}. Their probabilities are fused, polygonized, ranked "
        f"by corridor and tower priors, and matched against OpenStreetMap: what's left is a "
        f"<b>new lead</b> — a probable substation nobody has mapped yet."
    )
    bracket = (
        f'<b>{data["totals"]["n_known"]:,}</b> other candidates already match a mapped '
        f'OSM substation (shown here as context, not as leads). '
        f'<b>{data["totals"]["n_flagged"]:,}</b> new leads are flagged as likely urban/building '
        f'false positives (dense settlement or high building-footprint fill) and ranked low.'
    )
    howto = (
        "<b>How to read it.</b> Colour is the count of new leads per 0.1° cell "
        "(log scale). Hover a cell for its known-match count and top rank score. "
        "Ranking multiplies model confidence by distance-to-mapped-line, mapped-tower, "
        "and isolated-S1-tower priors, then down-weights leads flagged as sitting inside "
        "a mapped settlement or over high building-footprint fill (OSM + VIDA Open "
        "Buildings) — validated to not suppress any real substation on the sindh_test "
        "field-evaluation region."
    )
    method_lede = (
        "Full pipeline: OSM-guided ROI + hard-negative mining, dual-arm TerraMind "
        "segmentation (Sentinel-1 and Sentinel-2 trained independently), decision-level "
        "mean fusion, hysteresis polygonization, and a prior/veto ranking stack — see "
        "docs/architecture.md for the stage-by-stage detail and docs/assets/pipeline.svg "
        "for the high-level flow chart."
    )

    html = TEMPLATE.read_text()
    for key, value in {
        "__SUB_DATA_JSON__": json.dumps(data, separators=(",", ":")),
        "__PAGE_TITLE__": f"{title} Substation Lead Atlas",
        "__H1__": f"Where {title}'s grid still needs mapping",
        "__LEDE_HTML__": lede,
        "__BRACKET_HTML__": bracket,
        "__HOWTO_HTML__": howto,
        "__METHOD_LEDE__": method_lede,
        "__N_CELLS_TOTAL__": f"{len(grid):,}",
        "__FOOT_MODEL__": (
            "Model: dual-arm TerraMind-tiny (Sentinel-1 v5 + Sentinel-2 v9), mean-fused · "
            "threshold 0.30 · Sentinel-1 RTC + Sentinel-2 L2A composites · "
            "buildings from OpenStreetMap + VIDA Open Buildings · provinces from geoBoundaries."
        ),
    }.items():
        html = html.replace(key, value)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    log.info("wrote %s (%d new leads, %d cells, %d provinces)",
             out, len(new), len(grid), len(provinces))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", default="data/predictions_v9_mean/new_leads_pakistan_india_meanfusion.geoparquet")
    ap.add_argument("--out", default="docs/assets/substation_atlas.html")
    ap.add_argument("--title", default="Pakistan + India")
    a = ap.parse_args()
    build_atlas(ROOT / a.leads, ROOT / a.out, a.title)


if __name__ == "__main__":
    main()
