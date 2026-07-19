"""Urban + building-fill flags and the final ranking for an exported leads file.

Consolidates the veto stack validated on sindh_test (2026-07-19), applied after
the priors (rank_score_tower / rank_score_s1tower from tower_features_eval.py and
isolated_tower_prior.py):

  flag_urban          centroid inside an OSM residential polygon or a 250 m
                      village/hamlet disc ({aoi}_settlements.geojsonseq, see
                      build_settlement_hardneg_chips.py for the extraction).
                      Hard flag only -- soft settlement-distance priors HURT
                      (real substations sit near towns; AUC 0.930 -> 0.875).
  building_fill       fraction of the lead polygon covered by OSM buildings
                      ({aoi}_urbanfill.geojsonseq: building=* + industrial-ish
                      landuse + aeroway).
  vida_building_fill  same, from the VIDA Google+MS Open Buildings country
                      parquet (bbox-pruned duckdb query) -- catches the
                      unmapped-in-OSM buildings; ~12x more flags than OSM.
  flag_building       either fill > 0.25. Real substations are open-air: max
                      observed fill on sindh_test hits is 0.211 (VIDA) / 0.086
                      (OSM), so 0.25 flags zero true hits.
  industrial_overlap  informational ONLY -- 7/32 real sindh_test substations sit
                      inside mapped industrial landuse; vetoing on it drops
                      P@50 0.58 -> 0.46. Never fold it into the rank.

  rank_score_final = rank_score_s1tower * 0.3^flag_urban * 0.3^flag_building

Usage:
  pixi run -e ml python scripts/lead_flags.py \
      --leads data/predictions_v9_mean/new_leads_pakistan_india_meanfusion.geoparquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.errors import GEOSException
from shapely.geometry import shape
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
EQ = "EPSG:6933"
INDUS = {"industrial", "commercial", "retail", "construction", "railway"}
FILL_THRESHOLD = 0.25
VIDA = {"pakistan": "/run/media/tobi/aidisc/earthpv/data/vida/PAK.parquet",
        "india_pilot": "/run/media/tobi/aidata/vida/IND.parquet"}


def geojsonseq(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip().lstrip("\x1e")
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def settlement_flag(part: gpd.GeoDataFrame, seq: Path) -> pd.DataFrame:
    b = part.total_bounds
    geoms = []
    for feat in geojsonseq(seq):
        p = feat.get("properties", {})
        try:
            g = shape(feat["geometry"])
        except Exception:
            continue
        c = g.centroid
        if not (b[0] - 0.1 <= c.x <= b[2] + 0.1 and b[1] - 0.1 <= c.y <= b[3] + 0.1):
            continue
        if p.get("landuse") == "residential" and g.geom_type in ("Polygon", "MultiPolygon"):
            geoms.append(g)
        elif p.get("place") in ("village", "hamlet"):
            geoms.append(c.buffer(0.0025))
    sett = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326").to_crs(EQ)
    tree = STRtree(list(sett.geometry.values))
    sg = sett.geometry.values
    d = np.array([float(sg[tree.nearest(c)].distance(c))
                  for c in part.to_crs(EQ).geometry.centroid])
    return pd.DataFrame({"settle_dist_m": np.round(d, 1), "flag_urban": d <= 1.0},
                        index=part.index)


def osm_fill(part: gpd.GeoDataFrame, seq: Path) -> pd.DataFrame:
    geoms = list(part.geometry.values)
    tree = STRtree(geoms)
    fill = np.zeros(len(geoms))
    indus = np.zeros(len(geoms), bool)
    for feat in geojsonseq(seq):
        try:
            g = shape(feat["geometry"])
        except Exception:
            continue
        if g.geom_type == "Point":
            continue
        p = feat.get("properties", {})
        is_i = p.get("landuse") in INDUS or p.get("aeroway") is not None
        is_b = p.get("building") is not None
        if not (is_i or is_b):
            continue
        for i in tree.query(g):
            try:
                inter = geoms[i].intersection(g)
            except GEOSException:
                continue
            if inter.is_empty:
                continue
            if is_b:
                fill[i] += inter.area
            if is_i:
                indus[i] = True
    ca = np.array([g.area for g in geoms])
    return pd.DataFrame({"building_fill": np.round(np.where(ca > 0, fill / ca, 0), 3),
                         "industrial_overlap": indus}, index=part.index)


def vida_fill(part: gpd.GeoDataFrame, parquet: str) -> pd.Series:
    import duckdb
    from shapely import wkb as shapely_wkb

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial")
    part_eq = part.to_crs(EQ)
    fills = np.zeros(len(part))
    for i, (geom, geq) in enumerate(zip(part.geometry, part_eq.geometry)):
        x0, y0, x1, y1 = geom.bounds
        rows = con.execute(f"""
            SELECT ST_AsWKB(geometry) FROM read_parquet('{parquet}')
            WHERE bbox.xmin <= {x1} AND bbox.xmax >= {x0}
              AND bbox.ymin <= {y1} AND bbox.ymax >= {y0}""").fetchall()
        if rows:
            blds = gpd.GeoSeries([shapely_wkb.loads(bytes(r[0])) for r in rows],
                                 crs="EPSG:4326").to_crs(EQ)
            inter = sum(geq.intersection(bg).area for bg in blds.values
                        if geq.intersects(bg))
            fills[i] = inter / max(geq.area, 1.0)
    return pd.Series(np.round(fills, 3), index=part.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", required=True)
    ap.add_argument("--skip-vida", action="store_true")
    a = ap.parse_args()

    path = ROOT / a.leads
    leads = gpd.read_parquet(path)
    for aoi in leads.aoi.unique():
        m = leads.aoi == aoi
        part = leads[m]
        sf = settlement_flag(part, ROOT / f"data/osm/{aoi}_settlements.geojsonseq")
        of = osm_fill(part, ROOT / f"data/osm/{aoi}_urbanfill.geojsonseq")
        for df in (sf, of):
            for col in df.columns:
                leads.loc[m, col] = df[col]
        if not a.skip_vida and aoi in VIDA and Path(VIDA[aoi]).exists():
            leads.loc[m, "vida_building_fill"] = vida_fill(part, VIDA[aoi])
        print(f"{aoi}: flags done", flush=True)

    if "vida_building_fill" not in leads:
        leads["vida_building_fill"] = 0.0
    leads["flag_building"] = (leads.building_fill > FILL_THRESHOLD) | \
                             (leads.vida_building_fill > FILL_THRESHOLD)
    base = leads["rank_score_s1tower"] if "rank_score_s1tower" in leads else leads["rank_score"]
    leads["rank_score_final"] = np.round(
        base * np.where(leads.flag_urban, 0.3, 1.0)
             * np.where(leads.flag_building, 0.3, 1.0), 4)
    leads = leads.sort_values("rank_score_final", ascending=False).reset_index(drop=True)
    leads.to_parquet(path)
    leads.to_file(str(path).replace(".geoparquet", ".geojson"), driver="GeoJSON")
    print(f"updated {path}: flag_urban={int(leads.flag_urban.sum())}, "
          f"flag_building={int(leads.flag_building.sum())}, "
          f"any={int((leads.flag_urban | leads.flag_building).sum())}/{len(leads)}")


if __name__ == "__main__":
    main()
