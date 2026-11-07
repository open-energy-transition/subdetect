"""A second field-evaluation region, built the way sindh_test was (v18, 2026-11-05).

sindh_test is the only region every model has been judged on: 112 substations of >= 1,000 m2
in its rasters (9 / 39 / 50 / 14 by size bucket), so two or three hits move every metric,
and reseeds have moved fused AUC by 0.04. This plans and composes a second region with
osmose_detect's own steps 1-4 (open Osmose "unfinished major power line" endpoints more than
700 m from a mapped substation, every 0.1 deg cell within 10 km of one, S2 + S1 dry-season
composites on the same windows), and stops there: no inference, so no GPU and no model is
baked in. Models are scored on it with scripts/infer_region_ckpts.py and field_eval.py
(Overpass ground truth, like any region without local labels).

TorchGeo has training chips almost everywhere (3,804 in India, 26 inside the Sindh holdout),
so `exclusion` lists the TorchGeo chips inside a region's cells; v18 training indexes drop
them (scripts/build_v18_indexes.py).

Usage:
  pixi run python scripts/build_test_region.py plan --region gujarat_test \
      --country india_gujarat --bbox 68.0,21.5,72.6,24.7
  pixi run python scripts/build_test_region.py compose --region gujarat_test --workers 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
log = logging.getLogger("build_test_region")

# A stalled HTTP read otherwise hangs a fetch for good (2026-11-05: three S1 sockets idle 12 min
# with no error). GDAL drops a transfer under 1 kB/s for 60 s and retries it, like earthpv's
# compose._harden_python_network.
for _k, _v in {"GDAL_HTTP_TIMEOUT": "180", "GDAL_HTTP_LOW_SPEED_TIME": "60",
               "GDAL_HTTP_LOW_SPEED_LIMIT": "1024", "GDAL_HTTP_MAX_RETRY": "4",
               "GDAL_HTTP_RETRY_DELAY": "3", "GDAL_HTTP_TCP_KEEPALIVE": "YES"}.items():
    os.environ.setdefault(_k, _v)
REG = ROOT / "data/osmose_regions"


def cmd_plan(a) -> None:
    import osmose_detect as od

    from subdetect.roi import CELL_DEG

    out = REG / a.region
    out.mkdir(parents=True, exist_ok=True)
    bbox = tuple(float(x) for x in a.bbox.split(","))
    issues = od.fetch_osmose(a.country, bbox, tile_deg=a.tile_deg)
    pts = gpd.GeoDataFrame(issues, geometry=[Point(xy) for xy in zip(issues.lon, issues.lat)],
                           crs="EPSG:4326")
    pts = pts[pts.within(box(*bbox))]
    ext = pts.total_bounds
    subs = od.fetch_substations((ext[0] - 0.2, ext[1] - 0.2, ext[2] + 0.2, ext[3] + 0.2))
    near = gpd.sjoin_nearest(pts.to_crs(od.EQ), subs.to_crs(od.EQ)[["geometry"]], how="left",
                             distance_col="sub_dist_m")
    near = near[~near.index.duplicated(keep="first")]
    pts["sub_dist_m"] = near["sub_dist_m"].round(1).values
    keep = pts[pts.sub_dist_m > a.sub_dist_m].reset_index(drop=True)
    log.info("%d/%d endpoints are >%.0f m from any mapped substation", len(keep), len(pts),
             a.sub_dist_m)
    keep.to_file(out / "endpoints.geojson", driver="GeoJSON")
    buf = keep.to_crs(od.EQ).buffer(a.search_km * 1000).to_crs("EPSG:4326").union_all()
    cells, b = {}, buf.bounds
    for ix in range(int(np.floor(b[0] / CELL_DEG)), int(np.ceil(b[2] / CELL_DEG))):
        for iy in range(int(np.floor(b[1] / CELL_DEG)), int(np.ceil(b[3] / CELL_DEG))):
            cb = (ix * CELL_DEG, iy * CELL_DEG, (ix + 1) * CELL_DEG, (iy + 1) * CELL_DEG)
            if box(*cb).intersects(buf):
                cells[f"{ix:05d}_{iy:05d}"] = cb
    (out / "cells.json").write_text(json.dumps(
        {"bbox": bbox, "country": a.country, "cells": cells}, indent=1))
    log.info("cell plan: %d cells -> %s", len(cells), out / "cells.json")


def cmd_compose(a) -> None:
    import osmose_detect as od

    from subdetect.config import Settings

    out = REG / a.region
    cells = json.loads((out / "cells.json").read_text())["cells"]
    raw = Settings.load().raw
    s2w = tuple(raw.get("s2_window", ("2025-11-01", "2026-03-15")))
    s1w = tuple(raw.get("s1_window", ("2025-11-01", "2026-03-15")))
    comp = out / "composites"
    todo = [n for n in sorted(cells) if not ((comp / n / "composite_0.tif").exists()
                                             and (comp / n / "composite_s1.tif").exists())]
    log.info("%d of %d cells to compose (workers %d)", len(todo), len(cells), a.workers)
    with ThreadPoolExecutor(a.workers) as ex:
        oks = list(ex.map(lambda n: od.compose_cell(comp / n, tuple(cells[n]), s2w, s1w), todo))
    log.info("composed %d of %d; %d failed (re-run to retry)", sum(oks), len(todo),
             len(todo) - sum(oks))


def cmd_labels(a) -> None:
    """data/labels/<region>/substations_{poly,node}.parquet from Overpass, in the Pakistan
    labels' schema (geometry, voltage_v, name, area_m2), so field_eval.py scores area buckets
    exactly as on sindh_test. Ways become polygons; relations (multipolygons) are skipped like
    any unclosed way and counted in the log."""
    import requests
    from shapely.geometry import Polygon

    import osmose_detect as od

    from subdetect.config import geodesic_area_m2

    cells = json.loads((REG / a.region / "cells.json").read_text())["cells"]
    xs = [v for b in cells.values() for v in (b[0], b[2])]
    ys = [v for b in cells.values() for v in (b[1], b[3])]
    bb = (min(xs) - 0.1, min(ys) - 0.1, max(xs) + 0.1, max(ys) + 0.1)
    q = f"""[out:json][timeout:240];
    nwr["power"="substation"]({bb[1]},{bb[0]},{bb[3]},{bb[2]});
    out geom;"""
    r = requests.post(od.OVERPASS_API, data={"data": q}, headers=od.UA, timeout=300)
    r.raise_for_status()

    def volt(t):
        try:
            return float(str(t.get("voltage", "")).split(";")[0])
        except ValueError:
            return float("nan")

    polys, nodes, skipped = [], [], 0
    for el in r.json().get("elements", []):
        t = el.get("tags", {})
        if el["type"] == "node":
            nodes.append({"geometry": Point(el["lon"], el["lat"]), "voltage_v": volt(t),
                          "name": t.get("name")})
        elif el["type"] == "way" and len(el.get("geometry", [])) >= 4 \
                and el["geometry"][0] == el["geometry"][-1]:
            g = Polygon([(p["lon"], p["lat"]) for p in el["geometry"]])
            if g.is_valid and g.area > 0:
                polys.append({"geometry": g, "voltage_v": volt(t), "name": t.get("name"),
                              "area_m2": geodesic_area_m2(g)})
                continue
            skipped += 1
        else:
            skipped += 1
    out = ROOT / "data/labels" / a.region
    out.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(polys, crs="EPSG:4326").to_parquet(out / "substations_poly.parquet")
    gpd.GeoDataFrame(nodes, crs="EPSG:4326").to_parquet(out / "substations_node.parquet")
    log.info("labels: %d polygons, %d nodes, %d skipped (relations/unclosed) -> %s",
             len(polys), len(nodes), skipped, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--region", required=True)
    p.add_argument("--country", required=True)
    p.add_argument("--bbox", required=True)
    p.add_argument("--sub-dist-m", type=float, default=700.0)
    p.add_argument("--search-km", type=float, default=10.0)
    p.add_argument("--tile-deg", type=float, default=2.0)
    c = sub.add_parser("compose")
    c.add_argument("--region", required=True)
    c.add_argument("--workers", type=int, default=2)
    lb = sub.add_parser("labels")
    lb.add_argument("--region", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"plan": cmd_plan, "compose": cmd_compose, "labels": cmd_labels}[a.cmd](a)


if __name__ == "__main__":
    main()
