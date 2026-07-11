"""Osmose 'unfinished power line' issues -> missing-substation leads.

Fetches Osmose QA issues (item 7040, default class 2 = unfinished major power
line) for one or more Osmose country codes, drops issues that are within
--sub-dist-m of any OSM-mapped substation (from local labels), and searches the
model candidate polygons within --search-km of each surviving endpoint. Output:
one GeoJSON of endpoints (with match info) and one of matched candidate leads.

Issues are fetched by bbox tiles over the AOI (the API caps at ~500 per request),
deduplicated by issue id.

Usage:
  pixi run python scripts/osmose_leads.py --aoi pakistan \
      --candidates data/predictions_v3b/pakistan/candidates.parquet
  pixi run python scripts/osmose_leads.py --aoi india_pilot \
      --candidates data/predictions_v3b/india_pilot/candidates_full.parquet
"""
import argparse, logging, sys, time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

sys.path.insert(0, "src")
from subdetect.local_source import load_substation_labels
from subdetect.config import Settings, resolve_aoi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("osmose")

API = "https://osmose.openstreetmap.fr/api/0.3/issues"
EQ = "EPSG:6933"
UA = {"User-Agent": "subdetect-osmose-leads/0.1 (OpenEnergyTransition; substation mapping research)"}


def fetch_issues_tiled(bbox, item: int, klass: str, tile_deg: float = 2.0,
                       limit: int = 500) -> pd.DataFrame:
    """Fetch issues over the AOI bbox in tiles (the API caps at ~500 per request).

    The response carries only {id, lat, lon, item} — the class is tagged from the
    request parameters, not parsed from the payload.
    """
    rows = []
    lon = bbox[0]
    while lon < bbox[2]:
        lat = bbox[1]
        while lat < bbox[3]:
            tb = (lon, lat, min(lon + tile_deg, bbox[2]), min(lat + tile_deg, bbox[3]))
            params = {"item": item, "class": klass, "status": "open", "limit": limit,
                      "bbox": f"{tb[0]},{tb[1]},{tb[2]},{tb[3]}"}
            r = requests.get(API, params=params, headers=UA, timeout=60)
            r.raise_for_status()
            issues = r.json().get("issues", [])
            if len(issues) >= limit:
                log.warning("tile %s hit the %d cap; consider smaller tile_deg", tb, limit)
            rows += [{"osmose_id": i.get("id"), "lat": i.get("lat"), "lon": i.get("lon"),
                      "class": klass} for i in issues]
            time.sleep(0.5)  # be polite to the QA API
            lat += tile_deg
        lon += tile_deg
    df = pd.DataFrame(rows).drop_duplicates("osmose_id") if rows else pd.DataFrame(rows)
    log.info("bbox %s: %d unique open issues (item %s class %s)", bbox, len(df), item, klass)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True, help="AOI name (for labels + bbox tiling)")
    ap.add_argument("--candidates", required=True, help="Model candidates parquet")
    ap.add_argument("--item", type=int, default=7040, help="Osmose item (7040 = power lines)")
    ap.add_argument("--classes", default="2", help="Osmose class(es), comma-sep (2 = unfinished major line)")
    ap.add_argument("--sub-dist-m", type=float, default=700.0,
                    help="Drop issues within this distance of a mapped substation")
    ap.add_argument("--search-km", type=float, default=5.0,
                    help="Search radius around each endpoint for model candidates")
    ap.add_argument("--out-dir", default="data/osmose")
    a = ap.parse_args()

    settings = Settings.load()
    _, cfg = resolve_aoi(a.aoi, settings)
    bbox = cfg["bbox"]

    frames = [fetch_issues_tiled(bbox, a.item, klass.strip())
              for klass in a.classes.split(",")]
    issues = pd.concat(frames, ignore_index=True).dropna(subset=["lat", "lon"])
    if not issues.empty:
        issues = issues.drop_duplicates("osmose_id")
    if issues.empty:
        log.info("No open issues found; nothing to do.")
        return
    pts = gpd.GeoDataFrame(issues, geometry=[Point(xy) for xy in zip(issues.lon, issues.lat)],
                           crs="EPSG:4326").reset_index(drop=True)
    log.info("%d issues inside AOI bbox %s", len(pts), bbox)

    # drop endpoints already near ANY mapped substation (any size, incl. nodes)
    subs = load_substation_labels(Path("data/labels") / a.aoi, min_area_m2=0.0)
    pu = pts.to_crs(EQ)
    su = subs.to_crs(EQ)
    joined = gpd.sjoin_nearest(pu, su[["geometry"]], how="left", distance_col="sub_dist_m")
    joined = joined[~joined.index.duplicated(keep="first")]
    pts["sub_dist_m"] = joined["sub_dist_m"].round(1).values
    keep = pts[pts.sub_dist_m > a.sub_dist_m].reset_index(drop=True)
    log.info("%d endpoints remain after dropping %d within %.0f m of a mapped substation",
             len(keep), len(pts) - len(keep), a.sub_dist_m)

    # search model candidates within the radius of each endpoint
    cands = gpd.read_parquet(a.candidates)
    cu = cands.to_crs(EQ)
    ku = keep.to_crs(EQ)
    radius = a.search_km * 1000.0
    score_col = "lead_score" if "lead_score" in cands.columns else "rank_score"
    matches = []
    for i, g in enumerate(ku.geometry):
        buf = g.buffer(radius)
        idxs = [j for j in cu.sindex.query(buf) if cu.geometry.iloc[j].intersects(buf)]
        for j in idxs:
            matches.append({
                "endpoint_idx": i, "cand_idx": int(j),
                "dist_m": round(float(g.distance(cu.geometry.iloc[j])), 1),
                "status": cands.iloc[j].get("status", ""),
                "score": float(cands.iloc[j].get(score_col, np.nan)),
            })
    m = pd.DataFrame(matches)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.aoi

    keep["n_candidates_in_radius"] = 0
    keep["best_cand_dist_m"] = np.nan
    keep["best_cand_score"] = np.nan
    if not m.empty:
        grp = m.groupby("endpoint_idx")
        keep.loc[grp.size().index, "n_candidates_in_radius"] = grp.size().values
        best = m.sort_values(["endpoint_idx", "score"], ascending=[True, False]) \
                .drop_duplicates("endpoint_idx").set_index("endpoint_idx")
        keep.loc[best.index, "best_cand_dist_m"] = best["dist_m"].values
        keep.loc[best.index, "best_cand_score"] = best["score"].values
    ep_out = out_dir / f"{tag}_unfinished_line_endpoints.geojson"
    keep.to_file(ep_out, driver="GeoJSON")

    lead_rows = []
    if not m.empty:
        new_m = m[m.status == "new"]
        for cand_idx, sub in new_m.groupby("cand_idx"):
            c = cands.iloc[cand_idx]
            lead_rows.append({
                "geometry": c.geometry, "score": float(sub.score.iloc[0]),
                "min_endpoint_dist_m": float(sub.dist_m.min()),
                "n_endpoints": int(sub.endpoint_idx.nunique()),
                "area_m2": float(c.get("area_m2", np.nan)),
                "confidence": float(c.get("confidence", np.nan)),
            })
    leads = gpd.GeoDataFrame(lead_rows, crs="EPSG:4326") if lead_rows else \
        gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    if not leads.empty:
        leads = leads.sort_values(["n_endpoints", "score"], ascending=False)
    ld_out = out_dir / f"{tag}_osmose_matched_leads.geojson"
    leads.to_file(ld_out, driver="GeoJSON")

    log.info("wrote %s (%d endpoints, %d with a candidate in %.1f km)",
             ep_out, len(keep), int((keep.n_candidates_in_radius > 0).sum()), a.search_km)
    log.info("wrote %s (%d NEW candidate polygons matched to unfinished lines)",
             ld_out, len(leads))


if __name__ == "__main__":
    main()
