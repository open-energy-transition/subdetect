"""Export substation candidates for OSM validation workflows.

Outputs:
- <aoi>_sub_candidates.geoparquet / .geojson — full attribute set (known + new)
- <aoi>_sub_maproulette.geojson — line-delimited FeatureCollections, one task per *new*
  candidate (known ones are already mapped), with imagery links + tagging instructions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd

log = logging.getLogger(__name__)


def _imagery_links(lon: float, lat: float) -> dict[str, str]:
    return {
        "osm": f"https://www.openstreetmap.org/edit#map=18/{lat:.5f}/{lon:.5f}",
        "bing": f"https://www.bing.com/maps?cp={lat:.5f}~{lon:.5f}&lvl=18&style=a",
        "google": f"https://www.google.com/maps/@{lat:.5f},{lon:.5f},400m/data=!3m1!1e3",
    }


def run_export(aoi: str, pred_dir: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pred_dir = Path(pred_dir) / aoi
    cands = gpd.read_parquet(pred_dir / "candidates.parquet")
    if cands.empty:
        log.warning("No candidates to export for %s", aoi)
        return
    sort_col = "rank_score" if "rank_score" in cands.columns else "confidence"
    cands = cands.sort_values(sort_col, ascending=False).reset_index(drop=True)
    cands["candidate_id"] = [f"{aoi}-sub-{i:06d}" for i in range(len(cands))]

    gpq = pred_dir / f"{aoi}_sub_candidates.geoparquet"
    cands.to_parquet(gpq)
    cands.to_file(pred_dir / f"{aoi}_sub_candidates.geojson", driver="GeoJSON")

    # MapRoulette: newline-delimited FeatureCollections, only for NEW leads.
    new = cands[cands.get("status", "new") == "new"] if "status" in cands.columns else cands
    mr = pred_dir / f"{aoi}_sub_maproulette.geojson"
    with mr.open("w") as f:
        for _, row in new.iterrows():
            c = row.geometry.centroid
            props = {
                "candidate_id": row.candidate_id,
                "confidence": round(float(row.confidence), 3),
                "rank_score": round(float(row.rank_score), 3) if "rank_score" in cands else None,
                "area_m2": round(float(row.area_m2), 1),
                "line_dist_m": round(float(row.line_dist_m), 1) if "line_dist_m" in cands else None,
                "instruction": (
                    f"Possible electrical substation (~{row.area_m2:.0f} m2, "
                    f"confidence {row.confidence:.2f}). Check imagery; if confirmed, map "
                    "power=substation + substation=transmission + voltage=* and connect the "
                    "incoming power=line."
                ),
                **_imagery_links(c.x, c.y),
            }
            fc = {"type": "FeatureCollection",
                  "features": [{"type": "Feature", "geometry": row.geometry.__geo_interface__,
                                "properties": props}]}
            f.write(json.dumps(fc) + "\n")
    log.info("Exported %d candidates (%d new -> MapRoulette) -> %s", len(cands), len(new), gpq.name)
