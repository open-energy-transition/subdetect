"""Per-candidate OSM building-fill and industrial-landuse overlap (streaming).

Substations are open-air: building footprint fill inside the candidate polygon is
low. Factories/warehouses ARE buildings: fill is high. Streams the osmium-exported
geojsonseq once, intersecting against an STRtree of candidate polygons.

Usage: pixi run python scripts/building_fill.py <candidates(_s1).parquet> <buildings.geojsonseq>
"""
import sys, json
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from shapely.strtree import STRtree
from shapely.errors import GEOSException

cand_path, bpath = Path(sys.argv[1]), Path(sys.argv[2])
cands = gpd.read_parquet(cand_path).reset_index(drop=True)
geoms = list(cands.geometry.values)
tree = STRtree(geoms)

build_area = np.zeros(len(cands))   # m2-ish (deg2 scaled later via candidate area ratio)
indus_hit = np.zeros(len(cands), dtype=bool)

INDUS = {"industrial", "commercial", "retail", "construction", "railway"}
n_feat = 0
with open(bpath) as f:
    for line in f:
        line = line.strip().lstrip("\x1e")
        if not line:
            continue
        try:
            feat = json.loads(line)
            g = shape(feat["geometry"])
        except Exception:
            continue
        n_feat += 1
        props = feat.get("properties", {})
        is_indus = props.get("landuse") in INDUS or props.get("aeroway") is not None
        is_building = props.get("building") is not None
        if not (is_indus or is_building):
            continue
        for idx in tree.query(g):
            try:
                inter = geoms[idx].intersection(g)
            except GEOSException:
                continue
            if inter.is_empty:
                continue
            if is_building:
                build_area[idx] += inter.area
            if is_indus:
                indus_hit[idx] = True
        if n_feat % 500000 == 0:
            print(f"{n_feat} features streamed", flush=True)

cand_deg_area = np.array([g.area for g in geoms])
cands["building_fill"] = np.where(cand_deg_area > 0, build_area / cand_deg_area, 0).round(3)
cands["industrial_overlap"] = indus_hit
out = cand_path.parent / (cand_path.stem + "_bld.parquet")
cands.to_parquet(out)
print(f"wrote {out} ({n_feat} features streamed)")

kn = cands[cands.status == "known"]
new_top = cands[cands.status == "new"].sort_values(
    "rank_score_s1" if "rank_score_s1" in cands else "rank_score", ascending=False).head(150)
print("\nbuilding_fill: known median %.2f | top150-new median %.2f" %
      (kn.building_fill.median(), new_top.building_fill.median()))
print("industrial_overlap: known %.0f%% | top150-new %.0f%%" %
      (100*kn.industrial_overlap.mean(), 100*new_top.industrial_overlap.mean()))
