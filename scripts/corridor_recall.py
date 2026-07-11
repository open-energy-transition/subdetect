"""Corridor-level recall of the exported candidates against OSM substations.

Deployment metric (vs the strict chip-based per-installation recall in evaluate.py): over
the seamless full-inference candidate set, an OSM substation counts as recalled if any
candidate polygon lands within MATCH_M of it. Reports recall in the held-out val_bbox
(honest, geographic holdout) and over the whole inferred area (optimistic, includes
training cells), broken down by area and voltage. Also reports candidate precision proxy:
share of candidates that match a known OSM substation.

Usage: python scripts/corridor_recall.py [aoi=pakistan] [match_m=60]
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

from subdetect.config import Settings, resolve_aoi
from subdetect.local_source import load_substation_labels

EQ_AREA = "EPSG:6933"
AREA_BUCKETS = [(1000, 2000), (2000, 5000), (5000, 20000), (20000, np.inf)]
VOLT_BUCKETS = [(">=220kV", 220000, np.inf), ("66-220kV", 66000, 220000),
                ("<66kV/unknown", 0, 66000)]


def _recall_table(subs: gpd.GeoDataFrame, cand_union_m, match_m: float) -> str:
    sm = subs.to_crs(EQ_AREA)
    dist = sm.geometry.distance(cand_union_m) if cand_union_m is not None else \
        np.full(len(sm), np.inf)
    hit = np.asarray(dist) <= match_m
    lines = []
    for lo, hi in AREA_BUCKETS:
        sel = (subs.area_m2 >= lo) & (subs.area_m2 < hi)
        n = int(sel.sum())
        lines.append(f"  {lo}-{'inf' if hi == np.inf else int(hi)} m2: "
                     f"{int(hit[sel.values].sum())}/{n} = "
                     f"{hit[sel.values].mean():.3f}" if n else
                     f"  {lo}-{'inf' if hi == np.inf else int(hi)} m2: 0/0")
    for name, lo, hi in VOLT_BUCKETS:
        v = subs.voltage_v
        sel = ((v >= lo) & (v < hi)) if lo > 0 else (v.isna() | (v < hi))
        n = int(sel.sum())
        if n:
            lines.append(f"  {name}: {int(hit[sel.values].sum())}/{n} = "
                         f"{hit[sel.values].mean():.3f}")
    lines.append(f"  OVERALL: {int(hit.sum())}/{len(subs)} = "
                 f"{hit.mean():.3f}" if len(subs) else "  OVERALL: 0/0")
    return "\n".join(lines)


def main(aoi: str = "pakistan", match_m: float = 60.0) -> None:
    settings = Settings.load()
    _, cfg = resolve_aoi(aoi, settings)
    cand_p = Path("data/predictions") / aoi / "candidates.parquet"
    cands = gpd.read_parquet(cand_p)
    print(f"candidates: {len(cands)}", cands.status.value_counts().to_dict()
          if "status" in cands.columns and len(cands) else "")
    cand_union_m = cands.to_crs(EQ_AREA).union_all() if len(cands) else None

    subs = load_substation_labels(Path("data/labels") / aoi, settings.min_sub_area_m2)
    subs = subs[subs.role == "pos"].reset_index(drop=True)

    # Restrict to the inferred footprint (only score substations we actually imaged).
    from subdetect.local_source import CompositeIndex

    cov = CompositeIndex(Path("data/composites") / aoi).coverage
    subs = subs[subs.geometry.centroid.within(cov)].reset_index(drop=True)
    print(f"OSM substations in inferred coverage: {len(subs)} (match distance {match_m:.0f} m)")

    vb = cfg.get("val_bbox")
    if vb is not None:
        c = subs.geometry.centroid
        held = subs[(c.x >= vb[0]) & (c.x <= vb[2]) & (c.y >= vb[1]) & (c.y <= vb[3])]
        print(f"\n== HELD-OUT val_bbox recall ({len(held)} substations) ==")
        print(_recall_table(held.reset_index(drop=True), cand_union_m, match_m))
    print(f"\n== ALL inferred-area recall ({len(subs)} substations, includes train) ==")
    print(_recall_table(subs, cand_union_m, match_m))

    if len(cands):
        known = (cands.status == "known").sum() if "status" in cands.columns else 0
        print(f"\ncandidates matching a known OSM substation: {known}/{len(cands)} "
              f"({known/len(cands):.2f}); remaining {len(cands)-known} are new leads")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pakistan",
         float(sys.argv[2]) if len(sys.argv) > 2 else 60.0)
