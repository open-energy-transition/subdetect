"""Report substation/label/ROI statistics for an AOI so ROI radius and the val block
can be tuned before the (network-bound) compositing starts.

Usage: python scripts/report_cells.py <aoi> [radius_km]
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd

from subdetect.config import Settings, resolve_aoi
from subdetect.roi import _snapped_origin, roi_cells, training_cells

ROOT = Path(__file__).resolve().parents[1]


def _in_val(g, vb) -> bool:
    c = g.representative_point()
    return vb and (vb[0] <= c.x <= vb[2]) and (vb[1] <= c.y <= vb[3])


def main(aoi: str, radius_km: float = 20.0) -> None:
    settings = Settings.load()
    _, cfg = resolve_aoi(aoi, settings)
    ld = Path("data/labels") / aoi
    subs = gpd.read_parquet(ld / "substations_poly.parquet")
    big = subs[subs.area_m2 >= settings.min_sub_area_m2]
    print(f"== {aoi} ==")
    print(f"substation polygons: {len(subs)} total, {len(big)} >= {settings.min_sub_area_m2:.0f} m2")
    print(f"  area quantiles (m2): {subs.area_m2.quantile([.5, .9, .99]).round(0).to_dict()}")
    for p in ("substations_node", "plants", "lines"):
        fp = ld / f"{p}.parquet"
        if fp.exists():
            print(f"{p}: {len(gpd.read_parquet(fp))}")

    vb = cfg.get("val_bbox")
    if vb is not None and len(big):
        n_val = int(big.geometry.apply(lambda g: _in_val(g, vb)).sum())
        print(f"val_bbox {vb}: {n_val}/{len(big)} positives ({100*n_val/max(len(big),1):.1f}%)")

    if cfg.get("max_train_cells"):
        cells = training_cells(aoi, cfg, settings)
    else:
        cells = roi_cells(aoi, cfg, settings, radius_km)
    print(f"cells to composite: {len(cells)} ({int(cells.has_label.sum())} contain substations)")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 20.0)
