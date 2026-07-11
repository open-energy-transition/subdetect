"""Hardlink earthpv's already-composited Pakistan S2 cells into subdetect.

earthpv composited ~1360 Pakistan cells on the SAME 0.1 deg grid (shared grid_origin),
so a cell named 0037_0036 is the identical footprint in both projects. Hardlinking
(same filesystem) reuses them at zero byte and zero download cost. Only cells in
subdetect's ROI are linked. Idempotent; never writes into earthpv's tree.

Run after `subdetect roi --aoi pakistan`. Safe to re-run once earthpv's compose job
adds more cells.

Usage: python scripts/link_pakistan_composites.py [aoi=pakistan]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main(aoi: str = "pakistan") -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "aoi.yaml").read_text())
    src_root = Path(cfg["earthpv_composites"]) / aoi / "composites"
    dst_root = ROOT / "data" / "composites" / aoi / "composites"
    dst_root.mkdir(parents=True, exist_ok=True)

    roi = ROOT / "data" / "roi" / f"{aoi}_cells.parquet"
    if roi.exists():
        cells = set(pd.read_parquet(roi)["name"].tolist())
        print(f"ROI restricts to {len(cells)} cells")
    else:
        cells = None
        print("No ROI file; linking every earthpv cell (run `subdetect roi` to restrict)")

    if not src_root.exists():
        raise FileNotFoundError(f"earthpv composites not found at {src_root}")

    linked = skipped = missing = 0
    for cell_dir in sorted(src_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        name = cell_dir.name
        if cells is not None and name not in cells:
            continue
        src = cell_dir / "composite_0.tif"
        if not src.exists():
            missing += 1
            continue
        dst_dir = dst_root / name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "composite_0.tif"
        if dst.exists():
            skipped += 1
            continue
        try:
            os.link(src, dst)
            linked += 1
        except OSError as e:
            print(f"  link failed for {name}: {e}")
    print(f"linked {linked}, already present {skipped}, source missing composite_0 {missing}")
    if cells is not None:
        have = {d.name for d in dst_root.iterdir() if (d / "composite_0.tif").exists()}
        print(f"ROI cells still needing S2 compose: {len(cells - have)} of {len(cells)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pakistan")
