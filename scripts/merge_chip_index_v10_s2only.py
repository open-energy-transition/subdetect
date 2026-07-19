"""v10 s2only index: v9's combined recipe (v5 combined + TorchGeo global substation
chips -- see merge_chip_index_v9_s2only.py) plus the global hard-negative chips mined
from well-mapped cities/water bodies (scripts/build_global_hardneg_chips.py).

v9 alone is 94.5% positive chips (29,548/31,252) because every TorchGeo location was
curated around a real substation with zero accompanying hard negatives -- see memory
subdetect-improvement-levers and docs/expanding-training-data.md. This index adds 299
Overpass-verified clean negatives (53 locations: cities' parks/rivers/residential
fringes + large open water) to partially rebalance that skew. Still S2-only only --
the hard-negative chips have no Sentinel-1 companion either.

Writes data/chips_v10/combined/index.parquet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    v9 = pd.read_parquet(ROOT / "data" / "chips_v9" / "combined" / "index.parquet")
    print(f"v9 combined: {len(v9)} chips ({int((v9.split == 'val').sum())} val, "
          f"{int((v9.sub_pixels > 0).sum())} with substation)")

    hardneg = pd.read_parquet(ROOT / "data" / "chips_global_hardneg" / "index.parquet")
    hardneg = hardneg.assign(aoi="global_hardneg")
    print(f"global_hardneg: {len(hardneg)} chips ({int((hardneg.split == 'val').sum())} val, "
          f"{int((hardneg.sub_pixels > 0).sum())} with substation)")

    out = pd.concat([v9, hardneg], ignore_index=True)
    out_path = ROOT / "data" / "chips_v10" / "combined" / "index.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    n_pos = int((out.sub_pixels > 0).sum())
    print(f"combined: {len(out)} chips ({int((out.split == 'val').sum())} val, "
          f"{n_pos} positive [{100*n_pos/len(out):.1f}%]) -> {out_path}")


if __name__ == "__main__":
    main()
