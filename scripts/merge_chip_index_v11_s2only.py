"""v11 s2only index: v9's combined recipe + BOTH hard-negative sets -- the 299 global
city/water chips (scripts/build_global_hardneg_chips.py, the never-trained v10 addition)
and the 5,533 settlement/construction chips mined from Pakistan+India OSM
(scripts/build_settlement_hardneg_chips.py).

Motivation: field validation of the v9 leads (2026-07-19) found the dominant FP classes
are villages, construction sites and bare land -- confusers the 94.5%-positive v9 index
never shows as labeled background. This index drops the positive share to ~80% with
negatives drawn from exactly those classes, in the deployment regions.

NOTE val/mIoU is NOT comparable to v9 (the val split gains the hard-negative chips'
held-out rows). Compare via scripts/field_eval.py on sindh_test.

Writes data/chips_v11/combined/index.parquet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parts = []
    for name, rel, aoi in [("v9 combined", "data/chips_v9/combined/index.parquet", None),
                           ("global_hardneg", "data/chips_global_hardneg/index.parquet", "global_hardneg"),
                           ("settlement_hardneg", "data/chips_settlement_hardneg/index.parquet", None)]:
        df = pd.read_parquet(ROOT / rel)
        if aoi is not None:
            df = df.assign(aoi=aoi)
        print(f"{name}: {len(df)} chips ({int((df.split == 'val').sum())} val, "
              f"{int((df.sub_pixels > 0).sum())} with substation)")
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    out_path = ROOT / "data" / "chips_v11" / "combined" / "index.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    n_pos = int((out.sub_pixels > 0).sum())
    print(f"combined: {len(out)} chips ({int((out.split == 'val').sum())} val, "
          f"{n_pos} positive [{100*n_pos/len(out):.1f}%]) -> {out_path}")


if __name__ == "__main__":
    main()
