"""v9 s2only index: v5's exact combined recipe (india_pilot + pakistan:2 +
pakistan_hardneg, raw/unrefined labels, no size floor) plus the TransitionZero/TorchGeo
global Substation dataset (see scripts/build_chips_from_substation_ds.py) -- 26k+
S2-only chips with no local-region bias, targeting the data-starved S2-only arm
specifically (v5_s2only val IoU_1 0.111 vs v5_s1only 0.212; see memory
subdetect-improvement-levers). No Sentinel-1 companion for the new chips, so this index
is only valid for an S2-only (`modalities: [S2L2A]`) training config -- s1fusion/s1only
must keep using data/chips_v5/combined.

Writes data/chips_v9/combined/index.parquet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    v5 = pd.read_parquet(ROOT / "data" / "chips_v5" / "combined" / "index.parquet")
    print(f"v5 combined: {len(v5)} chips ({int((v5.split == 'val').sum())} val, "
          f"{int((v5.sub_pixels > 0).sum())} with substation)")

    tg = pd.read_parquet(ROOT / "data" / "chips_torchgeo" / "substation_global" / "index.parquet")
    tg = tg.assign(aoi="torchgeo_substation")
    print(f"torchgeo_substation: {len(tg)} chips ({int((tg.split == 'val').sum())} val, "
          f"{int((tg.sub_pixels > 0).sum())} with substation)")

    out = pd.concat([v5, tg], ignore_index=True)
    out_path = ROOT / "data" / "chips_v9" / "combined" / "index.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"combined: {len(out)} chips ({int((out.split == 'val').sum())} val) -> {out_path}")


if __name__ == "__main__":
    main()
