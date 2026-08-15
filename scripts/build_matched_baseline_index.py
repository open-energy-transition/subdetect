"""Matched single-image baseline index: same chip_ids as the multidate build (only
those that succeeded), original chips_v5 median-composite image path. Needed for a
clean ablation -- comparing v14_multitemporal against full-scale v5/v9 numbers would
confound the multi-date variable with an 8-50x dataset-size difference.

Usage:
  pixi run -e ml python scripts/build_matched_baseline_index.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE_INDEX = ROOT / "data/chips_v5/combined/index.parquet"
MULTIDATE_INDEX = ROOT / "data/chips_multidate/pakistan_india/index.parquet"
OUT = ROOT / "data/chips_multidate/pakistan_india/index_baseline_matched.parquet"


def main() -> None:
    base = pd.read_parquet(BASE_INDEX)
    multidate = pd.read_parquet(MULTIDATE_INDEX)
    # chips_v5/combined intentionally duplicates some chip_ids (pakistan:2 oversampling,
    # see subdetect-tower-s1-priors memory); the multidate stratified sample picked up
    # 8 of these as independent draws, so multidate itself has 8 duplicated chip_ids
    # (same file, harmless -- mirrors the original oversampling). Left-merge on
    # chip_id against a deduped base so row COUNT and duplicate PATTERN match exactly
    # (same 318 rows, same 8 chip_ids repeated), just with the single-image path.
    matched = multidate[["chip_id"]].merge(
        base.drop_duplicates(subset="chip_id"), on="chip_id", how="left"
    )
    assert len(matched) == len(multidate), (len(matched), len(multidate))
    assert matched.image.notna().all(), "some multidate chip_ids missing from base index"
    matched.to_parquet(OUT)
    print(f"wrote {OUT} ({len(matched)} chips, same subset as {MULTIDATE_INDEX.name})")


if __name__ == "__main__":
    main()
