"""v7 index: v5's exact recipe (india_pilot + pakistan:2 + pakistan_hardneg, no
yunnan_hardneg, raw/unrefined labels) but chips built with --voltage-only, so only
substations with a valid OSM voltage tag train as positive. Isolates "does restricting
to voltage-tagged substations improve quality" as a single-variable test against v5.
Writes data/chips_v7/combined/index.parquet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, aoi: str, repeat: int = 1) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["aoi"] = aoi
    if repeat > 1:
        train = df[df.split == "train"]
        df = pd.concat([df] + [train] * (repeat - 1), ignore_index=True)
    print(f"{aoi} (x{repeat} train): {len(df)} chips ({int((df.split == 'val').sum())} val, "
          f"{int((df.sub_pixels > 0).sum())} with substation)")
    return df


def main() -> None:
    frames = [
        _load(ROOT / "data" / "chips_v7" / "india_pilot" / "index.parquet", "india_pilot"),
        _load(ROOT / "data" / "chips_v7" / "pakistan" / "index.parquet", "pakistan", repeat=2),
        _load(ROOT / "data" / "chips" / "pakistan_hardneg" / "index.parquet", "pakistan_hardneg"),
    ]
    out = pd.concat(frames, ignore_index=True)
    out_path = ROOT / "data" / "chips_v7" / "combined" / "index.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"combined: {len(out)} chips -> {out_path}")


if __name__ == "__main__":
    main()
