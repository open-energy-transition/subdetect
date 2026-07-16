"""Merge the v6 chip indexes: refined-label india_pilot + pakistan (chips_v6) plus the
existing hard-negative sets (chips/pakistan_hardneg, chips/yunnan_hardneg -- background-only,
unaffected by label refinement, reused as-is). Mirrors merge_chip_index.py's india_pilot +
pakistan:2 oversampling; writes data/chips_v6/combined/index.parquet.
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
        _load(ROOT / "data" / "chips_v6" / "india_pilot" / "index.parquet", "india_pilot"),
        _load(ROOT / "data" / "chips_v6" / "pakistan" / "index.parquet", "pakistan", repeat=2),
        _load(ROOT / "data" / "chips" / "pakistan_hardneg" / "index.parquet", "pakistan_hardneg"),
        _load(ROOT / "data" / "chips" / "yunnan_hardneg" / "index.parquet", "yunnan_hardneg"),
    ]
    out = pd.concat(frames, ignore_index=True)
    out_path = ROOT / "data" / "chips_v6" / "combined" / "index.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"combined: {len(out)} chips -> {out_path}")


if __name__ == "__main__":
    main()
