"""v6c index: same recipe as v6b (refined-label india_pilot + pakistan:2 + pakistan_hardneg,
no yunnan_hardneg) but on chips_v6c, built after adding the building-density guard to
label_refine.py. Tests whether fixing the guard (which caught 5 Pakistan + 1 India
village/building-clutter mislabels the earlier v6b refinement missed) closes v6b's
remaining mild regression vs v5. Writes data/chips_v6c/combined/index.parquet.
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
        _load(ROOT / "data" / "chips_v6c" / "india_pilot" / "index.parquet", "india_pilot"),
        _load(ROOT / "data" / "chips_v6c" / "pakistan" / "index.parquet", "pakistan", repeat=2),
        _load(ROOT / "data" / "chips" / "pakistan_hardneg" / "index.parquet", "pakistan_hardneg"),
    ]
    out = pd.concat(frames, ignore_index=True)
    out_path = ROOT / "data" / "chips_v6c" / "combined" / "index.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"combined: {len(out)} chips -> {out_path}")


if __name__ == "__main__":
    main()
