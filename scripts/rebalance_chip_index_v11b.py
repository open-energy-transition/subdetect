"""v11b: fix the v11 regression by rebalancing negative-chip count, not removing them.

v11_s2only (2026-07-20 field_eval) collapsed: sindh_test candidates 201->48, hits
22->4, mean per-cell max probability 0.772->0.435, cells with any confident
detection 87%->57% -- a global suppression, confirmed real (not an inference bug:
raw prob stats collapsed identically for the arm alone and its fusion).

Root cause: v9's index carried 1,670 negative-only chips against 29,582 positive
(94.6% positive). v11 added 5,832 settlement/global hard negatives on top, taking
total negatives to 7,502 -- a 4.5x jump -- while class_weights=[0.25,0.75] and the
Tversky alpha/beta (tuned for the OLD ~95%-positive balance) carried over
unchanged. The model over-learned "predict background".

Fix: keep the same source chips (so field-validated FP classes -- villages,
residential, construction -- stay represented) but cap the added hard-negative
chips at 2x v9's original negative count (~3,340) via proportional sampling
across the four sources (global city/water, settlement village/residential/
construction), instead of dropping any set entirely. Loss config (class_weights,
tversky alpha/beta) is unchanged in the v11b training config -- deliberately one
variable (negative count) so this isolates whether ratio, not the negative
classes themselves, was the problem.

Usage:
  pixi run -e ml python scripts/rebalance_chip_index_v11b.py [--negative-cap 3340]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negative-cap", type=int, default=3340,
                    help="~2x v9's original 1,670 negative-only chips")
    a = ap.parse_args()
    rng = np.random.default_rng(11)

    v11 = pd.read_parquet(ROOT / "data/chips_v11/combined/index.parquet")
    pos = v11[v11.sub_pixels > 0]
    neg = v11[v11.sub_pixels == 0]
    print(f"v11: {len(v11)} chips ({len(pos)} positive, {len(neg)} negative-only)")

    if len(neg) <= a.negative_cap:
        neg_kept = neg
    else:
        # proportional cap per source (aoi/tile group) so no negative class vanishes
        frac = a.negative_cap / len(neg)
        parts = []
        for key, grp in neg.groupby("aoi"):
            n = max(1, round(len(grp) * frac))
            parts.append(grp.sample(min(n, len(grp)), random_state=int(rng.integers(1 << 31))))
        neg_kept = pd.concat(parts, ignore_index=True)

    out = pd.concat([pos, neg_kept], ignore_index=True)
    out_path = ROOT / "data/chips_v11b/combined/index.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    n_pos = int((out.sub_pixels > 0).sum())
    print(f"v11b: {len(out)} chips ({n_pos} positive [{100*n_pos/len(out):.1f}%], "
          f"{len(out)-n_pos} negative) -> {out_path}")
    print("negative source breakdown:",
          neg_kept.aoi.value_counts().to_dict())


if __name__ == "__main__":
    main()
