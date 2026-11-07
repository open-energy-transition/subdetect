"""The v18 verdicts (docs/issues/v18-improvements.md), read from data/eval_results/field_eval.csv.

For each comparison and region: fused AUC, P@50, hits and recall by bucket, d = arm - control,
and the pre-registered rule's verdict. Controls with several seeds (the v17 runs) are averaged.
sindh_test rows of production and v17 come from their historical run names; gujarat_test rows
from the v18 baseline step (run names v18_gujarat_*). Missing rows are reported as pending.

Usage: pixi run python scripts/v18_ab.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/eval_results/field_eval.csv"
REGIONS = ("sindh", "gujarat")
# (sindh run name, gujarat run name) per model; a list is a seed set (averaged).
FUSED = {
    "production": ("v12_up2_mean_fusion", "v18_gujarat_v12_x_v5"),
    "datemax": ("v18_sindh_v12dmax_x_v5", "v18_gujarat_v12dmax_x_v5"),
    "v17_s2_ctl": ([f"v17_s2_ctl_seed{s}_fused" for s in (0, 1)],
                   [f"v18_gujarat_v17s2c{s}_x_v5" for s in (0, 1)]),
    "v17_s1_ctl": ([f"v17_s1_ctl_seed{s}_fused" for s in (0, 1, 2)],
                   [f"v18_gujarat_v17s1c{s}_x_v12" for s in (0, 1, 2)]),
}
for arm, partner in [("s2_recipe", "v5"), ("s1_recipe", "v12"), ("s2_1d", "v5"),
                     ("s2_4d", "v5"), ("s1_big", "v12")]:
    FUSED[arm] = tuple(f"v18_{r}_{arm}_x_{partner}" for r in REGIONS)
METRICS = ["auc", "p_at_50", "n_hits", "recall_2000-5000", "recall_5000-20000", "recall_20000-inf"]
RULES = {  # comparison -> (arm, control, kind)
    "datemax inference": ("datemax", "production", "adopt"),
    "S2 recipe": ("s2_recipe", "v17_s2_ctl", "noninferior"),
    "S1 recipe": ("s1_recipe", "v17_s1_ctl", "noninferior"),
    "four dates": ("s2_4d", "s2_1d", "passfail"),
    "S1 data": ("s1_big", "s1_recipe", "passfail"),
}


def row(df: pd.DataFrame, names) -> pd.Series | None:
    names = [names] if isinstance(names, str) else names
    sel = df[(df.eval_version == 2) & df.run_name.isin(names)].drop_duplicates("run_name", keep="last")
    if len(sel) < len(names):
        return None
    return sel[METRICS].mean()


def main() -> None:
    df = pd.read_csv(CSV)
    for title, (arm, ctl, kind) in RULES.items():
        print(f"\n== {title}: {arm} vs {ctl}")
        ds, p50 = {}, {}
        for i, region in enumerate(REGIONS):
            a, c = row(df, FUSED[arm][i]), row(df, FUSED[ctl][i])
            if a is None or c is None:
                print(f"  {region}: pending")
                continue
            ds[region], p50[region] = a.auc - c.auc, a.p_at_50 - c.p_at_50
            print(f"  {region}: AUC {c.auc:.3f} -> {a.auc:.3f} (d {ds[region]:+.3f}), "
                  f"P@50 {c.p_at_50:.2f} -> {a.p_at_50:.2f}, hits {c.n_hits:.1f} -> {a.n_hits:.0f}, "
                  "recall 2-5k/5-20k/20k+ "
                  + "/".join(f"{c[m]:.2f}->{a[m]:.2f}" for m in METRICS[3:]))
        if len(ds) < 2:
            print("  verdict: pending")
            continue
        if kind == "adopt":
            v = ("ADOPT" if all(d >= 0.02 for d in ds.values()) and all(p >= 0 for p in p50.values())
                 else "NOT ADOPTED")
        elif kind == "noninferior":
            v = "ADOPTED (non-inferior)" if all(d > -0.02 for d in ds.values()) else "REJECTED"
        else:
            if all(d >= 0.02 for d in ds.values()) and all(p >= 0 for p in p50.values()):
                v = "PASS (needs a second seed pair before promotion)"
            elif all(d <= -0.02 for d in ds.values()):
                v = "FAIL"
            else:
                v = "NEUTRAL"
        print(f"  verdict: {v}")


if __name__ == "__main__":
    main()
