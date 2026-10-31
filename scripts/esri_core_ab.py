"""Paired A/B verdict for the ESRI core-label study (docs/issues/esri-core-labels.md).

Reads field_eval.csv rows (eval_version 2) for `<prefix>_ctl_seed{s}` / `<prefix>_core_seed{s}`
run names, computes per-seed paired differences, and applies the pre-registered rule:

  PASS     mean d >= +0.02 AND every d_s > 0 AND mean P@50 difference >= 0
  FAIL     mean d <= -0.02 AND every d_s < 0
  NEUTRAL  otherwise

on the primary metric (AUC of the deployed mean fusion, run names suffixed `_fused`), with
arm-alone rows and secondary metrics reported alongside. Also computes the secondary
localisation metric: of all pixels a run predicts at P >= 0.5, the share whose centre lies
inside a raw OSM substation polygon.

Usage:
  pixi run -e ml python scripts/esri_core_ab.py --prefix v17_s1 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/eval_results/field_eval.csv"
REG = ROOT / "data/osmose_regions"
METRICS = ["auc", "p_at_20", "p_at_50", "n_hits", "n_candidates", "n_fp_proxy",
           "recall_2000-5000", "recall_5000-20000", "recall_20000-inf"]


def inside_share(region: str, polys: gpd.GeoDataFrame, thr: int = 128) -> float:
    hit = tot = 0
    for f in sorted((REG / region / "prob").glob("*.tif")):
        with rasterio.open(f) as src:
            p = src.read(1)
            if (p >= thr).sum() == 0:
                continue
            g = polys.to_crs(src.crs)
            g = g[g.intersects(box(*src.bounds))]
            ins = features.rasterize([(x, 1) for x in g.geometry], out_shape=p.shape,
                                     transform=src.transform, fill=0, dtype="uint8") \
                if len(g) else np.zeros_like(p)
        sel = p >= thr
        hit += int((ins[sel] == 1).sum())
        tot += int(sel.sum())
    return hit / tot if tot else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="e.g. v17_s1")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--labels", default="pakistan")
    a = ap.parse_args()

    df = pd.read_csv(CSV)
    df = df[df.eval_version == 2].drop_duplicates("run_name", keep="last").set_index("run_name")
    polys = gpd.read_parquet(ROOT / "data/labels" / a.labels / "substations_poly.parquet")
    rows = []
    for kind in ("alone", "fused"):
        for s in a.seeds:
            for arm in ("ctl", "core"):
                rn = f"{a.prefix}_{arm}_seed{s}" + ("_fused" if kind == "fused" else "")
                if rn not in df.index:
                    print(f"missing field_eval row: {rn}")
                    continue
                r = df.loc[rn]
                rows.append(dict(kind=kind, seed=s, arm=arm, run=rn, region=r.region,
                                 inside=inside_share(r.region, polys),
                                 **{m: float(r[m]) for m in METRICS}))
    t = pd.DataFrame(rows)
    if t.empty:
        return
    pd.set_option("display.width", 200)
    print(t.drop(columns=["run"]).round(3).to_string(index=False))

    for kind in ("alone", "fused"):
        w = t[t.kind == kind].pivot(index="seed", columns="arm")
        if not {"ctl", "core"} <= set(t[t.kind == kind].arm):
            continue
        d = pd.DataFrame({m: w[m]["core"] - w[m]["ctl"] for m in METRICS + ["inside"]})
        print(f"\n[{kind}] paired differences core - ctl:")
        print(d.round(3).to_string())
        print("mean:", d.mean().round(3).to_dict())
        if kind == "fused":
            dauc = d["auc"]
            if dauc.mean() >= 0.02 and (dauc > 0).all() and d["p_at_50"].mean() >= 0:
                verdict = "PASS"
            elif dauc.mean() <= -0.02 and (dauc < 0).all():
                verdict = "FAIL"
            else:
                verdict = "NEUTRAL"
            print(f"\nVERDICT ({a.prefix}, primary = fused AUC, n_seeds={len(dauc)}): {verdict}")


if __name__ == "__main__":
    main()
