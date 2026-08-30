"""Multivariate separability + the guards that make ~159 sites trustworthy. (Stage 4b)

Consumes `data/eval_results/hydro_detectability_samples.csv` (site and matched-control
feature rows) and answers the question the per-feature AUCs cannot: does *any combination*
of what S2, S1 and terrain can see separate a hydro site from a matched patch of the same
valley, well enough to justify GPU time?

Four guards, each of them a lesson this repo has already paid for:

  GroupKFold on cluster. A site and its own controls must land in the same fold, and so
  must neighbouring sites, or the model memorises a cell's radiometry. Random folds over
  spatially autocorrelated features at n~159 is precisely the small-n false positive trap.

  Bootstrap CI resampling SITES, not rows. At this n the 95% CI on an AUC near 0.75 is
  roughly +/-0.07, which is why the gates below have a deliberate 0.65-0.75 dead band
  rather than a single cut point.

  A label-permutation null, shuffled WITHIN cluster. If the null's 95th percentile clears
  0.65, the gate thresholds are too low for this n and must be raised before the real
  number is read.

  Feature-group ablation. If context/terrain alone is as good as everything together, the
  classifier is detecting access and landform, not hydropower -- and segmentation is the
  wrong product. The EHV study found the mirror image of this (S1+S2 fusion *diluting* a
  clean S1 signal, 0.909 -> 0.835); a fusion number is not self-evidently the right one.

The held-out valley (Swat/Panjkora, `split == "holdout"`) is scored once, at the end.

    pixi run -e ml python scripts/hydro_stage4_model.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
RESULTS = ROOT / "data" / "eval_results"
EVAL_VERSION = 1

log = logging.getLogger("hydro_stage4_model")

S1_PREFIX = ("vv_", "vh_", "n_bright")
TERRAIN = {"elev_m", "slope_deg", "rel_elev_500m", "relief_1km", "northness",
           "d_river_m", "slope_anomaly"}
META = {"run_name", "site_id", "cell", "cluster", "group", "cell_0p5", "size_bucket",
        "split", "is_control",
        "lon", "lat", "area_m2", "core_area_m2", "capacity_mw", "has_penstock",
        "conduit_tunnel_frac", "valid_frac_s2"}


def group_of(col: str) -> str:
    if col in TERRAIN:
        return "context"
    if col.startswith(S1_PREFIX):
        return "s1"
    return "s2"


def fit_auc(X, y, groups, seed=0):
    """Out-of-fold AUC under GroupKFold, standardized features, L2 logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return float("nan"), None, None
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    ok = np.isfinite(oof)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return float("nan"), None, None
    full = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0, random_state=seed)).fit(X, y)
    # `oof` is returned so the confidence interval can be built on OUT-OF-FOLD scores.
    # Bootstrapping the full model's in-sample predictions gives a CI whose lower bound can
    # exceed the CV point estimate -- observed here as ci_lo 0.975 against auc_cv 0.771 --
    # which would wave G2.1 through on a number that never left the training set.
    return float(roc_auc_score(y[ok], oof[ok])), full, oof


def site_bootstrap(y, score, site_ids, n=1000, seed=0):
    """95% CI resampling whole sites, so a site and its controls move together."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    sites = np.unique(site_ids)
    out = []
    for _ in range(n):
        pick = rng.choice(sites, sites.size, replace=True)
        idx = np.concatenate([np.flatnonzero(site_ids == s) for s in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], score[idx]))
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, default=RESULTS / "hydro_detectability_samples.csv")
    ap.add_argument("--run-name", default="north_pk_v1")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_csv(a.samples)
    # The named CLUSTERS boxes cover only ~60% of sites, and dropping the rest cost 34 of
    # 82 matched sites. The group only has to capture spatial autocorrelation so a site and
    # its neighbours share a fold, so fall back to the 0.5-degree cell where no named
    # cluster applies. This is a power fix, not a threshold change.
    df = df.copy()
    grp = df.cluster.fillna("").astype(str)
    if "cell_0p5" in df.columns:
        grp = grp.where(grp != "", "c" + df.cell_0p5.astype(str))
    else:
        grp = grp.where(grp != "", "c" + df.cell.astype(str).str[:4])
    df["group"] = grp
    df = df[df.group != ""]
    feat_cols = [c for c in df.columns if c not in META and df[c].dtype.kind in "fi"]
    df = df.dropna(subset=feat_cols, how="any")
    if df.empty:
        raise SystemExit("no complete feature rows")

    study = df[df.split != "holdout"]
    hold = df[df.split == "holdout"]
    y = study.is_control.values == False  # noqa: E712
    y = y.astype(int)
    groups = study.group.values
    log.info("study rows %d (%d sites, %d groups), holdout rows %d (%d sites), %d features",
             len(study), study[~study.is_control].site_id.nunique(), study.group.nunique(),
             len(hold), hold[~hold.is_control].site_id.nunique(), len(feat_cols))

    seeds = [int(s) for s in a.seeds.split(",")]
    rows = []
    for gname, cols in [("all", feat_cols),
                        ("s2", [c for c in feat_cols if group_of(c) == "s2"]),
                        ("s1", [c for c in feat_cols if group_of(c) == "s1"]),
                        ("context", [c for c in feat_cols if group_of(c) == "context"])]:
        if not cols:
            continue
        aucs = []
        for sd in seeds:
            auc, _m, _o = fit_auc(study[cols].values, y, groups, seed=sd)
            aucs.append(auc)
        auc = float(np.nanmean(aucs))
        spread = float(np.nanmax(aucs) - np.nanmin(aucs)) if len(aucs) > 1 else 0.0
        _, model, oof = fit_auc(study[cols].values, y, groups, seed=seeds[0])
        lo = hi = h_auc = float("nan")
        if model is not None:
            ok = np.isfinite(oof)
            lo, hi = site_bootstrap(y[ok], oof[ok], study.site_id.values[ok], seed=seeds[0])
            if len(hold) and hold.is_control.nunique() > 1:
                from sklearn.metrics import roc_auc_score
                hy = (hold.is_control.values == False).astype(int)  # noqa: E712
                h_auc = float(roc_auc_score(hy, model.predict_proba(hold[cols].values)[:, 1]))
        rows.append({"group": gname, "n_features": len(cols), "auc_cv": auc,
                     "auc_seed_spread": spread, "ci_lo": lo, "ci_hi": hi,
                     "auc_holdout": h_auc})
    res = pd.DataFrame(rows)

    # Permutation null, shuffled within cluster.
    rng = np.random.default_rng(0)
    null = []
    for _ in range(a.n_perm):
        yp = y.copy()
        for g in np.unique(groups):
            m = groups == g
            yp[m] = rng.permutation(yp[m])
        auc, _m, _o = fit_auc(study[feat_cols].values, yp, groups, seed=0)
        if np.isfinite(auc):
            null.append(auc)
    null = np.array(null)
    null95 = float(np.percentile(null, 95)) if null.size else float("nan")
    all_auc = float(res.loc[res.group == "all", "auc_cv"].iloc[0])
    pval = float((null >= all_auc).mean()) if null.size else float("nan")

    print("\n=== Stage 4b: multivariate separability ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\npermutation null (within-cluster shuffle, n={null.size}): "
          f"median {np.median(null):.3f}, 95th pct {null95:.3f}, p={pval:.3f}")

    ctx = float(res.loc[res.group == "context", "auc_cv"].iloc[0]) if (res.group == "context").any() else np.nan
    hold_auc = float(res.loc[res.group == "all", "auc_holdout"].iloc[0])
    spread = float(res.loc[res.group == "all", "auc_seed_spread"].iloc[0])
    ci_lo = float(res.loc[res.group == "all", "ci_lo"].iloc[0])

    print("\n--- gates ---")
    if null.size and null95 > 0.65:
        print(f"  FAIL  null too high ({null95:.3f} > 0.65): thresholds are invalid at this n. "
              "Raise them before reading the real number.")
    g21 = all_auc >= 0.75 and ci_lo >= 0.68 and hold_auc >= 0.70
    g22 = all_auc < 0.65 or hold_auc < 0.60 or ci_lo <= 0.55
    g24 = np.isfinite(ctx) and ctx >= all_auc - 0.03
    print(f"  G2.1 PASS   cv {all_auc:.3f}>=0.75, ci_lo {ci_lo:.3f}>=0.68, "
          f"holdout {hold_auc:.3f}>=0.70  -> {'YES' if g21 else 'no'}")
    print(f"  G2.2 FAIL   cv<0.65 or holdout<0.60 or ci_lo<=0.55        -> {'YES' if g22 else 'no'}")
    print(f"  G2.4 road?  context-only {ctx:.3f} vs all {all_auc:.3f}    -> "
          f"{'CONTEXT DOMINATES: the product is a prior, not a U-Net' if g24 else 'no'}")
    if spread > 0.03:
        print(f"  !! seed spread {spread:.3f} > 0.03 -- report the mean and widen the CI")
    verdict = ("PASS -> Stage 5 (build chips, fine-tune)" if g21 and not g24 else
               "FAIL -> do not spend GPU; write up the negative result" if g22 else
               "AMBIGUOUS -> one pre-declared extension only (multi-date temporal-max S1)")
    print(f"\nVERDICT: {verdict}")

    # Size stratification is mandatory reporting: this lineage's central lesson is that
    # everything turns out to be a size story.
    if "core_area_m2" in study.columns:
        print("\n--- AUC by powerhouse size bucket (reporting, not a gate) ---")
        for b, grp in study.groupby("size_bucket"):
            sid = grp[~grp.is_control].site_id.nunique()
            if sid < 5:
                continue
            auc, _m, _o = fit_auc(grp[feat_cols].values,
                             (grp.is_control.values == False).astype(int),  # noqa: E712
                             grp.group.values, seed=0)
            print(f"  {b:>10}  n_sites={sid:3d}  auc={auc:.3f}")

    res.insert(0, "run_name", a.run_name)
    res["eval_version"] = EVAL_VERSION
    res["null_p95"] = null95
    res["perm_p"] = pval
    res["generated"] = dt.datetime.now().isoformat(timespec="seconds")
    out = RESULTS / "hydro_stage2_summary.csv"
    if out.exists():
        prev = pd.read_csv(out)
        prev = prev[~((prev.run_name == a.run_name) & (prev.eval_version == EVAL_VERSION))]
        res = pd.concat([prev, res], ignore_index=True)
    res.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
