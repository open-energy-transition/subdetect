"""Multivariate separability for BESS + the guards that make ~259 sites trustworthy. (2b)

Consumes `bess_detectability_samples_tier{1,2}.csv` and answers what the per-feature AUCs
cannot: does *any combination* of what S2 and S1 can see separate a battery yard from a
matched control, well enough to justify GPU time?

Four guards, each a lesson this repo has already paid for, all carried over verbatim from
`hydro_stage4_model.py`:

  GroupKFold on the half-degree cell. A site and its own controls must land in the same
  fold, and so must neighbouring sites, or the model memorises a cell's radiometry.

  Bootstrap CI resampling SITES, not rows -- and on OUT-OF-FOLD scores. The in-sample
  variant produced ci_lo 0.975 against a CV point estimate of 0.771 in the hydro run: a
  false-PASS generator. Do not re-derive it.

  A label-permutation null shuffled WITHIN group, with both a 95th-percentile ceiling
  (thresholds invalid at this n) and -- new here -- a MEDIAN ceiling, which is what
  detects residual geometry asymmetry between positives and controls.

  Feature-group ablation. If context alone matches everything together, the classifier is
  reading stereotyped siting, not batteries.

WHAT IS DIFFERENT FROM THE HYDRO VERSION
----------------------------------------
  * The holdout is a REGION, not a bbox, and both directions are scored (G2.5). If they
    disagree by more than 0.10 the headline is the lower of the two.
  * A fourth ablation arm, `change`, holds the bi-temporal `chg_*` features. It is
    excluded from the headline `all` arm on purpose: it is the single pre-declared G2.3
    extension, and in Tier 2 it is also the most confounded family, since a newly built
    substation shows the same bare-ground-to-built transition a battery does.
  * A `battery_only` arm scores the site minus any overlapping substation against the
    same controls -- the operational form of this study's central question (G2.4-T2).

    pixi run -e ml python scripts/bess_stage2_model.py --tier 2 --run-name bess_v1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
RESULTS = ROOT / "data" / "eval_results"
EVAL_VERSION = 1

log = logging.getLogger("bess_stage2_model")

S1_PREFIX = ("vv_", "vh_", "n_bright")
CONTEXT = {"elev_m", "slope_deg", "d_road_m", "d_line_m", "d_industrial_m"}
META = {"run_name", "site_id", "region", "cell", "cell_0p5", "group", "label", "role",
        "ctrl_id", "ctrl_cell", "size_bucket", "eval_version",
        # Anything describing the target rather than measuring the imagery. d_substation_m
        # is the sharpest of these: it is near zero for every Tier-2 control by
        # construction, so leaving it in would hand the model the answer.
        "site_area_m2", "core_area_m2", "capacity_mw", "energy_mwh", "construction_year",
        "d_substation_m", "overlaps_substation", "is_hybrid", "has_pre",
        "battery_only_area_m2", "ctrl_same_cell", "ctrl_dist_m", "valid_frac_s2",
        "chg_valid_frac"}


def group_of(col: str) -> str:
    if col.startswith("chg_"):
        return "change"
    if col in CONTEXT:
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
    # `oof` is returned so the CI is built on OUT-OF-FOLD scores. See the module docstring.
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


def _prep(df: pd.DataFrame, roles: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    d = df[df.role.isin(roles)].copy()
    d["group"] = "c" + d.cell_0p5.astype(str)
    feat = [c for c in d.columns if c not in META and d[c].dtype.kind in "fi"]
    return d, feat


def _arm(d: pd.DataFrame, cols: list[str], train_mask, hold_mask, seeds) -> dict:
    from sklearn.metrics import roc_auc_score
    tr, ho = d[train_mask], d[hold_mask]
    y = tr.label.values.astype(int)
    groups = tr.group.values
    aucs = [fit_auc(tr[cols].values, y, groups, seed=sd)[0] for sd in seeds]
    auc = float(np.nanmean(aucs))
    spread = float(np.nanmax(aucs) - np.nanmin(aucs)) if len(aucs) > 1 else 0.0
    _, model, oof = fit_auc(tr[cols].values, y, groups, seed=seeds[0])
    lo = hi = h = float("nan")
    if model is not None:
        ok = np.isfinite(oof)
        lo, hi = site_bootstrap(y[ok], oof[ok], tr.site_id.values[ok], seed=seeds[0])
        if len(ho) and ho.label.nunique() > 1:
            h = float(roc_auc_score(ho.label.values.astype(int),
                                    model.predict_proba(ho[cols].values)[:, 1]))
    return {"n_features": len(cols), "auc_cv": auc, "auc_seed_spread": spread,
            "ci_lo": lo, "ci_hi": hi, "auc_holdout": h}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", type=int, choices=(1, 2), required=True)
    ap.add_argument("--run-name", default="bess_v1")
    ap.add_argument("--samples", type=Path, default=None)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seeds = [int(s) for s in a.seeds.split(",")]
    samples = a.samples or RESULTS / f"bess_detectability_samples_tier{a.tier}.csv"

    raw = pd.read_csv(samples)
    d, feat = _prep(raw, ("site", "control"))
    # Static feature set = the headline. chg_* is held back for G2.3 and scored separately.
    static = [c for c in feat if group_of(c) != "change"]
    n_before = len(d)
    # how="any" matches the hydro script, but a single feature that is NaN in most rows
    # (bright_ring_contrast and vv_contrast_max both can be) would silently wipe the
    # sample. Report the per-feature NaN counts when the drop is severe rather than
    # letting a 90% loss pass as a normal run.
    d = d.dropna(subset=static, how="any")
    if len(d) < 0.7 * n_before:
        nanc = raw[static].isna().sum().sort_values(ascending=False)
        log.warning("dropna removed %d of %d rows; worst offenders:\n%s",
                    n_before - len(d), n_before, nanc[nanc > 0].head(8).to_string())
    if d.empty:
        raise SystemExit("no complete feature rows")
    n_sites = d[d.label == 1].site_id.nunique()
    log.info("tier %d: %d rows, %d sites, %d groups, %d static features (%d change)",
             a.tier, len(d), n_sites, d.group.nunique(), len(static), len(feat) - len(static))

    # G2.5: both transfer directions, scored separately, never averaged.
    us = d.region.isin(["ca", "tx"])
    directions = {"gb->us": (~us, us), "us->gb": (us, ~us)}

    rows = []
    for dname, (trm, hom) in directions.items():
        if trm.sum() == 0 or hom.sum() == 0:
            continue
        for gname, cols in [("all", static),
                            ("s2", [c for c in static if group_of(c) == "s2"]),
                            ("s1", [c for c in static if group_of(c) == "s1"]),
                            ("context", [c for c in static if group_of(c) == "context"])]:
            if not cols:
                continue
            rows.append({"direction": dname, "arm": gname,
                         **_arm(d, cols, trm, hom, seeds)})
    res = pd.DataFrame(rows)

    # Pooled CV (no region holdout) as the headline CV number, with the region holdout
    # reported alongside. Groups are half-degree cells, which never straddle regions.
    pooled = []
    allmask = pd.Series(True, index=d.index)
    nomask = pd.Series(False, index=d.index)
    for gname, cols in [("all", static),
                        ("s2", [c for c in static if group_of(c) == "s2"]),
                        ("s1", [c for c in static if group_of(c) == "s1"]),
                        ("context", [c for c in static if group_of(c) == "context"])]:
        if cols:
            pooled.append({"direction": "pooled", "arm": gname,
                           **_arm(d, cols, allmask, nomask, seeds)})
    res = pd.concat([pd.DataFrame(pooled), res], ignore_index=True)

    def _get(direction, arm, col="auc_cv"):
        m = (res.direction == direction) & (res.arm == arm)
        return float(res.loc[m, col].iloc[0]) if m.any() else float("nan")

    all_auc, ci_lo = _get("pooled", "all"), _get("pooled", "all", "ci_lo")
    spread = _get("pooled", "all", "auc_seed_spread")
    ctx = _get("pooled", "context")
    h_gb_us, h_us_gb = _get("gb->us", "all", "auc_holdout"), _get("us->gb", "all", "auc_holdout")
    hold_auc = float(np.nanmin([h_gb_us, h_us_gb]))   # G2.5: the lower one is the headline

    # --- permutation null, shuffled within group -------------------------------------
    rng = np.random.default_rng(0)
    y = d.label.values.astype(int)
    groups = d.group.values
    null = []
    for _ in range(a.n_perm):
        yp = y.copy()
        for g in np.unique(groups):
            m = groups == g
            yp[m] = rng.permutation(yp[m])
        auc, _m, _o = fit_auc(d[static].values, yp, groups, seed=0)
        if np.isfinite(auc):
            null.append(auc)
    null = np.array(null)
    null50 = float(np.median(null)) if null.size else float("nan")
    null95 = float(np.percentile(null, 95)) if null.size else float("nan")
    pval = float((null >= all_auc).mean()) if null.size else float("nan")

    # --- G2.3 change arm and G2.4 battery-only arm -----------------------------------
    chg_cols = [c for c in feat if group_of(c) == "change"]
    chg_auc = float("nan")
    if chg_cols:
        dc = d.dropna(subset=chg_cols, how="any")
        if dc[dc.label == 1].site_id.nunique() >= 30:
            chg_auc = _arm(dc, static + chg_cols, pd.Series(True, index=dc.index),
                           pd.Series(False, index=dc.index), seeds)["auc_cv"]

    bo_auc, n_bo = float("nan"), 0
    dbo, fbo = _prep(raw, ("site_bo", "control"))
    sbo = [c for c in fbo if group_of(c) != "change"]
    dbo = dbo.dropna(subset=sbo, how="any")
    n_bo = dbo[dbo.label == 1].site_id.nunique()
    if n_bo >= 30:
        bo_auc = _arm(dbo, sbo, pd.Series(True, index=dbo.index),
                      pd.Series(False, index=dbo.index), seeds)["auc_cv"]

    # ------------------------------------------------------------------ report
    tier_label = ("Tier 1 (vs matched generic land -- CANARY AND UPPER BOUND, "
                  "never the verdict)" if a.tier == 1 else
                  "Tier 2 (vs area- and shape-matched battery-free substations -- "
                  "THE HONEST NUMBER)")
    print(f"\n=== Stage 2b: {tier_label} ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\npermutation null (within-group, n={null.size}): median {null50:.3f}, "
          f"95th pct {null95:.3f}, p={pval:.3f}")

    bal = RESULTS / f"bess_balance_tier{a.tier}.csv"
    bal_ok = True
    if bal.exists():
        bt = pd.read_csv(bal)
        bad = bt[~bt.ok.astype(bool)]
        bal_ok = bad.empty
        print(f"\nP1 balance: {'PASS' if bal_ok else 'FAIL on ' + ', '.join(bad.covariate)}"
              "   (read this before the AUCs above)")
    p2_ok = null.size and null50 <= 0.55 and null95 < 0.65
    print(f"P2 null geometry: {'PASS' if p2_ok else 'FAIL'} "
          f"(median {null50:.3f} <= 0.55, 95th {null95:.3f} < 0.65)")

    print("\n--- gates ---")
    if a.tier == 1:
        g1 = all_auc >= 0.85 and ci_lo >= 0.78 and hold_auc >= 0.78
        g2 = all_auc < 0.75
        print(f"  G2.1-T1 PASS  cv {all_auc:.3f}>=0.85, ci_lo {ci_lo:.3f}>=0.78, "
              f"holdout {hold_auc:.3f}>=0.78 -> {'YES' if g1 else 'no'}")
        print(f"  G2.2-T1       cv<0.75 -> {'YES: STOP AND DEBUG THE HARNESS' if g2 else 'no'}")
        verdict = ("harness sound, Tier 2 may be read" if g1 else
                   "DEBUG THE HARNESS before reading Tier 2" if g2 else
                   "Tier 1 marginal -- report, and read Tier 2 with that in mind")
    else:
        g1 = all_auc >= 0.75 and ci_lo >= 0.68 and hold_auc >= 0.70
        g2 = all_auc < 0.65 or hold_auc < 0.60 or ci_lo <= 0.55
        g4 = np.isfinite(bo_auc) and bo_auc < all_auc - 0.10
        g7 = np.isfinite(ctx) and ctx >= all_auc - 0.03
        print(f"  G2.1-T2 PASS  cv {all_auc:.3f}>=0.75, ci_lo {ci_lo:.3f}>=0.68, "
              f"holdout {hold_auc:.3f}>=0.70 -> {'YES' if g1 else 'no'}")
        print(f"  G2.2-T2 FAIL  cv<0.65 or holdout<0.60 or ci_lo<=0.55 -> "
              f"{'YES' if g2 else 'no'}")
        print(f"  G2.4-T2 switchyard  battery-only {bo_auc:.3f} (n={n_bo}) vs all "
              f"{all_auc:.3f} -> "
              f"{'READS THE SWITCHYARD, NOT THE BATTERY' if g4 else 'battery-only holds up'}")
        print(f"  G2.7 ablation  context-only {ctx:.3f} vs all {all_auc:.3f} -> "
              f"{'CONTEXT DOMINATES: a siting prior, not a detector' if g7 else 'no'}")
        print(f"  G2.5 transfer  gb->us {h_gb_us:.3f}, us->gb {h_us_gb:.3f} "
              f"(headline uses the lower: {hold_auc:.3f})"
              + ("  !! directions differ by >0.10" if abs(h_gb_us - h_us_gb) > 0.10 else ""))
        if np.isfinite(chg_auc):
            print(f"  G2.3 extension  static+change {chg_auc:.3f} vs static "
                  f"{all_auc:.3f}   (only to be read in the 0.65-0.75 dead band)")
        verdict = ("PASS -> Stage 5 (build chips, fine-tune)" if g1 and not g7 and not g4 else
                   "FAIL -> do not spend GPU; write up the negative result" if g2 else
                   "AMBIGUOUS -> the one pre-declared extension (bi-temporal change), "
                   "then decide")
    if not bal_ok:
        verdict = "INVALID (covariate balance failed) -- " + verdict
    if not p2_ok:
        verdict = "INVALID (permutation null geometry) -- " + verdict
    if spread > 0.03:
        print(f"  !! seed spread {spread:.3f} > 0.03 -- report the mean and widen the CI")
    print(f"\nVERDICT (tier {a.tier}): {verdict}")

    # G2.6 size and hybrid strata: mandatory reporting. In this lineage everything turns
    # out to be a size story.
    strata = []
    for key in ("size_bucket", "is_hybrid"):
        if key not in d.columns:
            continue
        print(f"\n--- AUC by {key} (reporting, not a gate) ---")
        for b, grp in d.groupby(key):
            sid = grp[grp.label == 1].site_id.nunique()
            if sid < 5:
                continue
            auc, _m, _o = fit_auc(grp[static].values, grp.label.values.astype(int),
                                  grp.group.values, seed=seeds[0])
            print(f"  {str(b):>10}  n_sites={sid:3d}  auc={auc:.3f}")
            strata.append({"run_name": a.run_name, "tier": a.tier, "stratum": key,
                           "value": str(b), "n_sites": sid, "auc": auc})
    if strata:
        sp = RESULTS / "bess_size_buckets.csv"
        sdf = pd.DataFrame(strata)
        if sp.exists():
            prev = pd.read_csv(sp)
            prev = prev[~((prev.run_name == a.run_name) & (prev.tier == a.tier))]
            sdf = pd.concat([prev, sdf], ignore_index=True)
        sdf.to_csv(sp, index=False)

    res.insert(0, "run_name", a.run_name)
    res.insert(1, "tier", a.tier)
    res["eval_version"] = EVAL_VERSION
    res["null_p50"] = null50
    res["null_p95"] = null95
    res["perm_p"] = pval
    res["n_sites"] = n_sites
    res["balance_pass"] = bal_ok
    res["auc_battery_only"] = bo_auc
    res["n_battery_only"] = n_bo
    res["auc_static_plus_change"] = chg_auc
    res["verdict"] = verdict
    res["generated"] = dt.datetime.now().isoformat(timespec="seconds")
    out = RESULTS / "bess_stage2_summary.csv"
    if out.exists():
        prev = pd.read_csv(out)
        prev = prev[~((prev.run_name == a.run_name) & (prev.tier == a.tier)
                      & (prev.eval_version == EVAL_VERSION))]
        res = pd.concat([prev, res], ignore_index=True)
    res.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
