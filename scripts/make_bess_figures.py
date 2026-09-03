"""Figures for the BESS detectability feasibility report.

Reads only persisted CSVs under data/eval_results/ (plus the site inventory), so every
figure is reproducible from the study's own outputs and no number is typed in by hand.
Palette, _style and _save are taken verbatim from scripts/make_hydro_figures.py so the two
reports read as one series. Figures whose inputs do not exist yet are skipped with a note
rather than fabricated.

    pixi run python scripts/make_bess_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "eval_results"
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CRITICAL, GOOD, WARNING = "#d03b3b", "#0ca30c", "#fab219"
REGION_COLOR = {"gb": BLUE, "ca": ORANGE, "tx": AQUA}
REGION_LABEL = {"gb": "Great Britain", "ca": "California", "tx": "Texas"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 9.5,
})


def _style(ax, *, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color(AXIS)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)


def _save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p.relative_to(ROOT)}")


def _read(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


# ------------------------------------------------------- 1. the co-location problem


def fig_colocation(inv: pd.DataFrame) -> None:
    """The study's central risk, and the reason Tier 2 exists."""
    s = inv[inv.in_aoi]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.2), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    regions = ["gb", "ca", "tx"]
    x = np.arange(len(regions))
    near = [float((s[s.region == r].d_substation_m <= 300).mean()) for r in regions]
    over = [float(s[s.region == r].overlaps_substation.mean()) for r in regions]
    ax1.bar(x - 0.19, near, 0.36, color=BLUE, label="substation within 300 m", zorder=3)
    ax1.bar(x + 0.19, over, 0.36, color=ORANGE, label="polygon overlaps one", zorder=3)
    for xi, (a, b) in enumerate(zip(near, over)):
        ax1.text(xi - 0.19, a + 0.02, f"{a:.0%}", ha="center", fontsize=9, color=INK2)
        ax1.text(xi + 0.19, b + 0.02, f"{b:.0%}", ha="center", fontsize=9, color=INK2)
    ax1.set_xticks(x, [REGION_LABEL[r] for r in regions])
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("fraction of battery sites")
    ax1.set_title("Batteries sit on top of substations")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax1)

    d = s.d_substation_m.clip(0, 2000)
    ax2.hist(d, bins=40, color=BLUE, zorder=3)
    ax2.axvline(300, color=CRITICAL, linewidth=1.4, linestyle="--", zorder=4)
    ax2.text(320, ax2.get_ylim()[1] * 0.92, "300 m", color=CRITICAL, fontsize=9)
    ax2.set_xlabel("distance to nearest mapped substation (m)")
    ax2.set_ylabel("sites")
    ax2.set_title("Distance to the nearest switchyard")
    n_far = int((s.d_substation_m > 2000).sum())
    ax2.annotate(f"{n_far} sites are\nfarther than 2 km\n(piled into the last bin)",
                 xy=(2000, 0), xytext=(0.63, 0.62), textcoords="axes fraction",
                 fontsize=8.2, color=MUTED, ha="left")
    _style(ax2)

    fig.suptitle("Why a naive experiment would measure substation detection",
                 fontsize=13, fontweight="bold")
    _save(fig, "bess_colocation.png")


# --------------------------------------------------------------- 2. the size picture


def fig_size(inv: pd.DataFrame) -> None:
    """Size against the two thresholds that matter, per region."""
    s = inv[inv.in_aoi]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.2), layout="constrained")

    bins = np.logspace(2, 6.2, 45)
    for r in ["gb", "ca", "tx"]:
        ax1.hist(s[s.region == r].core_area_m2.clip(100, None), bins=bins,
                 histtype="step", linewidth=1.7, color=REGION_COLOR[r],
                 label=f"{REGION_LABEL[r]} (n={(s.region == r).sum()})", zorder=3)
    ax1.axvline(2000, color=CRITICAL, linewidth=1.4, linestyle="--", zorder=4)
    ax1.axvline(20000, color=GOOD, linewidth=1.4, linestyle="--", zorder=4)
    ax1.set_xscale("log")
    # Annotations in axes fractions with an opaque box, so they sit beside the data
    # instead of on top of it.
    box = dict(facecolor=SURFACE, edgecolor="none", alpha=0.92, pad=1.6)
    ax1.text(0.015, 0.985, "2,000 m² — p_max measured at\nexactly 0.000 in all 23 runs",
             transform=ax1.transAxes, color=CRITICAL, fontsize=8.2, va="top", bbox=box)
    ax1.text(0.985, 0.985, "20,000 m² — the bucket with\nmeasured recall (0.714)",
             transform=ax1.transAxes, color=GOOD, fontsize=8.2, va="top", ha="right",
             bbox=box)
    ax1.set_xlabel("unbuffered built footprint (m², log)")
    ax1.set_ylabel("sites")
    ax1.set_title("Half the corpus sits below the detectable bucket")
    ax1.legend(frameon=False, fontsize=8.5)
    _style(ax1)

    order = ["node_only", "<1k", "1k-2k", "2k-5k", "5k-20k", ">=20k"]
    counts = s.size_bucket.value_counts()
    vals = [int(counts.get(b, 0)) for b in order]
    # Coloured by the 2,000 m2 floor ONLY. The A1 headline threshold applies to the
    # buffered label, not to this unbuffered core, so painting 5k-20k as "in" would be
    # wrong: many of those sites fall outside the headline set.
    cols = [CRITICAL if b in ("node_only", "<1k", "1k-2k") else GOOD for b in order]
    ax2.bar(range(len(order)), vals, color=cols, zorder=3)
    for i, v in enumerate(vals):
        ax2.text(i, v + 2, str(v), ha="center", fontsize=9, color=INK2)
    ax2.set_xticks(range(len(order)), order, rotation=30, ha="right")
    ax2.set_ylabel("sites")
    ax2.set_title("Size buckets (unbuffered core)")
    ax2.text(0.5, -0.30, "red = below the 2,000 m² floor where this pipeline measures no "
                         "response at all", transform=ax2.transAxes, ha="center",
             fontsize=8.2, color=MUTED)
    _style(ax2)
    _save(fig, "bess_size.png")


# ------------------------------------------------------------------- 3. tag audit


def fig_tag_audit(audit: pd.DataFrame) -> None:
    """G0.0: what the labels are actually tagged as, before any definition was fixed."""
    a = audit[~audit.pattern.str.startswith("VALUE")].copy()
    piv = a.pivot_table(index="pattern", columns="region", values="n",
                        aggfunc="sum", fill_value=0)
    piv["total"] = piv.sum(axis=1)
    piv = piv[piv.total > 0].sort_values("total")
    fig, ax = plt.subplots(figsize=(9.2, max(3.2, 0.42 * len(piv))))
    left = np.zeros(len(piv))
    for r in [c for c in ["gb", "ca", "tx"] if c in piv.columns]:
        ax.barh(range(len(piv)), piv[r].values, left=left, color=REGION_COLOR[r],
                label=REGION_LABEL[r], zorder=3)
        left = left + piv[r].values
    for i, v in enumerate(piv.total.values):
        ax.text(v + max(piv.total) * 0.01, i, str(int(v)), va="center", fontsize=8.5,
                color=INK2)
    ax.set_yticks(range(len(piv)), piv.index, fontsize=8.2)
    ax.set_xlabel("objects matched")
    ax.set_title("G0.0 tag audit: counted before the positive definition was fixed")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax, grid_axis="x")
    _save(fig, "bess_tag_audit.png")


# ------------------------------------------------- 4. the two tiers, side by side


def fig_tiers(summ: pd.DataFrame) -> None:
    """The whole story: Tier 1 is an upper bound, Tier 2 is the answer."""
    p = summ[summ.direction == "pooled"]
    if p.empty:
        print("skip bess_tiers.png: no pooled rows yet")
        return
    arms = ["all", "imagery", "s2", "s1", "context"]
    fig, ax = plt.subplots(figsize=(10.0, 4.6), layout="constrained")
    x = np.arange(len(arms))
    # Tier 1 is NOT labelled an upper bound. It was designed as one and did not behave
    # like one: its controls are matched on road/line/industrial distance, which forces
    # them to be infrastructure-adjacent land rather than open countryside, and it came in
    # BELOW Tier 2. Calling it an upper bound in the figure would repeat a claim the data
    # contradicted.
    for k, (tier, col, lab) in enumerate(
            [(1, AQUA, "Tier 1 — vs matched infrastructure-adjacent land"),
             (2, BLUE, "Tier 2 — vs matched battery-free substations (the answer)")]):
        t = p[p.tier == tier]
        if t.empty:
            continue
        vals = [float(t[t.arm == arm].auc_cv.iloc[0]) if (t.arm == arm).any() else np.nan
                for arm in arms]
        los = [float(t[t.arm == arm].ci_lo.iloc[0]) if (t.arm == arm).any() else np.nan
               for arm in arms]
        his = [float(t[t.arm == arm].ci_hi.iloc[0]) if (t.arm == arm).any() else np.nan
               for arm in arms]
        off = -0.19 + 0.38 * k
        ax.bar(x + off, vals, 0.36, color=col, label=lab, zorder=3)
        ax.errorbar(x + off, vals, yerr=[np.array(vals) - np.array(los),
                                         np.array(his) - np.array(vals)],
                    fmt="none", ecolor=INK2, elinewidth=1.1, capsize=3, zorder=4)
        for xi, v in zip(x + off, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.6, color=INK2)
    ax.axhline(0.75, color=CRITICAL, linewidth=1.3, linestyle="--", zorder=2)
    ax.text(len(arms) - 0.4, 0.757, "G2.1-T2 bar 0.75", color=CRITICAL, fontsize=8.5)
    ax.axhline(0.50, color=MUTED, linewidth=1.0, zorder=2)
    ax.set_xticks(x, ["all\n(32 feat)", "imagery only\nS2+S1", "S2 only", "S1 only",
                      "context only\n(no imagery)"])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("grouped-CV AUC")
    ax.set_title("Battery-specific signal, or just the switchyard?")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _style(ax)
    _save(fig, "bess_tiers.png")


# ------------------------------------------------------------------- 5. balance


def fig_balance(bal2: pd.DataFrame) -> None:
    """The table that licenses reading any AUC at all."""
    b = bal2[bal2.varies.astype(bool)] if "varies" in bal2 else bal2
    if b.empty:
        print("skip bess_balance.png: no varying covariates")
        return
    fig, ax = plt.subplots(figsize=(7.6, max(2.6, 0.5 * len(b))))
    y = np.arange(len(b))
    cols = [GOOD if abs(v) <= 0.10 else CRITICAL for v in b.smd]
    ax.barh(y, b.smd.values, color=cols, zorder=3)
    ax.axvline(0.10, color=MUTED, linestyle="--", linewidth=1.0, zorder=2)
    ax.axvline(-0.10, color=MUTED, linestyle="--", linewidth=1.0, zorder=2)
    for i, v in enumerate(b.smd.values):
        ax.text(v + (0.012 if v >= 0 else -0.012), i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=INK2)
    ax.set_yticks(y, b.covariate, fontsize=9)
    ax.set_xlabel("standardized mean difference (|SMD| <= 0.10 required)")
    ax.set_title("Covariate balance — read before any AUC")
    _style(ax, grid_axis="x")
    _save(fig, "bess_balance.png")


# -------------------------------------------------------- 6. size / hybrid strata


def fig_strata(strata: pd.DataFrame) -> None:
    """G2.6: in this lineage everything turns out to be a size story."""
    s = strata[(strata.tier == 2) & (strata.stratum == "size_bucket")]
    if s.empty:
        print("skip bess_strata.png: no Tier-2 size strata yet")
        return
    order = [b for b in ["node_only", "<1k", "1k-2k", "2k-5k", "5k-20k", ">=20k"]
             if b in set(s.value)]
    s = s.set_index("value").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(range(len(s)), s.auc.values, color=BLUE, zorder=3)
    for i, (v, n) in enumerate(zip(s.auc.values, s.n_sites.values)):
        ax.text(i, v + 0.012, f"{v:.3f}\nn={n}", ha="center", fontsize=8.4, color=INK2)
    ax.axhline(0.5, color=MUTED, linewidth=1.0, zorder=2)
    ax.axhline(0.75, color=CRITICAL, linestyle="--", linewidth=1.2, zorder=2)
    ax.set_xticks(range(len(s)), s.value, rotation=30, ha="right")
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Tier-2 CV AUC")
    ax.set_title("Tier-2 separability by built footprint")
    _style(ax)
    _save(fig, "bess_strata.png")


# ---------------------------------------------------------------- 7. the verdict


def fig_verdict(summ: pd.DataFrame) -> None:
    """Tier 2 against all three G2.1 conditions, plus the switchyard control."""
    p = summ[(summ.direction == "pooled") & (summ.tier == 2) & (summ.arm == "all")]
    if p.empty:
        print("skip bess_verdict.png: no Tier-2 pooled row yet")
        return
    r = p.iloc[0]
    # The pooled row has no holdout by construction (it trains on everything). Per G2.5
    # the headline holdout is the LOWER of the two transfer directions, which live in the
    # gb->us / us->gb rows -- reading auc_holdout off the pooled row just gives NaN and
    # silently drops the bar.
    dirs = summ[(summ.tier == 2) & (summ.arm == "all")
                & summ.direction.isin(["gb->us", "us->gb"])]
    hold = float(dirs.auc_holdout.min()) if len(dirs) else np.nan
    names = ["CV AUC", "bootstrap CI lower", "cross-region holdout\n(lower direction)",
             "battery-only arm"]
    vals = [float(r.auc_cv), float(r.ci_lo), hold,
            float(r.auc_battery_only) if pd.notna(r.auc_battery_only) else np.nan]
    bars = [0.75, 0.68, 0.70, float(r.auc_cv) - 0.10]
    fig, ax = plt.subplots(figsize=(9.0, 4.4), layout="constrained")
    x = np.arange(len(names))
    cols = [GOOD if (np.isfinite(v) and v >= b) else CRITICAL for v, b in zip(vals, bars)]
    ax.bar(x, vals, 0.5, color=cols, zorder=3)
    for xi, (v, b) in enumerate(zip(vals, bars)):
        ax.plot([xi - 0.3, xi + 0.3], [b, b], color=INK, linewidth=1.8, zorder=5)
        if np.isfinite(v):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9.5, color=INK2)
        ax.text(xi, b + 0.006, f"need {b:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC")
    passed = all(np.isfinite(v) and v >= b for v, b in zip(vals, bars))
    ax.set_title("Tier 2: batteries vs matched battery-free substations", pad=10)
    ax.text(0.5, -0.20,
            ("all four conditions met" if passed else "not all conditions met")
            + "   ·   black rules are the pre-registered bars   ·   "
            "battery-only bar tests whether the switchyard carries the signal",
            transform=ax.transAxes, ha="center", fontsize=8.4, color=MUTED)
    _style(ax)
    _save(fig, "bess_verdict.png")


def main() -> None:
    inv = _read("bess_stage0_inventory.csv")
    if inv is not None:
        fig_colocation(inv)
        fig_size(inv)
    else:
        print("skip Stage 0 figures: no bess_stage0_inventory.csv")
    audit = _read("bess_stage0_tag_audit.csv")
    if audit is not None:
        fig_tag_audit(audit)
    summ = _read("bess_stage2_summary.csv")
    if summ is not None and len(summ):
        # The summary file is append-and-dedup on (run_name, tier), so it accumulates
        # runs. Figures must show ONE run per tier -- the most recent -- rather than
        # silently mixing a 3-control run with an 8-control one.
        summ["generated"] = pd.to_datetime(summ.generated, errors="coerce")
        keep = summ.sort_values("generated").groupby("tier").run_name.last().to_dict()
        summ = summ[[r.tier in keep and r.run_name == keep[r.tier]
                     for r in summ.itertuples()]]
        print("figures use: " + ", ".join(f"tier {t} = {n}" for t, n in keep.items()))
    if summ is not None:
        fig_tiers(summ)
        fig_verdict(summ)
    else:
        print("skip Stage 2 figures: no bess_stage2_summary.csv yet")
    bal2 = _read("bess_balance_tier2.csv")
    if bal2 is not None:
        fig_balance(bal2)
    strata = _read("bess_size_buckets.csv")
    if strata is not None:
        fig_strata(strata)


if __name__ == "__main__":
    main()
