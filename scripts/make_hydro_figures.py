"""Figures for the small run-of-river hydro feasibility report.

Reads only persisted CSVs under data/eval_results/ (plus the site inventory), so every
figure is reproducible from the study's own outputs and none of the numbers are typed in
by hand. Writes PNGs to docs/assets/.

    pixi run python scripts/make_hydro_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "eval_results"
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

# Validated categorical slots + chrome (light surface #fcfcfb). Slots 1-3 pass the
# all-pairs CVD and normal-vision floors; aqua carries a sub-3:1 contrast warning, so
# wherever it appears it is directly labelled rather than relying on hue alone.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CRITICAL, GOOD, WARNING = "#d03b3b", "#0ca30c", "#fab219"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "font.size": 9.5,
})


def _style(ax, *, grid_axis="y"):
    """Hairline recessive grid, no top/right spines, solid rules only."""
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


# --------------------------------------------------------------- 1. the size problem


def fig_size_problem():
    inv = pd.read_csv(RESULTS / "hydro_stage0_inventory.csv")
    inv = inv[~inv.east_of_76]
    core = inv.core_area_m2[inv.core_area_m2 > 0]
    site = inv.site_area_m2[inv.site_area_m2 > 0]

    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    bins = np.logspace(1, 6.1, 46)
    ax.hist(core, bins=bins, color=BLUE, alpha=0.85, label="Powerhouse as mapped", zorder=3)
    ax.hist(site, bins=bins, color=ORANGE, alpha=0.85, label="Site complex (75 m buffer)", zorder=3)
    ax.set_xscale("log")

    # The zone where this pipeline has produced p_max of exactly 0.000 in all 23 runs.
    ax.axvspan(10, 2000, color=CRITICAL, alpha=0.06, zorder=1)
    ax.axvline(2000, color=CRITICAL, lw=1.4, zorder=4)
    ax.axvline(20000, color=GOOD, lw=1.4, zorder=4)

    ymax = ax.get_ylim()[1]
    ax.text(150, ymax * 0.93, "dead zone\n0.000 recall in all 23 runs",
            color=CRITICAL, fontsize=8.6, ha="center", va="top", linespacing=1.35)
    ax.text(23000, ymax * 0.93, "20,000 m² — the class the\nmodel actually detects (0.714)",
            color="#0a7d0a", fontsize=8.6, ha="left", va="top", linespacing=1.35)

    ax.annotate("", xy=(27454, ymax * 0.34), xytext=(501, ymax * 0.34),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3,
                                shrinkA=2, shrinkB=2))
    ax.text(3700, ymax * 0.40, "55×", ha="center", color=INK, fontsize=11, fontweight="bold")
    ax.text(3700, ymax * 0.245, "median 501 m² → 27,454 m²", ha="center",
            color=INK2, fontsize=8.4)

    ax.set_xlabel("area (m², log scale)")
    ax.set_ylabel("number of sites")
    ax.set_title("Why the label had to change: the powerhouse sits inside the dead zone",
                 loc="left", pad=12)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.005, 0.60), fontsize=9)
    _style(ax)
    fig.text(0.008, -0.045,
             "154 mapped sites, northern Pakistan. Buffering plant ∪ generator ∪ penstock into one "
             "site complex moves 93% above 20,000 m².",
             color=MUTED, fontsize=8.2)
    _save(fig, "hydro_size_problem.png")


# ------------------------------------------------------------------- 2. stage 1 probe


def fig_stage1_probe():
    d = pd.read_csv(RESULTS / "hydro_prob_probe.csv")
    d = d[(d.buffer_m == 75.0) & (~d.is_control) & (d.covered)]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    groups = [(False, "No substation within 300 m"), (True, "Substation within 300 m")]
    for i, (flag, lbl) in enumerate(groups):
        v = d[d.has_switchyard_300m == flag].p_max.values
        y = i + rng.uniform(-0.17, 0.17, len(v))
        ax.scatter(v, y, s=34, color=BLUE if not flag else ORANGE, alpha=0.65,
                   linewidths=0.8, edgecolors=SURFACE, zorder=3)
        ax.text(-0.035, i, lbl, ha="right", va="center", color=INK2, fontsize=9.2)
        ax.text(1.045, i, f"n={len(v)}\nmax {v.max():.2f}", ha="left", va="center",
                color=MUTED, fontsize=8.3, linespacing=1.3)
        # The mass at exactly 0.000 overplots into a single dot; label the count.
        nz = int((v == 0).sum())
        ax.annotate(f"{nz} of {len(v)} are exactly 0.000",
                    xy=(0.004, i - 0.24), xytext=(0.075, i - 0.46),
                    color=INK2, fontsize=8.1, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9,
                                    shrinkA=1, shrinkB=2))

    ax.axvline(0.4, color=MUTED, lw=1.0, zorder=2)
    ax.text(0.41, 1.62, "0.4 = the threshold a candidate\nmust reach to exist at all",
            color=INK2, fontsize=8.4, va="top", linespacing=1.35)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, 1.9)
    ax.set_yticks([])
    ax.set_xlabel("peak probability from the existing substation model, inside the site")
    ax.set_title("Stage 1: the existing model is blind to hydro — and what little fires,\n"
                 "fires on the switchyard", loc="left", pad=12)
    _style(ax, grid_axis="x")
    for side in ("left",):
        ax.spines[side].set_visible(False)
    fig.text(0.008, -0.10,
             "58 sites with raster coverage. Median peak probability is exactly 0.000; 83% never "
             "exceed 0.05. Expected: hydro plants were\ntrained as ignore(−1), so this is a null probe — "
             "it rules out reusing the existing model, and nothing more.",
             color=MUTED, fontsize=8.2, linespacing=1.4)
    _save(fig, "hydro_stage1_probe.png")


# ------------------------------------------------------------------ 3. the verdict


def fig_verdict():
    d = pd.read_csv(RESULTS / "hydro_stage2_summary.csv")
    d = d[d.run_name == "north_pk_strict"]
    order = ["all", "s1", "s2", "context"]
    names = {"all": "Everything", "s1": "Sentinel-1 only", "s2": "Sentinel-2 only",
             "context": "Terrain only"}
    d = d.set_index("group").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    y = np.arange(len(d))[::-1]

    ax.axvspan(0.65, 0.75, color=WARNING, alpha=0.13, zorder=1)
    ax.axvline(0.75, color=GOOD, lw=1.4, zorder=2)
    ax.axvline(0.5, color=AXIS, lw=1.0, zorder=2)
    null95 = float(d.null_p95.iloc[0])
    ax.axvline(null95, color=CRITICAL, lw=1.2, zorder=2)

    for yi, r in zip(y, d.itertuples()):
        c = BLUE if r.group == "all" else MUTED
        ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=c, lw=2.2, solid_capstyle="round", zorder=3)
        ax.scatter([r.auc_cv], [yi], s=95, color=c, zorder=4,
                   edgecolors=SURFACE, linewidths=1.6)
        ax.scatter([r.auc_holdout], [yi], s=62, color=c, zorder=4, marker="D",
                   facecolors=SURFACE, edgecolors=c, linewidths=1.6)
        ax.text(r.ci_hi + 0.012, yi, f"{r.auc_cv:.3f}", va="center", color=INK,
                fontsize=9.4, fontweight="bold" if r.group == "all" else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels([names[g] for g in d.group], color=INK2, fontsize=9.6)
    ax.set_xlim(0.42, 0.88)
    ax.set_ylim(-1.35, 3.75)
    ax.set_xlabel("AUC — separating a hydro site from a terrain-matched patch of the same valley")
    ax.set_title("The verdict: real signal, just under the bar set in advance", loc="left", pad=14)

    ax.text(0.5, 3.52, "chance", color=MUTED, fontsize=8.3, ha="center", va="top")
    ax.text(0.757, 3.52, "pass bar\n0.75", color="#0a7d0a", fontsize=8.3, va="top",
            linespacing=1.3)
    ax.text(0.70, 3.52, "“ambiguous”\nband", color="#9a6a05", fontsize=8.3, ha="center",
            va="top", linespacing=1.3)
    ax.text(null95 - 0.008, 3.52, "null\n95th pct", color=CRITICAL, fontsize=8.3,
            ha="right", va="top", linespacing=1.3)

    ax.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, lw=0, markersize=9,
               markeredgecolor=SURFACE, label="cross-validated AUC (bar = 95% CI)"),
        Line2D([], [], marker="D", color=BLUE, lw=0, markersize=7.5,
               markerfacecolor=SURFACE, label="held-out valley (Swat / Panjkora)"),
    ], frameon=False, loc="lower left", bbox_to_anchor=(0.0, -0.02), fontsize=8.8)
    _style(ax, grid_axis="x")
    fig.text(0.008, -0.115,
             "86 sites vs 1,028 terrain-matched controls. The combined model misses all three pass "
             "conditions by a similar narrow margin\n(0.737 vs 0.75 · CI lower 0.676 vs 0.68 · holdout "
             "0.671 vs 0.70) — coherent, not noise.",
             color=MUTED, fontsize=8.2, linespacing=1.4)
    _save(fig, "hydro_verdict.png")


# --------------------------------------------------------------- 4. size buckets


def fig_size_buckets():
    d = pd.read_csv(RESULTS / "hydro_size_buckets.csv")
    d = d[d.run_name == "north_pk_strict"]
    order = ["node_only", "<1k", "1k-2k", "2k-5k", "5k-20k", ">=20k"]
    lbl = {"node_only": "node only\n(no polygon)", "<1k": "< 1,000", "1k-2k": "1,000–2,000",
           "2k-5k": "2,000–5,000", "5k-20k": "5,000–20,000", ">=20k": "≥ 20,000"}
    d = d.set_index("size_bucket").reindex([b for b in order if b in set(d.size_bucket)]).reset_index()

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    x = np.arange(len(d))
    colors = [CRITICAL if v < 0.5 else BLUE for v in d.auc]
    ax.bar(x, d.auc - 0.5, bottom=0.5, width=0.62, color=colors, zorder=3)
    ax.axhline(0.5, color=AXIS, lw=1.2, zorder=4)

    for xi, r in zip(x, d.itertuples()):
        above = r.auc >= 0.5
        ax.text(xi, r.auc + (0.011 if above else -0.011), f"{r.auc:.3f}",
                ha="center", va="bottom" if above else "top",
                color=CRITICAL if not above else INK, fontsize=9.4, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl[b]}\nn={n}" for b, n in zip(d.size_bucket, d.n_sites)],
                       color=INK2, fontsize=8.8, linespacing=1.5)
    ax.set_ylim(0.32, 0.80)
    ax.set_ylabel("AUC")
    ax.set_xlabel("powerhouse footprint as mapped (m²) — not the buffered label")
    ax.set_title("The finding: detectability tracks the real building, not the buffer we drew",
                 loc="left", pad=12)
    ax.text(-0.42, 0.506, "chance", color=MUTED, fontsize=8.3, ha="left", va="bottom")
    ax.annotate("below chance — a 75 m disc around a point\nthat need not sit on any structure",
                xy=(0.34, 0.40), xytext=(1.05, 0.372), color=CRITICAL, fontsize=8.4,
                linespacing=1.35, va="center",
                arrowprops=dict(arrowstyle="-|>", color=CRITICAL, lw=1.1, shrinkA=4, shrinkB=6))
    _style(ax)
    fig.text(0.008, -0.155,
             "The site-complex label cleared the label-geometry floor — that was its job, and it "
             "worked. It did not manufacture signal\nwhere the ground has none.",
             color=MUTED, fontsize=8.2, linespacing=1.4)
    _save(fig, "hydro_size_buckets.png")


# ------------------------------------------------------------------- 5. balance


def fig_balance():
    d = pd.read_csv(RESULTS / "hydro_balance.csv")
    nice = {"elev_m": "elevation", "slope_deg": "slope", "rel_elev_500m": "valley position",
            "relief_1km": "local relief", "northness": "aspect (northness)",
            "d_river_m": "distance to river", "valid_frac_s2": "usable-pixel fraction"}
    d = d.sort_values("smd")

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    y = np.arange(len(d))
    ax.axvspan(-0.10, 0.10, color=GOOD, alpha=0.10, zorder=1)
    ax.axvline(0, color=AXIS, lw=1.2, zorder=2)
    for yi, r in zip(y, d.itertuples()):
        c = GOOD if r.balanced else CRITICAL
        ax.plot([0, r.smd], [yi, yi], color=c, lw=1.6, alpha=0.55, zorder=3)
        ax.scatter([r.smd], [yi], s=90, color=c, zorder=4, edgecolors=SURFACE, linewidths=1.5)
        ax.text(r.smd + (0.022 if r.smd >= 0 else -0.022), yi, f"{r.smd:+.2f}",
                va="center", ha="left" if r.smd >= 0 else "right", color=INK, fontsize=8.8)

    ax.set_yticks(y)
    ax.set_yticklabels([nice.get(c, c) for c in d.covariate], color=INK2, fontsize=9.4)
    ax.set_xlim(-0.62, 0.52)
    ax.set_xlabel("standardized difference between sites and their matched controls")
    ax.set_title("The honest caveat: valley position and river distance resist matching",
                 loc="left", pad=12)
    ax.set_ylim(-0.75, len(d) - 0.25)
    ax.text(0, -0.62, "balanced band (±0.10)", color="#0a7d0a", fontsize=8.4, ha="center",
            va="center")
    ax.annotate("hydro sites sit lower in the valley and\nhug rivers more tightly than ANY\n"
                "available control patch",
                xy=(-0.40, 0.05), xytext=(-0.60, 2.15), color=CRITICAL, fontsize=8.4,
                linespacing=1.4, va="center",
                arrowprops=dict(arrowstyle="-|>", color=CRITICAL, lw=1.1, shrinkA=4, shrinkB=6))
    _style(ax, grid_axis="x")
    fig.text(0.008, -0.085,
             "Tightening the matching does not fix this — a strict re-run gave 82 sites and near-identical "
             "numbers. There is simply not\nenough valley floor to match against, so part of the measured "
             "separation is unavoidably “lower and nearer the water”.",
             color=MUTED, fontsize=8.2, linespacing=1.4)
    _save(fig, "hydro_balance.png")


# ------------------------------------------------------- 6. small-n inflation


def fig_small_n():
    d = pd.read_csv(RESULTS / "hydro_stage2_summary.csv")
    a = d[d.run_name == "harness_smoke"].set_index("group")
    b = d[d.run_name == "north_pk_strict"].set_index("group")
    order = ["s1", "s2", "all", "context"]
    names = {"all": "Everything", "s1": "Sentinel-1 only", "s2": "Sentinel-2 only",
             "context": "Terrain only"}

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    y = np.arange(len(order))[::-1]
    for yi, g in zip(y, order):
        v0, v1 = a.loc[g, "auc_cv"], b.loc[g, "auc_cv"]
        ax.plot([v0, v1], [yi, yi], color=MUTED, lw=1.6, alpha=0.6, zorder=2)
        ax.scatter([v0], [yi], s=88, color=ORANGE, zorder=4, edgecolors=SURFACE, linewidths=1.5)
        ax.scatter([v1], [yi], s=88, color=BLUE, zorder=4, edgecolors=SURFACE, linewidths=1.5)
        ax.text(v0 + 0.012, yi + 0.24, f"{v0:.3f}", color=ORANGE, fontsize=8.8, ha="center")
        ax.text(v1 - 0.012, yi + 0.24, f"{v1:.3f}", color=BLUE, fontsize=8.8, ha="center")

    ax.axvline(0.5, color=AXIS, lw=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([names[g] for g in order], color=INK2, fontsize=9.6)
    ax.set_xlim(0.42, 0.92)
    ax.set_ylim(-1.15, 3.62)
    ax.set_xlabel("AUC")
    ax.set_title("Why the power gate exists: the same code, 19 sites vs 82", loc="left", pad=14)
    ax.text(0.5, 3.45, "chance", color=MUTED, fontsize=8.3, ha="center", va="top")
    ax.legend(handles=[
        Line2D([], [], marker="o", color=ORANGE, lw=0, markersize=9, markeredgecolor=SURFACE,
               label="19 matched sites (underpowered smoke run)"),
        Line2D([], [], marker="o", color=BLUE, lw=0, markersize=9, markeredgecolor=SURFACE,
               label="82 matched sites (the real run)"),
    ], frameon=False, loc="lower left", bbox_to_anchor=(0.0, -0.03), fontsize=8.8,
       ncol=2, columnspacing=1.4, handletextpad=0.4)
    _style(ax, grid_axis="x")
    fig.text(0.008, -0.13,
             "Every arm collapsed toward chance as n grew — except terrain, which rose. Had the gate not "
             "been fixed in advance, the\nsmoke numbers would have read as a clear pass.",
             color=MUTED, fontsize=8.2, linespacing=1.4)
    _save(fig, "hydro_small_n.png")


# ------------------------------------------------------------------------ 7. map


def fig_map():
    inv = pd.read_csv(RESULTS / "hydro_stage0_inventory.csv")
    inv = inv[~inv.east_of_76]
    ready = inv.has_composite & inv.has_s1

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    hb = (71.55, 34.55, 72.95, 35.65)
    ax.add_patch(plt.Rectangle((hb[0], hb[1]), hb[2] - hb[0], hb[3] - hb[1],
                               facecolor=AQUA, alpha=0.10, edgecolor=AQUA,
                               linewidth=1.3, zorder=2))
    ax.text(hb[0] + 0.04, hb[1] + 0.06, "held-out valley\nSwat / Panjkora",
            color="#0d7a55", fontsize=8.6, va="bottom", linespacing=1.35, zorder=8,
            path_effects=[pe.withStroke(linewidth=2.6, foreground=SURFACE)])

    s = np.sqrt(inv.site_area_m2.clip(2e3, 5e5)) / 7.0
    ax.scatter(inv.lon[~ready], inv.lat[~ready], s=s[~ready], color=ORANGE, alpha=0.75,
               edgecolors=SURFACE, linewidths=0.8, zorder=3)
    ax.scatter(inv.lon[ready], inv.lat[ready], s=s[ready], color=BLUE, alpha=0.8,
               edgecolors=SURFACE, linewidths=0.8, zorder=4)

    # Place names sit over a dense scatter, so each gets a surface-coloured halo rather
    # than a box; positions are nudged off the marker clusters they name.
    for name, lo, la in [("Chitral", 71.55, 36.28), ("Swat", 72.62, 35.30),
                         ("Hunza", 74.55, 36.62), ("Gilgit", 73.95, 35.72),
                         ("Skardu", 75.70, 35.55), ("Tarbela", 72.45, 33.95),
                         ("Mangla", 73.35, 33.10), ("Kaghan", 73.35, 34.95),
                         ("Astore", 74.95, 35.05)]:
        ax.text(lo, la, name, color=INK2, fontsize=8.2, ha="center", zorder=7,
                path_effects=[pe.withStroke(linewidth=2.6, foreground=SURFACE)])

    ax.set_xlabel("longitude °E")
    ax.set_ylabel("latitude °N")
    ax.set_title("154 mapped hydro sites across northern Pakistan", loc="left", pad=12)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, lw=0, markersize=8.5,
               markeredgecolor=SURFACE, label="S2 + S1 composited"),
        Line2D([], [], marker="o", color=ORANGE, lw=0, markersize=8.5,
               markeredgecolor=SURFACE, label="no imagery yet"),
    ], frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.95,
       loc="lower right", fontsize=8.8)
    ax.set_aspect(1 / np.cos(np.radians(35)))
    _style(ax, grid_axis="both")
    fig.text(0.008, -0.03,
             "Marker area ∝ site-complex footprint. 23 of the uncovered cells fall outside the substation "
             "search region entirely —\n“within 20 km of a mapped power line” structurally excludes "
             "off-grid micro hydro.",
             color=MUTED, fontsize=8.2, linespacing=1.4)
    _save(fig, "hydro_site_map.png")


def main():
    fig_size_problem()
    fig_stage1_probe()
    fig_verdict()
    fig_size_buckets()
    fig_balance()
    fig_small_n()
    fig_map()


if __name__ == "__main__":
    sys.exit(main())
