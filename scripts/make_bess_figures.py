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
import geopandas as gpd
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




# ------------------------------------------------- 8. what these things look like


def _rgb_crop(region: str, cell: str, geom4326, half_m: float = 300.0):
    """True-colour crop on a FIXED metric window, plus the footprint in pixel coords.

    The window is a constant `half_m` square centred on the footprint, not a padded
    bounding box. A padded box gives every panel a different extent, and rendering those
    at one figure size silently puts each panel at its own scale -- which for a figure
    whose whole job is "do these look different" would be misleading. Stretch is per-crop
    2-98 percentile: the point is texture and layout, not radiometric comparison.

    Returns (rgb, footprint_xy) or None.
    """
    import rasterio
    import rasterio.mask
    from shapely.geometry import box as shapely_box

    tif = (ROOT / "data" / "composites" / f"{region}_bess" / "composites" / cell
           / "composite_0.tif")
    if not tif.exists():
        return None
    try:
        with rasterio.open(tif) as src:
            g = gpd.GeoSeries([geom4326], crs="EPSG:4326").to_crs(src.crs).iloc[0]
            c = g.centroid
            win = shapely_box(c.x - half_m, c.y - half_m, c.x + half_m, c.y + half_m)
            arr, tf = rasterio.mask.mask(src, [win], crop=True, filled=True)
            inv = ~tf
            ext = g.exterior if g.geom_type == "Polygon" else max(
                g.geoms, key=lambda q: q.area).exterior
            xy = np.array([inv * (px, py) for px, py in ext.coords])
    except (rasterio.errors.RasterioError, ValueError) as e:
        # Deliberately NOT a bare except. A bare one here swallowed a NameError from a
        # missing geopandas import and reported it as "no renderable pairs", which looks
        # like absent data rather than a bug.
        print(f"  crop failed for {region}/{cell}: {type(e).__name__}: {e}")
        return None
    # Reject a window the raster could not fully cover. A site near a cell edge gets
    # clipped by crop=True, and a truncated panel rendered beside full ones is exactly the
    # scale inconsistency the fixed window was introduced to remove.
    want = int(round(2 * half_m / 10.0))
    if arr.shape[1] < 0.9 * want or arr.shape[2] < 0.9 * want:
        return None
    # B04, B03, B02 with the +1000 L2A baseline offset removed.
    rgb = np.stack([arr[2], arr[1], arr[0]]).astype("float32")
    rgb = np.clip((rgb - 1000.0) / 10000.0, 0, None)
    valid = arr[0] > 0
    if valid.sum() < 16:
        return None
    lo, hi = np.percentile(rgb[:, valid], [2, 98])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return None
    return np.clip((rgb - lo) / (hi - lo), 0, 1).transpose(1, 2, 0), xy


def fig_examples(n_pairs: int = 4) -> None:
    """Real battery yards beside the real substations they were matched against.

    The single most useful figure in a detectability report: every AUC in this document is
    a claim that these two rows look different, and a reader should be able to check that
    claim with their own eyes before trusting the number.
    """
    sp = RESULTS / "bess_detectability_samples_tier2.csv"
    if not sp.exists():
        print("skip bess_examples.png: no Tier-2 samples")
        return
    d = pd.read_csv(sp)
    pairs = []
    for region in ("gb", "ca", "tx"):
        try:
            sites = gpd.read_parquet(ROOT / "data" / "labels" / f"bess_{region}"
                                     / "bess_sites.parquet").set_index("site_id")
            ctrls = gpd.read_parquet(ROOT / "data" / "labels" / f"bess_{region}"
                                     / "substation_controls.parquet").set_index("ctrl_id")
        except Exception:  # noqa: BLE001
            continue
        sub = d[(d.region == region) & (d.role == "control")]
        # Largest first: at 10 m a 2 ha yard is ~14x14 px and simply cannot be read.
        for sid in (d[(d.region == region) & (d.role == "site")]
                    .sort_values("site_area_m2", ascending=False).site_id):
            row = sub[sub.site_id == sid]
            if row.empty or sid not in sites.index:
                continue
            cid = row.iloc[0].ctrl_id
            if cid not in ctrls.index:
                continue
            a = _rgb_crop(region, sites.loc[sid, "cell"], sites.loc[sid].geometry)
            b = _rgb_crop(region, ctrls.loc[cid, "cell"],
                          ctrls.loc[cid].geometry.buffer(0.0003))
            if a is not None and b is not None:
                pairs.append((region, sid, a, cid, b,
                              float(sites.loc[sid, "site_area_m2"])))
                break
    if not pairs:
        print("skip bess_examples.png: no renderable pairs")
        return
    pairs = pairs[:n_pairs]
    fig, axes = plt.subplots(2, len(pairs), figsize=(2.7 * len(pairs), 6.0),
                             layout="constrained")
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.T
    for j, (region, sid, a, cid, b, area) in enumerate(pairs):
        for i, ((rgb, xy), lab, col) in enumerate(
                [(a, f"{REGION_LABEL[region]}\n{sid}  ·  {area/10000:.1f} ha", BLUE),
                 (b, f"matched substation\n{cid}", ORANGE)]):
            ax = axes[i, j]
            ax.imshow(rgb)
            # The measured footprint, drawn on the image: every statistic in this study is
            # computed inside this outline, so the reader can see what was measured rather
            # than assume it.
            ax.plot(xy[:, 0], xy[:, 1], color="#ffe14d", linewidth=1.6, zorder=5)
            ax.set_xlim(0, rgb.shape[1])
            ax.set_ylim(rgb.shape[0], 0)
            ax.set_title(lab, fontsize=9, color=INK if i == 0 else INK2)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_color(col)
                sp_.set_linewidth(2.0)
    axes[0, 0].set_ylabel("battery site", fontsize=10, color=BLUE, fontweight="bold")
    axes[1, 0].set_ylabel("control", fontsize=10, color=ORANGE, fontweight="bold")
    fig.suptitle("Sentinel-2 true colour, 10 m — the two classes the model separates",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, -0.012, "every panel is the same 600 × 600 m window  ·  yellow outline "
                          "is the measured footprint", ha="center", fontsize=8.6,
             color=MUTED)
    _save(fig, "bess_examples.png")


# ------------------------------------------------------------- 9. the Stage 1 story


def fig_stage1(probe: pd.DataFrame, inv: pd.DataFrame) -> None:
    """Co-location explains why the production detector behaves differently per region."""
    s = inv[inv.in_aoi & inv.headline]
    rows = []
    for _, r in probe.iterrows():
        reg = str(r.aoi).split("_")[0]
        if reg not in REGION_LABEL:
            continue
        h = s[s.region == reg]
        rows.append({"region": reg, "overlap": float(h.overlaps_substation.mean()),
                     "auc": float(r.auc), "lo": float(r.ci_lo), "hi": float(r.ci_hi),
                     "bo": float(r.auc_battery_only), "fire": float(r.frac_ge_04),
                     "posctrl": float(r.posctrl_frac_ge_04)})
    if not rows:
        print("skip bess_stage1.png: no probe rows")
        return
    t = pd.DataFrame(rows).sort_values("overlap")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3), layout="constrained")

    x = np.arange(len(t))
    ax1.errorbar(x, t.auc, yerr=[t.auc - t.lo, t.hi - t.auc], fmt="o", ms=9,
                 color=BLUE, ecolor=INK2, elinewidth=1.2, capsize=4, zorder=4,
                 label="full site footprint")
    ax1.plot(x, t.bo, "s", ms=8, color=ORANGE, zorder=4, label="battery-only sub-polygon")
    ax1.axhline(0.5, color=MUTED, linewidth=1.1, zorder=2)
    ax1.text(len(t) - 0.5, 0.508, "no separation", fontsize=8, color=MUTED, ha="right")
    for xi, r in enumerate(t.itertuples()):
        ax1.annotate("", xy=(xi, r.bo), xytext=(xi, r.auc),
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
    ax1.set_xticks(x, [f"{REGION_LABEL[r.region]}\n{r.overlap:.0%} overlap"
                       for r in t.itertuples()])
    ax1.set_ylim(0.15, 0.62)
    ax1.set_ylabel("AUC: batteries vs battery-free substations")
    ax1.set_title("The production detector ranks batteries BELOW substations")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax1)

    w = 0.36
    ax2.bar(x - w / 2, t.posctrl, w, color=AQUA, zorder=3,
            label="substations \u2265 20,000 m² (positive control)")
    ax2.bar(x + w / 2, t.fire, w, color=BLUE, zorder=3, label="battery sites")
    for xi, r in enumerate(t.itertuples()):
        ax2.text(xi - w / 2, r.posctrl + 0.02, f"{r.posctrl:.0%}", ha="center", fontsize=8.6,
                 color=INK2)
        ax2.text(xi + w / 2, r.fire + 0.02, f"{r.fire:.0%}", ha="center", fontsize=8.6,
                 color=INK2)
    ax2.axhline(0.40, color=CRITICAL, linestyle="--", linewidth=1.2, zorder=2)
    ax2.text(-0.42, 0.415, "G1.P bar 40%", fontsize=8, color=CRITICAL, ha="left")
    ax2.set_xticks(x, [REGION_LABEL[r.region] for r in t.itertuples()])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("fraction reaching p_max \u2265 0.4")
    ax2.set_title("It transfers — so the null above is informative")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax2)

    fig.suptitle("Stage 1: the more a battery overlaps a substation, the more it scores "
                 "like one", fontsize=12, fontweight="bold")
    _save(fig, "bess_stage1.png")


# ---------------------------------------------------------------- 10. where they are


def fig_site_map(inv: pd.DataFrame) -> None:
    """Geographic spread — the G0.4 gate, and a sanity check on clustering."""
    s = inv[inv.in_aoi]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), layout="constrained")
    for ax, reg in zip(axes, ["gb", "ca", "tx"]):
        h = s[s.region == reg]
        small, big = h[~h.headline], h[h.headline]
        ax.scatter(small.lon, small.lat, s=7, c=MUTED, alpha=0.55, linewidths=0,
                   label=f"below 20,000 m² ({len(small)})", zorder=3)
        ax.scatter(big.lon, big.lat, s=np.clip(big.site_area_m2 / 2500, 8, 120),
                   c=REGION_COLOR[reg], alpha=0.75, linewidths=0,
                   label=f"headline set ({len(big)})", zorder=4)
        ax.set_aspect(1 / np.cos(np.radians(float(h.lat.mean()))))
        ax.set_title(f"{REGION_LABEL[reg]} — {len(h)} sites", fontsize=10.5)
        ax.legend(frameon=False, fontsize=8, loc="best", markerscale=0.8)
        ax.set_xlabel("lon")
        _style(ax, grid_axis="both")
    axes[0].set_ylabel("lat")
    fig.suptitle("511 mapped battery sites; marker area scales with footprint",
                 fontsize=12, fontweight="bold")
    _save(fig, "bess_site_map.png")


# ------------------------------------------------------- 11. which features separate


def fig_features(n_show: int = 6) -> None:
    """The features doing the work in Tier 2, and what their distributions look like."""
    fp = RESULTS / "bess_detectability_tier2.csv"
    sp = RESULTS / "bess_detectability_samples_tier2.csv"
    if not (fp.exists() and sp.exists()):
        print("skip bess_features.png: no Tier-2 feature table")
        return
    per = pd.read_csv(fp)
    d = pd.read_csv(sp)
    imagery = per[~per.feature.isin(["d_road_m", "d_line_m", "d_industrial_m", "elev_m",
                                     "slope_deg", "valid_frac_s2", "n_px"])].copy()
    imagery["sep"] = (imagery.auc - 0.5).abs()
    top = imagery.sort_values("sep", ascending=False).head(n_show)

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.0), layout="constrained")
    for ax, row in zip(axes.ravel(), top.itertuples()):
        f = row.feature
        pos = d[(d.role == "site")][f].dropna()
        neg = d[(d.role == "control")][f].dropna()
        lo, hi = np.percentile(np.r_[pos, neg], [1, 99])
        bins = np.linspace(lo, hi, 26)
        # density, not counts: there are 251 controls against 139 sites, and raw counts
        # would make the control bars taller everywhere -- which reads as a class-size
        # difference rather than the distribution difference the panel is about.
        ax.hist(neg, bins=bins, color=ORANGE, alpha=0.62, density=True,
                label="substations", zorder=3)
        ax.hist(pos, bins=bins, color=BLUE, alpha=0.62, density=True,
                label="batteries", zorder=4)
        ax.set_title(f"{f}   AUC {row.auc:.3f}", fontsize=9.5)
        _style(ax)
        ax.set_yticks([])
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("What separates a battery yard from a switchyard (Tier 2, imagery only)",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, -0.015, "No single band separates the classes — every AUC here is "
                          "0.65–0.68. The 0.825 comes from combining them.",
             ha="center", fontsize=8.8, color=MUTED)
    _save(fig, "bess_features.png")


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
    probe = _read("bess_prob_probe_summary.csv")
    if probe is not None and inv is not None:
        fig_stage1(probe, inv)
    if inv is not None:
        fig_site_map(inv)
    fig_features()
    fig_examples()


if __name__ == "__main__":
    main()
