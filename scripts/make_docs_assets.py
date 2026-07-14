"""Generate every figure used in README.md / docs/*.md from real project data.

Reads composited imagery, probability rasters, and lead GeoJSONs already on disk
under data/osmose_regions/; writes PNG (+ SVG for the architecture diagram) into
docs/assets/. Nothing here re-runs inference or training -- it only visualizes
existing outputs, so it's safe to re-run any time those outputs change.

Usage:
  pixi run python scripts/make_docs_assets.py                  # all figures
  pixi run python scripts/make_docs_assets.py --only architecture
  pixi run python scripts/make_docs_assets.py --only worked-example
  pixi run python scripts/make_docs_assets.py --only regional-map --region sindh_test
  pixi run python scripts/make_docs_assets.py --only model-lineage
  pixi run python scripts/make_docs_assets.py --list            # print asset inventory, no writes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subdetect.config import S1_OFFSET_DB, S1_SCALE  # noqa: E402

OUT_DIR = ROOT / "docs" / "assets"
YUNNAN = ROOT / "data" / "osmose_regions" / "yunnan"
SINDH = ROOT / "data" / "osmose_regions" / "sindh_test"
EXAMPLE_CELL = "00977_00242"
EQ = "EPSG:6933"

# ASSET_INVENTORY: (name, output filename, one-line description) -- kept in sync
# with the generator functions below; --list prints this without importing rasterio.
ASSET_INVENTORY = [
    ("architecture", "architecture.svg", "Two-swimlane pipeline diagram: core CLI + Osmose side-path, earthpv relationship"),
    ("worked-example", "worked_example_panel.png", "4-panel figure for cell 00977_00242: S2 RGB, S1 false-color, probability heatmap, detected polygon"),
    ("regional-map", "regional_map_yunnan.png", "Yunnan: composited coverage, Osmose endpoints, all leads colored by confidence, top-50 highlighted"),
    ("regional-map-sindh", "regional_map_sindh_test.png", "Sindh: same treatment, older/narrower lead schema"),
    ("model-lineage-iou", "model_lineage_iou.png", "Pixel IoU across v2 -> v2_india -> v4_s2only -> v4_s1fusion -> v4_s1only"),
    ("model-lineage-ablation", "model_lineage_ablation.png", "3-arm ablation: >=20k m2 recall and >=220kV recall, s2only/s1fusion/s1only"),
]


def _percentile_stretch(band: np.ndarray, lo=2, hi=98) -> np.ndarray:
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype="float32")
    p_lo, p_hi = np.percentile(valid, [lo, hi])
    out = np.clip((band.astype("float32") - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    out[band <= 0] = 0
    return out


def make_architecture(out_dir: Path) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14.2, 7))
    ax.set_xlim(0, 14.1)
    ax.set_ylim(0, 7)
    ax.axis("off")

    colors = {"ingest": "#8ecae6", "compute": "#ffb703", "model": "#fb8500", "output": "#219ebc"}

    def box(x, y, w, h, label, kind, fontsize=9.5):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                            facecolor=colors[kind], edgecolor="black", linewidth=1.1, zorder=2)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, zorder=3, wrap=True)
        return (x, y, w, h)

    def arrow(b1, b2, style="-|>", color="black"):
        x1 = b1[0] + b1[2]
        y1 = b1[1] + b1[3] / 2
        x2 = b2[0]
        y2 = b2[1] + b2[3] / 2
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=14, color=color, linewidth=1.3, zorder=1))

    # Top swimlane: core 9-stage CLI
    ax.text(0.1, 6.6, "Core pipeline (src/subdetect/cli.py)", fontsize=11, fontweight="bold")
    w, h, y = 1.35, 0.9, 5.3
    xs = [0.1 + i * 1.55 for i in range(9)]
    labels = [("osm", "ingest"), ("roi", "compute"), ("compose", "ingest"), ("chips", "compute"),
              ("train", "model"), ("evaluate", "model"), ("infer", "model"),
              ("postprocess", "compute"), ("export", "output")]
    boxes = [box(x, y, w, h, lbl, kind) for x, (lbl, kind) in zip(xs, labels)]
    for a, b in zip(boxes, boxes[1:]):
        arrow(a, b)

    # earthpv dependency, dashed box feeding compose/chips
    ep = box(0.1, 3.7, 2.2, 0.75, "earthpv (sibling repo)\nshared grid_origin", "ingest", fontsize=8.5)
    ep_patch = ax.patches[-1]
    ep_patch.set_linestyle("dashed")
    ep_patch.set_facecolor("#e0e0e0")
    ax.annotate("", xy=(xs[2] + w / 2, y), xytext=(ep[0] + ep[2] / 2, 3.7 + 0.75),
                arrowprops=dict(arrowstyle="-|>", linestyle="dashed", color="#555", linewidth=1.1))
    ax.text(2.5, 3.55, "link_pakistan_composites.py: hardlinks already-composited\nPakistan S2 cells (byte-identical, zero re-download)",
            fontsize=7.5, color="#444", style="italic")

    # Bottom swimlane: Osmose side-path
    ax.text(0.1, 2.7, "Osmose regional lead-generation (scripts/osmose_detect.py)", fontsize=11, fontweight="bold")
    y2 = 1.4
    xs2 = [0.1 + i * 2.1 for i in range(6)]
    labels2 = [("fetch Osmose\nissues", "ingest"), ("filter vs\nOverpass subs", "compute"),
               ("cell plan", "compute"), ("compose S2+S1", "ingest"),
               ("dual-model\ndetect", "model"), ("hysteresis\npolygonize + rank", "output")]
    w2 = 1.85
    boxes2 = [box(x, y2, w2, h, lbl, kind, fontsize=8.5) for x, (lbl, kind) in zip(xs2, labels2)]
    for a, b in zip(boxes2, boxes2[1:]):
        arrow(a, b)

    # rejoin arrow from bottom lane's "detect" step up to top lane's infer/postprocess concept
    ax.annotate("", xy=(xs[6] + w / 2, y), xytext=(xs2[4] + w2 / 2, y2 + h),
                arrowprops=dict(arrowstyle="-|>", color="#888", linewidth=1.2,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(6.7, 3.05, "reuses the established best model stack\n(P = P_S1only x (0.5 + 0.5 x P_S2only))",
            fontsize=7.5, color="#444", style="italic")

    legend_items = [("ingest", "data ingest"), ("compute", "geometry / compositing"),
                    ("model", "TerraMind model step"), ("output", "review-ready output")]
    for i, (kind, label) in enumerate(legend_items):
        lx = 0.1 + i * 3.1
        ax.add_patch(FancyBboxPatch((lx, 0.05), 0.3, 0.3, boxstyle="round,pad=0.02",
                                    facecolor=colors[kind], edgecolor="black", linewidth=0.8))
        ax.text(lx + 0.42, 0.2, label, fontsize=8, va="center")

    fig.suptitle("subdetect pipeline: core CLI + Osmose-guided lead generation", fontsize=13, y=0.98)
    fig.savefig(out_dir / "architecture.svg", bbox_inches="tight")
    fig.savefig(out_dir / "architecture.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'architecture.svg'} (+.png)")


def _db_stretch(db: np.ndarray, valid: np.ndarray, lo=2, hi=98) -> np.ndarray:
    """Percentile-stretch a dB array (can be negative) using an explicit valid mask --
    unlike _percentile_stretch, correctness here does NOT depend on values being > 0."""
    out = np.zeros_like(db, dtype="float32")
    if not valid.any():
        return out
    p_lo, p_hi = np.percentile(db[valid], [lo, hi])
    out[valid] = np.clip((db[valid] - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    return out


def _extent_of(transform, shape) -> tuple[float, float, float, float]:
    """(left, right, bottom, top) for imshow(extent=...), matching a geopandas
    overlay in the same CRS -- required since raster pixel-index space and the
    leads' projected CRS are not the same coordinate system."""
    height, width = shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    return (left, right, bottom, top)


def make_worked_example(out_dir: Path, region_dir: Path = YUNNAN, cell: str = EXAMPLE_CELL) -> None:
    import geopandas as gpd
    import rasterio
    import rasterio.warp
    from shapely.geometry import box as shapely_box

    cell_dir = region_dir / "composites" / cell
    s2_path = cell_dir / "composite_0.tif"
    s1_path = cell_dir / "composite_s1.tif"
    prob_path = region_dir / "prob" / f"{cell}.tif"
    leads_path = region_dir / "leads_pilot.geojson" if (region_dir / "leads_pilot.geojson").exists() \
        else region_dir / "leads.geojson"

    with rasterio.open(s2_path) as src:
        s2 = src.read([3, 2, 1]).astype("float32") / 10000.0  # B04,B03,B02 -> RGB
        transform, crs, shape = src.transform, src.crs, (src.height, src.width)
        bounds4326 = rasterio.warp.transform_bounds(crs, "EPSG:4326", *src.bounds)
    rgb = np.dstack([_percentile_stretch(s2[i]) for i in range(3)])
    extent = _extent_of(transform, shape)

    with rasterio.open(s1_path) as src:
        s1 = src.read([1, 2]).astype("float32")
    valid = s1[0] > 0
    vv_db = s1[0] / S1_SCALE - S1_OFFSET_DB
    vh_db = s1[1] / S1_SCALE - S1_OFFSET_DB
    diff_db = vh_db - vv_db
    sar_rgb = np.dstack([
        _db_stretch(vv_db, valid),
        _db_stretch(vh_db, valid),
        _db_stretch(diff_db, valid),
    ])

    with rasterio.open(prob_path) as src:
        prob = src.read(1).astype("float32") / 255.0

    leads = gpd.read_file(leads_path)
    cell_bbox = shapely_box(*bounds4326)
    here = leads[leads.intersects(cell_bbox)].to_crs(crs)

    # Callout: the highest-confidence lead in this cell (not necessarily the
    # pilot's overall rank-1 -- a single 0.1 deg cell can hold several components).
    conf_col = "confidence" if "confidence" in here.columns else "conf_max"
    highlight = here.sort_values(conf_col, ascending=False).iloc[0] if not here.empty else None

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6))

    axes[0].imshow(rgb, extent=extent)
    axes[0].set_title("A. Sentinel-2 true color\n(B04, B03, B02)", fontsize=11)

    axes[1].imshow(sar_rgb, extent=extent)
    axes[1].set_title("B. Sentinel-1 false color\n(R=VV, G=VH, B=VH-VV, dB)", fontsize=11)

    axes[2].imshow(rgb * 0.5 + 0.25, extent=extent)
    im = axes[2].imshow(prob, cmap="magma", alpha=0.7, vmin=0, vmax=1, extent=extent)
    axes[2].set_title("C. Predicted probability\n(P = P_S1only x (0.5 + 0.5 x P_S2only))", fontsize=11)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(rgb, extent=extent)
    if not here.empty:
        here.boundary.plot(ax=axes[3], edgecolor="#39ff14", linewidth=2.2)
        conf = highlight.get("confidence", highlight.get("conf_max"))
        txt = (f"confidence {conf:.2f}\narea {highlight.area_m2/1e4:.1f} ha\n"
               f"endpoint {highlight.endpoint_dist_m:.0f} m")
        axes[3].text(0.02, 0.98, txt, transform=axes[3].transAxes, va="top", fontsize=9,
                     bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=4))
    axes[3].set_title("D. Detected polygon\n(hysteresis polygonize, seed 0.4 -> grow 0.2)", fontsize=11)

    bar_m = 2000.0  # 2 km scale bar, drawn directly in the raster's projected CRS units
    for ax in axes:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks([]); ax.set_yticks([])
        x0 = extent[0] + (extent[1] - extent[0]) * 0.05
        y0 = extent[2] + (extent[3] - extent[2]) * 0.06
        ax.plot([x0, x0 + bar_m], [y0, y0], color="white", linewidth=3,
                path_effects=[patheffects.withStroke(linewidth=5, foreground="black")])
        ax.text(x0, y0 + (extent[3] - extent[2]) * 0.02, f"{bar_m/1000:.1f} km", color="white",
                fontsize=8, path_effects=[patheffects.withStroke(linewidth=3, foreground="black")])

    fig.suptitle(f"Anatomy of one detection -- cell {cell} (Yunnan pilot)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = out_dir / "worked_example_panel.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def make_regional_map(out_dir: Path, region_dir: Path, region_name: str) -> None:
    import geopandas as gpd
    import rasterio
    import rasterio.warp
    from shapely.geometry import box as shapely_box

    cell_dirs = sorted((region_dir / "composites").glob("*/composite_0.tif"))
    footprints = []
    for tif in cell_dirs:
        with rasterio.open(tif) as src:
            footprints.append(shapely_box(*rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)))
    coverage = gpd.GeoDataFrame(geometry=footprints, crs="EPSG:4326")

    endpoints_path = region_dir / "endpoints.geojson"
    endpoints = gpd.read_file(endpoints_path) if endpoints_path.exists() else None

    leads_path = region_dir / "leads_pilot.geojson" if (region_dir / "leads_pilot.geojson").exists() \
        else region_dir / "leads.geojson"
    leads = gpd.read_file(leads_path)
    conf_col = "confidence" if "confidence" in leads.columns else "conf_max"

    top50_path = region_dir / "leads_pilot_top50.geojson"
    top = gpd.read_file(top50_path) if top50_path.exists() else leads.sort_values(conf_col, ascending=False).head(50)

    # Osmose endpoints are fetched province/region-wide (thousands, cheap to query),
    # but composited coverage is bandwidth-bounded to a much smaller pilot cluster --
    # zoom to that worked area so the actual pipeline output isn't lost as a speck
    # among endpoints that were never composited. Full count kept in the title.
    cb = coverage.total_bounds
    pad_x, pad_y = (cb[2] - cb[0]) * 0.15 + 0.02, (cb[3] - cb[1]) * 0.15 + 0.02

    fig, ax = plt.subplots(figsize=(11, 10))
    coverage.plot(ax=ax, facecolor="#eeeeee", edgecolor="#cccccc", linewidth=0.5, zorder=1)
    if endpoints is not None and not endpoints.empty:
        endpoints.plot(ax=ax, marker="x", color="#888888", markersize=18, linewidth=0.8,
                       zorder=2, label=f"Osmose endpoints (composited-area subset)")
    leads_c = leads.copy()
    leads_c["centroid"] = leads_c.geometry.centroid
    sizes = 8 + 40 * (leads_c.area_m2 / leads_c.area_m2.max()).clip(0, 1)
    sc = ax.scatter(leads_c.centroid.x, leads_c.centroid.y, c=leads_c[conf_col], cmap="viridis",
                    s=sizes, alpha=0.75, zorder=3, edgecolor="none")
    top_c = top.copy()
    top_c["centroid"] = top_c.geometry.centroid
    ax.scatter(top_c.centroid.x, top_c.centroid.y, facecolor="none", edgecolor="gold",
              linewidth=1.6, s=120, zorder=4, label=f"top {len(top)} curated leads")
    if "rank" in top_c.columns:
        r1 = top_c[top_c["rank"] == 1]
        if not r1.empty:
            pt = r1.iloc[0].centroid
            ax.annotate("rank 1\n(worked example)", xy=(pt.x, pt.y), xytext=(15, 15),
                       textcoords="offset points", fontsize=9, fontweight="bold",
                       arrowprops=dict(arrowstyle="->", color="black"))
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="confidence")
    ax.set_xlim(cb[0] - pad_x, cb[2] + pad_x)
    ax.set_ylim(cb[1] - pad_y, cb[3] + pad_y)
    ax.legend(loc="lower left", fontsize=9)
    n_ep = len(endpoints) if endpoints is not None else 0
    ax.set_title(f"{region_name}: {len(cell_dirs)} composited cells, {len(leads)} leads\n"
                f"({n_ep} Osmose endpoints found region-wide; compositing is bandwidth-bounded "
                "to this pilot cluster)", fontsize=12)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    out_path = out_dir / f"regional_map_{region_name}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# Source of truth: README.md's model lineage / ablation tables. Keep these two in
# sync by hand if the README table changes -- these are not re-derived from logs
# (no per-model summary metrics are cached to disk outside stdout).
def make_model_lineage(out_dir: Path) -> None:
    iou_data = [
        ("v2", 0.30),
        ("v2_india", 0.274),
        ("v4_s2only", 0.243),
        ("v4_s1fusion", 0.266),
        ("v4_s1only", 0.310),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [d[0] for d in iou_data]
    vals = [d[1] for d in iou_data]
    colors = ["#8ecae6"] * (len(iou_data) - 1) + ["#fb8500"]
    bars = ax.barh(names, vals, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("pixel IoU (Pakistan val)")
    ax.set_title("Model lineage: pixel IoU across the ablation history")
    ax.set_xlim(0, 0.36)
    fig.tight_layout()
    out1 = out_dir / "model_lineage_iou.png"
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out1}")

    arms = ["v4_s2only\n(control)", "v4_s1fusion\n(S2+S1)", "v4_s1only\n(VV/VH only)"]
    recall_20k = [32, 63, 84]
    recall_220 = [71, 71, 100]
    x = np.arange(len(arms))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - width / 2, recall_20k, width, label=">=20k m2 recall", color="#219ebc")
    b2 = ax.bar(x + width / 2, recall_220, width, label=">=220kV recall", color="#fb8500")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}%",
                    ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(arms)
    ax.set_ylabel("recall (%)")
    ax.set_ylim(0, 112)
    ax.set_title("3-arm ablation (2026-07-11): radar alone is the strongest single signal")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out2 = out_dir / "model_lineage_ablation.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out2}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["architecture", "worked-example", "regional-map", "model-lineage"])
    ap.add_argument("--region", default="yunnan", choices=["yunnan", "sindh_test"])
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for name, fname, desc in ASSET_INVENTORY:
            print(f"{name:<20} docs/assets/{fname:<32} {desc}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    region_dir = YUNNAN if a.region == "yunnan" else SINDH

    if a.only in (None, "architecture"):
        make_architecture(OUT_DIR)
    if a.only in (None, "worked-example"):
        make_worked_example(OUT_DIR)
    if a.only in (None, "regional-map"):
        make_regional_map(OUT_DIR, YUNNAN, "yunnan")
        if a.only == "regional-map" and a.region == "sindh_test":
            make_regional_map(OUT_DIR, SINDH, "sindh_test")
        elif a.only is None:
            make_regional_map(OUT_DIR, SINDH, "sindh_test")
    if a.only in (None, "model-lineage"):
        make_model_lineage(OUT_DIR)


if __name__ == "__main__":
    main()
