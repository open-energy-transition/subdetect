"""Fast, no-retrain proxy for the multi-temporal max-fusion hypothesis: does keeping
individual S2 revisits (instead of collapsing to one median composite) recover
small-substation signal that the current pipeline destroys?

For each real Pakistan substation label in the sindh_test footprint (any size, no
floor -- 170 total), fetches up to N individual least-cloudy S2 scenes for just that
substation's 224px window (cheap: one small bbox, not a full composited cell), runs
the existing v9_s2only checkpoint independently on each date AND on the current
median-composite baseline (from the already-built composite_0.tif), and compares the
peak predicted probability inside the substation polygon: baseline vs per-date max.

This is a raster/output-space ensemble (independent forward passes + pixel-wise max),
NOT a trained latent-space fusion model (see tower_prior.py-style modality additions
for how a real trainable multi-temporal encoder would need to hook into terratorch) --
it only tests whether the SIGNAL survives in individual dates that the median discards,
as a cheap upfront check before investing in the full architecture change.

Usage:
  pixi run -e ml python scripts/proxy_multidate_maxfusion.py --n-dates 4 --limit 0
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rio_features
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("proxy_multidate")

S2_CKPT = ROOT / "data/models/stageA_v9_s2only/terramind-sub-epoch=52-step=93545.ckpt"
WIN = 224
SINDH_BBOX = (67.0, 24.0, 69.5, 26.5)
S2_WINDOW = ("2025-11-01", "2026-03-15")


def fetch_dates(bbox_wgs84, date_range, max_dates: int):
    """Up to `max_dates` individually cloud-masked (10, H, W) uint16 S2 arrays for a
    small bbox, least-cloudy first. Mirrors imagery._annual_composite_via's search +
    load but returns the per-date stack instead of collapsing with .median()."""
    import odc.stac
    from rasterio.crs import CRS

    from subdetect.imagery import MAX_CLOUD, _SCL_VALID, _catalog

    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    search = _catalog().search(
        collections=["sentinel-2-l2a"], bbox=bbox_wgs84,
        datetime=f"{date_range[0]}/{date_range[1]}",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
    )
    items = sorted(search.items(), key=lambda it: it.properties.get("eo:cloud_cover", 100))
    if not items:
        return []
    items = items[:max_dates]
    lon = (bbox_wgs84[0] + bbox_wgs84[2]) / 2
    lat = (bbox_wgs84[1] + bbox_wgs84[3]) / 2
    epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) / 6) + 1
    ds = odc.stac.load(
        items, bands=[*bands, "SCL"], groupby="solar_day",
        chunks={"x": 512, "y": 512}, fail_on_error=False,
        bbox=bbox_wgs84, resolution=10, crs=CRS.from_epsg(epsg),
    )
    valid = ds["SCL"].isin(_SCL_VALID)
    masked = ds[bands].where(valid).fillna(0).astype("uint16").compute()
    out = []
    for t in range(masked.sizes["time"]):
        arr = np.stack([masked[b].isel(time=t).values for b in bands], axis=0)
        if (arr > 0).mean() > 0.5:  # skip mostly-empty reads
            out.append((arr, masked.odc.transform, CRS.from_epsg(epsg)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dates", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="cap number of substations (0=all)")
    a = ap.parse_args()

    from subdetect.infer import load_model, predict_window

    subs = gpd.read_parquet(ROOT / "data/labels/pakistan/substations_poly.parquet")
    sel = subs.cx[SINDH_BBOX[0]:SINDH_BBOX[2], SINDH_BBOX[1]:SINDH_BBOX[3]].reset_index(drop=True)
    if a.limit:
        sel = sel.sort_values("area_m2").head(a.limit)  # prioritize small subs if capped
    log.info("%d substations to test", len(sel))

    comp_dir = ROOT / "data/osmose_regions/sindh_test/composites"
    cell_files = [p / "composite_0.tif" for p in comp_dir.iterdir() if (p / "composite_0.tif").exists()]

    task, device = load_model(S2_CKPT)
    mods = task.hparams.get("model_args", {}).get("backbone_modalities", ["S2L2A"])

    rows = []
    for i, row in sel.iterrows():
        c = row.geometry.centroid
        lon, lat = c.x, c.y
        # locate the composited cell + pixel window covering this substation
        target = None
        for f in cell_files:
            with rasterio.open(f) as src:
                xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
                x, y = xs[0], ys[0]
                b = src.bounds
                if b.left <= x <= b.right and b.bottom <= y <= b.top:
                    target = (f, src.crs, x, y)
                    break
        if target is None:
            continue
        f, crs, x, y = target
        with rasterio.open(f) as src:
            row_px, col_px = src.index(x, y)
            half = WIN // 2
            r0, c0 = max(row_px - half, 0), max(col_px - half, 0)
            win = Window(c0, r0, WIN, WIN)
            base_arr = src.read(window=win, boundless=True, fill_value=0).astype("float32")[:10]
            transform = src.window_transform(win)

        base_prob = predict_window(task, mods, device, base_arr, None)

        # window bbox in WGS84 for the multi-date fetch (small, cheap)
        minx, miny = transform.c, transform.f + WIN * transform.e
        maxx, maxy = transform.c + WIN * transform.a, transform.f
        lons, lats = warp_transform(crs, "EPSG:4326", [minx, maxx], [miny, maxy])
        bbox_wgs84 = (min(lons), min(lats), max(lons), max(lats))

        try:
            dates = fetch_dates(bbox_wgs84, S2_WINDOW, a.n_dates)
        except Exception as e:  # noqa: BLE001
            log.warning("fetch failed for substation %d: %s", i, e)
            continue
        if not dates:
            continue

        date_probs = []
        for arr, _, _ in dates:
            h, w = arr.shape[-2:]
            if h != WIN or w != WIN:
                pad = np.zeros((10, WIN, WIN), "float32")
                pad[:, :h, :w] = arr[:, :WIN, :WIN]
                arr = pad
            date_probs.append(predict_window(task, mods, device, arr.astype("float32"), None))
        max_prob = np.maximum.reduce(date_probs)

        mask = rio_features.geometry_mask([row.geometry.__geo_interface__], out_shape=(WIN, WIN),
                                          transform=transform, invert=True, all_touched=True)
        if not mask.any():
            # tiny/point substation: use the single center pixel
            mask = np.zeros((WIN, WIN), bool)
            mask[WIN // 2, WIN // 2] = True

        rows.append({
            "area_m2": row.area_m2,
            "n_dates_used": len(dates),
            "base_peak": float(base_prob[mask].max()),
            "maxfused_peak": float(max_prob[mask].max()),
        })
        if len(rows) % 20 == 0:
            log.info("[%d/%d] processed", len(rows), len(sel))

    df = pd.DataFrame(rows)
    out = ROOT / "data/eval_results/proxy_multidate_maxfusion.csv"
    df.to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(df))

    bins = [0, 2000, 5000, 20000, np.inf]
    labels = ["<2000", "2000-5000", "5000-20000", ">=20000"]
    df["bucket"] = pd.cut(df.area_m2, bins=bins, labels=labels)
    df["improved"] = df.maxfused_peak > df.base_peak
    summary = df.groupby("bucket", observed=True).agg(
        n=("area_m2", "size"),
        base_mean_peak=("base_peak", "mean"),
        maxfused_mean_peak=("maxfused_peak", "mean"),
        frac_improved=("improved", "mean"),
    )
    print(summary)
    summary.to_csv(ROOT / "data/eval_results/proxy_multidate_maxfusion_summary.csv")


if __name__ == "__main__":
    main()
