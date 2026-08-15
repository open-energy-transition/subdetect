"""Per-installation eval for v14_multitemporal / v14_baseline_matched: same 170 real
Pakistan substations (sindh_test footprint) and same peak-probability-in-polygon
metric as scripts/proxy_multidate_maxfusion.py, so results are directly comparable to
that no-retrain proxy signal. Full field_eval.py (region-wide raster + polygonize +
AUC/P@50) would need multi-date fetches across every sliding window of all 145
sindh_test cells -- far more STAC calls than fetching once per labeled installation;
this script is the deliberately cheaper substitute for the overnight timeline.

Usage:
  pixi run -e ml python scripts/eval_v14_installations.py --checkpoint <ckpt> --multitemporal
  pixi run -e ml python scripts/eval_v14_installations.py --checkpoint <ckpt>   # single-image baseline
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
import torch
from rasterio import features as rio_features
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_v14")

WIN = 224
SINDH_BBOX = (67.0, 24.0, 69.5, 26.5)
S2_WINDOW = ("2025-11-01", "2026-03-15")


def main() -> None:
    from build_multidate_chips import fetch_dates_geobox
    from odc.geo.geobox import GeoBox
    from subdetect.infer import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--multitemporal", action="store_true")
    ap.add_argument("--n-dates", type=int, default=4)
    ap.add_argument("--out-name", default=None)
    a = ap.parse_args()

    task, device = load_model(Path(a.checkpoint))

    subs = gpd.read_parquet(ROOT / "data/labels/pakistan/substations_poly.parquet")
    sel = subs.cx[SINDH_BBOX[0]:SINDH_BBOX[2], SINDH_BBOX[1]:SINDH_BBOX[3]].reset_index(drop=True)
    log.info("%d substations to test (multitemporal=%s)", len(sel), a.multitemporal)

    comp_dir = ROOT / "data/osmose_regions/sindh_test/composites"
    cell_files = [p / "composite_0.tif" for p in comp_dir.iterdir() if (p / "composite_0.tif").exists()]

    rows = []
    for i, row in sel.iterrows():
        c = row.geometry.centroid
        lon, lat = c.x, c.y
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
            transform = src.window_transform(win)
            if a.multitemporal:
                gbox = GeoBox((WIN, WIN), transform, src.crs)
            else:
                base_arr = src.read(window=win, boundless=True, fill_value=0).astype("float32")[:10]

        if a.multitemporal:
            minx, miny = transform.c, transform.f + WIN * transform.e
            maxx, maxy = transform.c + WIN * transform.a, transform.f
            lons, lats = warp_transform(crs, "EPSG:4326", [minx, maxx], [miny, maxy])
            bbox_wgs84 = (min(lons), min(lats), max(lons), max(lats))
            try:
                dates = fetch_dates_geobox(bbox_wgs84, gbox, S2_WINDOW, a.n_dates)
            except Exception as e:  # noqa: BLE001
                log.warning("fetch failed for substation %d: %s", i, e)
                continue
            if not dates:
                continue
            while len(dates) < a.n_dates:
                dates.append(dates[-1])
            stack = np.stack(dates, axis=0)  # (T, 10, H, W)
            x_t = torch.from_numpy(stack.astype("float32") / 10000.0).permute(1, 0, 2, 3)[None].to(device)
        else:
            x_t = torch.from_numpy(base_arr / 10000.0)[None].to(device)

        with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
            out = task(x_t)
            logits = out.output if hasattr(out, "output") else out
            prob = torch.softmax(logits, dim=1)[0, 1].float().cpu().numpy()

        mask = rio_features.geometry_mask([row.geometry.__geo_interface__], out_shape=(WIN, WIN),
                                          transform=transform, invert=True, all_touched=True)
        if not mask.any():
            mask = np.zeros((WIN, WIN), bool)
            mask[WIN // 2, WIN // 2] = True

        rows.append({"area_m2": row.area_m2, "peak_prob": float(prob[mask].max())})
        if len(rows) % 20 == 0:
            log.info("[%d/%d] processed", len(rows), len(sel))

    df = pd.DataFrame(rows)
    name = a.out_name or ("multitemporal" if a.multitemporal else "baseline")
    out = ROOT / f"data/eval_results/v14_{name}_installations.csv"
    df.to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(df))

    bins = [0, 2000, 5000, 20000, np.inf]
    labels = ["<2000", "2000-5000", "5000-20000", ">=20000"]
    df["bucket"] = pd.cut(df.area_m2, bins=bins, labels=labels)
    summary = df.groupby("bucket", observed=True).agg(n=("area_m2", "size"),
                                                       mean_peak=("peak_prob", "mean"))
    print(summary)


if __name__ == "__main__":
    main()
