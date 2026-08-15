"""sindh_test prob rasters for the v14 multitemporal-vs-matched-baseline ablation, for
a real field_eval.py (region-wide AUC/P@50) instead of eval_v14_installations.py's
cheap peak-in-polygon proxy.

Multitemporal cost trick: fetch each cell's multi-date S2 stack ONCE per cell (on the
cell's own composite_0.tif grid, via odc GeoBox -- same co-registration trick as
build_multidate_chips.py / eval_v14_installations.py), then slide the 224px window
across that in-memory stack. This is ~145 STAC loads total, not one per sliding window.

Writes:
  sindh_test_v14baseline/prob   v14_baseline_matched (single median composite)
  sindh_test_v14multi/prob      v14_multitemporal (4-date stack, TemporalWrapper)

Usage:
  pixi run -e ml python scripts/infer_sindh_test_v14.py \
      --baseline-ckpt data/models/stageA_v14_baseline_matched/terramind-sub-epoch=45-step=3680.ckpt \
      --multi-ckpt data/models/stageA_v14_multitemporal/terramind-sub-epoch=83-step=5964.ckpt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("infer_sindh_test_v14")

WIN, STRIDE = 224, 104
N_DATES = 4
S2_WINDOW = ("2025-11-01", "2026-03-15")  # matches eval_v14_installations.py

REGION = ROOT / "data/osmose_regions/sindh_test"
OUT = {"baseline": ROOT / "data/osmose_regions/sindh_test_v14baseline/prob",
       "multi": ROOT / "data/osmose_regions/sindh_test_v14multi/prob"}


def _predict_multitemporal(task, device, stack: np.ndarray) -> np.ndarray:
    """stack: (T, 10, H, W) -> (10, T, H, W) -> (1, 10, T, H, W) -> prob (H, W)."""
    import torch

    x = torch.from_numpy(stack.astype("float32") / 10000.0).permute(1, 0, 2, 3)[None].to(device)
    with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
        out = task(x)
        logits = out.output if hasattr(out, "output") else out
        prob = torch.softmax(logits, dim=1)[0, 1].float().cpu().numpy()
    return prob


def main() -> None:
    from build_multidate_chips import fetch_dates_geobox
    from odc.geo.geobox import GeoBox
    from subdetect.infer import load_model, predict_window

    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-ckpt", required=True)
    ap.add_argument("--multi-ckpt", required=True)
    ap.add_argument("--n-dates", type=int, default=N_DATES)
    ap.add_argument("--limit", type=int, default=0, help="only process the first N cells (smoke test)")
    a = ap.parse_args()

    for d in OUT.values():
        d.mkdir(parents=True, exist_ok=True)

    base_task, base_device = load_model(Path(a.baseline_ckpt))
    multi_task, multi_device = load_model(Path(a.multi_ckpt))
    hann = np.outer(np.hanning(WIN), np.hanning(WIN)).astype("float32") + 1e-3

    comp_dir = REGION / "composites"
    cells = sorted(p.name for p in comp_dir.iterdir() if p.is_dir())
    if a.limit:
        cells = cells[:a.limit]
    log.info("%d cells", len(cells))

    for i, cell in enumerate(cells, 1):
        base_out = OUT["baseline"] / f"{cell}.tif"
        multi_out = OUT["multi"] / f"{cell}.tif"
        s2_path = comp_dir / cell / "composite_0.tif"
        if not s2_path.exists():
            log.warning("skipping %s (missing composite)", cell)
            continue

        with rasterio.open(s2_path) as src:
            H, W, transform, crs = src.height, src.width, src.transform, src.crs
            base_full = src.read().astype("float32")[:10] if not base_out.exists() else None

        rows = sorted(set(list(range(0, max(H - WIN, 0) + 1, STRIDE)) + [max(H - WIN, 0)]))
        cols = sorted(set(list(range(0, max(W - WIN, 0) + 1, STRIDE)) + [max(W - WIN, 0)]))

        # --- baseline (single median composite, cheap, purely local) ---
        if not base_out.exists():
            acc = np.zeros((H, W), "float32")
            wacc = np.zeros((H, W), "float32")
            valid_any = np.zeros((H, W), bool)
            for r in rows:
                for c in cols:
                    h, w = min(WIN, H - r), min(WIN, W - c)
                    arr = base_full[:, r:r + h, c:c + w]
                    if arr.size == 0 or (arr > 0).mean() < 0.2:
                        continue
                    win_arr = np.zeros((10, WIN, WIN), "float32")
                    win_arr[:, :h, :w] = arr
                    prob = predict_window(base_task, ["S2L2A"], base_device, win_arr)
                    acc[r:r+h, c:c+w] += prob[:h, :w] * hann[:h, :w]
                    wacc[r:r+h, c:c+w] += hann[:h, :w]
                    valid_any[r:r+h, c:c+w] |= (arr > 0).any(axis=0)
            prob_full = np.where(wacc > 0, acc / np.maximum(wacc, 1e-6), 0.0)
            prob_full[~valid_any] = 0.0
            with rasterio.open(base_out, "w", driver="GTiff", width=W, height=H, count=1,
                               dtype="uint8", crs=crs, transform=transform,
                               compress="deflate", predictor=2) as dst:
                dst.write((np.clip(prob_full, 0, 1) * 255).astype("uint8"), 1)

        # --- multitemporal (fetch the whole cell's multi-date stack ONCE) ---
        if not multi_out.exists():
            gbox = GeoBox((H, W), transform, crs)
            minx, miny = transform.c, transform.f + H * transform.e
            maxx, maxy = transform.c + W * transform.a, transform.f
            lons, lats = warp_transform(crs, "EPSG:4326", [minx, maxx], [miny, maxy])
            bbox_wgs84 = (min(lons), min(lats), max(lons), max(lats))
            try:
                dates = fetch_dates_geobox(bbox_wgs84, gbox, S2_WINDOW, a.n_dates)
            except Exception as e:  # noqa: BLE001
                log.warning("fetch failed for %s: %s", cell, e)
                dates = []
            if not dates:
                log.warning("no usable dates for %s, skipping multitemporal", cell)
            else:
                while len(dates) < a.n_dates:
                    dates.append(dates[-1])
                stack_full = np.stack(dates, axis=0)  # (T, 10, H, W)
                acc = np.zeros((H, W), "float32")
                wacc = np.zeros((H, W), "float32")
                valid_any = np.zeros((H, W), bool)
                for r in rows:
                    for c in cols:
                        h, w = min(WIN, H - r), min(WIN, W - c)
                        arr0 = stack_full[0, :, r:r + h, c:c + w]
                        if arr0.size == 0 or (arr0 > 0).mean() < 0.2:
                            continue
                        win_stack = np.zeros((a.n_dates, 10, WIN, WIN), "float32")
                        win_stack[:, :, :h, :w] = stack_full[:, :, r:r + h, c:c + w]
                        prob = _predict_multitemporal(multi_task, multi_device, win_stack)
                        acc[r:r+h, c:c+w] += prob[:h, :w] * hann[:h, :w]
                        wacc[r:r+h, c:c+w] += hann[:h, :w]
                        valid_any[r:r+h, c:c+w] |= (arr0 > 0).any(axis=0)
                prob_full = np.where(wacc > 0, acc / np.maximum(wacc, 1e-6), 0.0)
                prob_full[~valid_any] = 0.0
                with rasterio.open(multi_out, "w", driver="GTiff", width=W, height=H, count=1,
                                   dtype="uint8", crs=crs, transform=transform,
                                   compress="deflate", predictor=2) as dst:
                    dst.write((np.clip(prob_full, 0, 1) * 255).astype("uint8"), 1)

        if i % 10 == 0 or i == len(cells):
            log.info("[%d/%d] %s", i, len(cells), cell)


if __name__ == "__main__":
    main()
