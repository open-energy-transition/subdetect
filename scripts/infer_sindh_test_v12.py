"""sindh_test prob rasters for v12_up2 (2x input upsampling ablation): arm alone +
mean fusion with v5 S1.

Same windowing as infer_sindh_test_v11.py / infer_sindh_test_arms.py, but the S2 arm
is fed 2x-upsampled windows and its output probability map downsampled back to native
resolution before Hann-blending (see subdetect.infer.predict_window) -- this checkpoint
was trained with data.upsample: 2 (configs/terramind_sub_v12_up2_s2only.yaml) so raw
224px windows would be the wrong input size.

Writes:
  sindh_test_v12/prob         v12 S2 arm alone       (run-name v12_up2_s2only)
  sindh_test_v12mean/prob     (p_s1v5 + p_s2v12) / 2  (run-name v12_up2_mean_fusion)

Usage:
  pixi run -e ml python scripts/infer_sindh_test_v12.py --checkpoint data/models/stageA_v12_up2_s2only/<best>.ckpt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("infer_sindh_test_v12")

S1_V5 = ROOT / "data/models/stageA_v5_s1only/terramind-sub-epoch=24-step=6975.ckpt"
WIN, STRIDE = 224, 104
UPSAMPLE = 2  # must match configs/terramind_sub_v12_up2_s2only.yaml's data.upsample

REGION = ROOT / "data/osmose_regions/sindh_test"
OUT = {"v12s2": ROOT / "data/osmose_regions/sindh_test_v12/prob",
       "v12fused": ROOT / "data/osmose_regions/sindh_test_v12mean/prob"}


def main() -> None:
    from subdetect.infer import load_model, predict_window

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="v12_up2_s2only checkpoint")
    ap.add_argument("--s1-ckpt", default=str(S1_V5))
    a = ap.parse_args()

    for d in OUT.values():
        d.mkdir(parents=True, exist_ok=True)

    tasks = {}
    for name, ckpt in (("s1v5", Path(a.s1_ckpt)), ("s2v12", Path(a.checkpoint))):
        task, device = load_model(ckpt)
        mods = task.hparams.get("model_args", {}).get("backbone_modalities", ["S2L2A"])
        tasks[name] = (task, mods)
    hann = np.outer(np.hanning(WIN), np.hanning(WIN)).astype("float32") + 1e-3
    upsamples = {"s1v5": 1, "s2v12": UPSAMPLE}

    def predict(name, task, mods, s2_np, s1_np):
        return predict_window(task, mods, device, s2_np, s1_np, upsample=upsamples[name])

    comp_dir = REGION / "composites"
    cells = sorted(p.name for p in comp_dir.iterdir() if p.is_dir())
    log.info("%d cells", len(cells))
    for i, cell in enumerate(cells, 1):
        if all((d / f"{cell}.tif").exists() for d in OUT.values()):
            continue
        s2_path = comp_dir / cell / "composite_0.tif"
        s1_path = comp_dir / cell / "composite_s1.tif"
        if not (s2_path.exists() and s1_path.exists()):
            log.warning("skipping %s (missing composite)", cell)
            continue
        with rasterio.open(s2_path) as src, rasterio.open(s1_path) as src1:
            H, W, transform, crs = src.height, src.width, src.transform, src.crs
            acc = {k: np.zeros((H, W), "float32") for k in OUT}
            wacc = np.zeros((H, W), "float32")
            valid_any = np.zeros((H, W), bool)
            rows = sorted(set(list(range(0, max(H - WIN, 0) + 1, STRIDE)) + [max(H - WIN, 0)]))
            cols = sorted(set(list(range(0, max(W - WIN, 0) + 1, STRIDE)) + [max(W - WIN, 0)]))
            for r in rows:
                for c in cols:
                    win = Window(c, r, WIN, WIN)
                    s2_np = src.read(window=win, boundless=True, fill_value=0).astype("float32")[:10]
                    h, w = min(WIN, H - r), min(WIN, W - c)
                    if (s2_np[:, :h, :w] > 0).mean() < 0.2:
                        continue
                    s1_np = src1.read(window=win, boundless=True, fill_value=0).astype("float32")[:2]
                    p1 = predict("s1v5", *tasks["s1v5"], s2_np, s1_np)
                    p2 = predict("s2v12", *tasks["s2v12"], s2_np, None)
                    combo = {"v12s2": p2, "v12fused": (p1 + p2) / 2}
                    for k, prob in combo.items():
                        acc[k][r:r+h, c:c+w] += prob[:h, :w] * hann[:h, :w]
                    wacc[r:r+h, c:c+w] += hann[:h, :w]
                    valid_any[r:r+h, c:c+w] |= (s2_np[:, :h, :w] > 0).any(axis=0)
        for k, d in OUT.items():
            prob_full = np.where(wacc > 0, acc[k] / np.maximum(wacc, 1e-6), 0.0)
            prob_full[~valid_any] = 0.0
            with rasterio.open(d / f"{cell}.tif", "w", driver="GTiff", width=W, height=H,
                               count=1, dtype="uint8", crs=crs, transform=transform,
                               compress="deflate", predictor=2) as dst:
                dst.write((np.clip(prob_full, 0, 1) * 255).astype("uint8"), 1)
        if i % 25 == 0 or i == len(cells):
            log.info("[%d/%d] %s", i, len(cells), cell)


if __name__ == "__main__":
    main()
