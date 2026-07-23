"""sindh_test prob rasters for v9_retrain_s2only: arm alone + mean fusion with v5 S1.

v9_retrain is the same recipe as production v9_s2only but trained on the
geographic-split-fixed chip index (see MEMORY subdetect-val-split-fix). EarlyStopping
fired at epoch 28 with best checkpoint frozen at epoch 5 (val/IoU_1=0.181,
patience=15) -- comparing against production v9 here via field_eval on sindh_test,
not val/IoU_1 directly (not comparable across the index rebuild).

Same windowing as infer_sindh_test_arms.py / infer_sindh_test_v11.py.
Writes:
  sindh_test_v9retrain/prob      v9_retrain S2 arm alone      (run-name v9_retrain_s2only)
  sindh_test_v9retrainmean/prob  (p_s1v5 + p_s2v9retrain) / 2  (run-name v9_retrain_mean_fusion)

Usage:
  pixi run -e ml python scripts/infer_sindh_test_v9retrain.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("infer_sindh_test_v9retrain")

S1_V5 = ROOT / "data/models/stageA_v5_s1only/terramind-sub-epoch=24-step=6975.ckpt"
S2_V9RETRAIN = ROOT / "data/models/stageA_v9_retrain_s2only/terramind-sub-epoch=05-step=11616.ckpt"
WIN, STRIDE = 224, 104

REGION = ROOT / "data/osmose_regions/sindh_test"
OUT = {"v9retrains2": ROOT / "data/osmose_regions/sindh_test_v9retrain/prob",
       "v9retrainfused": ROOT / "data/osmose_regions/sindh_test_v9retrainmean/prob"}


def main() -> None:
    import torch
    from subdetect.infer import _standardize_s1, load_model

    for d in OUT.values():
        d.mkdir(parents=True, exist_ok=True)

    tasks = {}
    for name, ckpt in (("s1v5", S1_V5), ("s2v9retrain", S2_V9RETRAIN)):
        task, device = load_model(ckpt)
        mods = task.hparams.get("model_args", {}).get("backbone_modalities", ["S2L2A"])
        tasks[name] = (task, mods)
    hann = np.outer(np.hanning(WIN), np.hanning(WIN)).astype("float32") + 1e-3

    def predict(task, mods, s2_np, s1_np):
        s2_t = torch.from_numpy(s2_np / 10000.0)[None].to(device)
        x = ({m: (s2_t if m == "S2L2A" else torch.from_numpy(_standardize_s1(s1_np))[None].to(device))
              for m in mods} if "S1RTC" in mods else s2_t)
        with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
            out = task(x)
            logits = out.output if hasattr(out, "output") else out
            return torch.softmax(logits, 1)[0, 1].float().cpu().numpy()

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
                    p1 = predict(*tasks["s1v5"], s2_np, s1_np)
                    p2 = predict(*tasks["s2v9retrain"], s2_np, None)
                    combo = {"v9retrains2": p2, "v9retrainfused": (p1 + p2) / 2}
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
