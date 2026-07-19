"""Fused S1xS2 inference over a full AOI's composite cells.

Hann-blended 224px windows at stride 104 (same as scripts/osmose_detect.py
gated_inference) over data/composites/<aoi>/composites/<cell>/, with the S2 arm
swappable (default: v9 epoch-52). Writes FOUR rasters per cell so no fusion-formula
change ever needs re-inference (see docs/model-lineage.md "Open problem: the 0.5
plateau" -- this is the "expose what the fusion discards" fix at raster level):

  <out-dir>/<aoi>/prob_s1/<cell>.tif    S1 arm alone
  <out-dir>/<aoi>/prob_s2/<cell>.tif    S2 arm alone
  <out-dir>/<aoi>/prob/<cell>.tif       gated fusion p_s1 * (0.5 + 0.5 * p_s2)
  <out-dir>/<aoi>/prob_mean/<cell>.tif  arithmetic mean (p_s1 + p_s2) / 2 -- scored
                                        higher than gated in eval_decision_fusion.py

All uint8 0-255, ready for `subdetect postprocess --pred-dir <out-dir>`. Cells
missing composite_s1.tif fall back to the S2 arm alone in every variant (logged).

Usage:
  pixi run -e ml python scripts/infer_fused_aoi.py --aoi pakistan
  pixi run -e ml python scripts/infer_fused_aoi.py --aoi india_pilot
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
log = logging.getLogger("infer_fused_aoi")

S1_CKPT = ROOT / "data/models/stageA_v5_s1only/terramind-sub-epoch=24-step=6975.ckpt"
S2_CKPT = ROOT / "data/models/stageA_v9_s2only/terramind-sub-epoch=52-step=93545.ckpt"
WIN, STRIDE = 224, 104


def main() -> None:
    import torch
    from subdetect.infer import _standardize_s1, load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--out-dir", default="data/predictions_v9_fused")
    ap.add_argument("--s1-ckpt", default=str(S1_CKPT))
    ap.add_argument("--s2-ckpt", default=str(S2_CKPT))
    a = ap.parse_args()

    comp_dir = ROOT / "data/composites" / a.aoi / "composites"
    out_dirs = {v: ROOT / a.out_dir / a.aoi / d for v, d in
                (("s1", "prob_s1"), ("s2", "prob_s2"), ("gated", "prob"), ("mean", "prob_mean"))}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    tasks = {}
    for name, ckpt in (("s1", Path(a.s1_ckpt)), ("s2", Path(a.s2_ckpt))):
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

    cells = sorted(p.name for p in comp_dir.iterdir() if p.is_dir())
    log.info("%s: %d cells -> %s", a.aoi, len(cells), out_dirs["gated"].parent)
    for i, cell in enumerate(cells, 1):
        if all((d / f"{cell}.tif").exists() for d in out_dirs.values()):
            continue
        s2_path = comp_dir / cell / "composite_0.tif"
        s1_path = comp_dir / cell / "composite_s1.tif"
        if not s2_path.exists():
            log.warning("skipping %s (no S2 composite)", cell)
            continue
        has_s1 = s1_path.exists()
        if not has_s1:
            log.warning("%s: no S1 composite -- S2 arm alone for this cell", cell)
        with rasterio.open(s2_path) as src:
            src1 = rasterio.open(s1_path) if has_s1 else None
            H, W, transform, crs = src.height, src.width, src.transform, src.crs
            acc = {v: np.zeros((H, W), "float32") for v in out_dirs}
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
                    p2 = predict(*tasks["s2"], s2_np, None)
                    if has_s1:
                        s1_np = src1.read(window=win, boundless=True, fill_value=0).astype("float32")[:2]
                        p1 = predict(*tasks["s1"], s2_np, s1_np)
                        variants = {"s1": p1, "s2": p2,
                                    "gated": p1 * (0.5 + 0.5 * p2), "mean": (p1 + p2) / 2}
                    else:
                        variants = {v: p2 for v in out_dirs}
                    for v, prob in variants.items():
                        acc[v][r:r+h, c:c+w] += prob[:h, :w] * hann[:h, :w]
                    wacc[r:r+h, c:c+w] += hann[:h, :w]
                    valid_any[r:r+h, c:c+w] |= (s2_np[:, :h, :w] > 0).any(axis=0)
            if src1 is not None:
                src1.close()
        for v, d in out_dirs.items():
            prob_full = np.where(wacc > 0, acc[v] / np.maximum(wacc, 1e-6), 0.0)
            prob_full[~valid_any] = 0.0
            with rasterio.open(d / f"{cell}.tif", "w", driver="GTiff", width=W, height=H,
                               count=1, dtype="uint8", crs=crs, transform=transform,
                               compress="deflate", predictor=2) as dst:
                dst.write((np.clip(prob_full, 0, 1) * 255).astype("uint8"), 1)
        if i % 25 == 0 or i == len(cells):
            log.info("[%d/%d] %s", i, len(cells), cell)


if __name__ == "__main__":
    main()
