"""sindh_test prob raster for v13_towerprior: single 3-modality model (S2L2A + S1RTC +
power_tower_prior), NOT a post-hoc fusion like v9/v11/v12 -- the tower prior is a model
input, so there's only one arm to score. Compare directly against v5_fusion (run-name
v5_fusion, prob dir sindh_test_v5fused) in field_eval.py: same arch/loss/schedule, only
the extra modality differs, per configs/terramind_sub_v13_towerprior.yaml's header.

The tower-prior channel is derived on the fly from each cell's composite_s1.tif via
tower_prior.tower_prior_dn (pure post-processing, matches how chips.py/compose.py built
it for training) rather than requiring a precomputed composite_tower.tif for sindh_test.
Computed once per cell (full-extent distance transform), then windowed + decoded to
[0, 1] float per tile exactly like datamodule._load_tower.

Writes:
  sindh_test_v13/prob   v13_towerprior single arm

Usage:
  pixi run -e ml python scripts/infer_sindh_test_v13.py --checkpoint data/models/stageA_v13_towerprior/<best>.ckpt
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
log = logging.getLogger("infer_sindh_test_v13")

WIN, STRIDE = 224, 104
_TOWER_DN_SCALE = 65534.0

REGION = ROOT / "data/osmose_regions/sindh_test"
OUT = ROOT / "data/osmose_regions/sindh_test_v13/prob"


def _decode_tower(dn: np.ndarray) -> np.ndarray:
    """(1, H, W) uint16 DN -> [0, 1] float32, mirrors datamodule._load_tower."""
    dn = dn.astype("float32")
    nodata = dn <= 0
    x = np.clip((dn - 1.0) / _TOWER_DN_SCALE, 0.0, 1.0)
    x[nodata] = 0.0
    return x


def main() -> None:
    import torch

    from subdetect.infer import _standardize_s1, load_model
    from subdetect.tower_prior import tower_prior_dn

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="v13_towerprior checkpoint")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    task, device = load_model(Path(a.checkpoint))
    mods = task.hparams.get("model_args", {}).get("backbone_modalities", [])
    mods = [m if isinstance(m, str) else next(iter(m)) for m in mods]
    log.info("modalities=%s", mods)
    hann = np.outer(np.hanning(WIN), np.hanning(WIN)).astype("float32") + 1e-3

    def predict(s2_np, s1_np, tower_np):
        s2_t = torch.from_numpy(s2_np / 10000.0)[None].to(device)
        s1_t = torch.from_numpy(_standardize_s1(s1_np))[None].to(device)
        tower_t = torch.from_numpy(tower_np)[None].to(device)
        x = {"S2L2A": s2_t, "S1RTC": s1_t, "power_tower_prior": tower_t}
        x = {m: x[m] for m in mods}
        with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
            out = task(x)
            logits = out.output if hasattr(out, "output") else out
            return torch.softmax(logits, dim=1)[0, 1].float().cpu().numpy()

    comp_dir = REGION / "composites"
    cells = sorted(p.name for p in comp_dir.iterdir() if p.is_dir())
    log.info("%d cells", len(cells))
    for i, cell in enumerate(cells, 1):
        out_tif = OUT / f"{cell}.tif"
        if out_tif.exists():
            continue
        s2_path = comp_dir / cell / "composite_0.tif"
        s1_path = comp_dir / cell / "composite_s1.tif"
        if not (s2_path.exists() and s1_path.exists()):
            log.warning("skipping %s (missing composite)", cell)
            continue
        with rasterio.open(s2_path) as src, rasterio.open(s1_path) as src1:
            H, W, transform, crs = src.height, src.width, src.transform, src.crs
            s1_full = src1.read().astype("float32")[:2]
            tower_dn_full = tower_prior_dn(s1_full)  # (1, H, W) uint16, full-cell distance transform
            acc = np.zeros((H, W), "float32")
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
                    tower_np = _decode_tower(tower_dn_full[:, r:r + WIN, c:c + WIN])
                    if tower_np.shape[-2:] != (WIN, WIN):
                        pad = np.zeros((1, WIN, WIN), "float32")
                        pad[:, :tower_np.shape[-2], :tower_np.shape[-1]] = tower_np
                        tower_np = pad
                    prob = predict(s2_np, s1_np, tower_np)
                    acc[r:r+h, c:c+w] += prob[:h, :w] * hann[:h, :w]
                    wacc[r:r+h, c:c+w] += hann[:h, :w]
                    valid_any[r:r+h, c:c+w] |= (s2_np[:, :h, :w] > 0).any(axis=0)
        prob_full = np.where(wacc > 0, acc / np.maximum(wacc, 1e-6), 0.0)
        prob_full[~valid_any] = 0.0
        with rasterio.open(out_tif, "w", driver="GTiff", width=W, height=H,
                           count=1, dtype="uint8", crs=crs, transform=transform,
                           compress="deflate", predictor=2) as dst:
            dst.write((np.clip(prob_full, 0, 1) * 255).astype("uint8"), 1)
        if i % 25 == 0 or i == len(cells):
            log.info("[%d/%d] %s", i, len(cells), cell)


if __name__ == "__main__":
    main()
