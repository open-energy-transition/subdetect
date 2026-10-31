"""Arm-alone sindh_test prob rasters for any list of checkpoints, plus mean fusion.

Same windowing as infer_sindh_test_v12.py (224 px windows, stride 104, Hann blending,
valid-pixel mask) so rows are comparable with the historical sindh_test_<v> runs, but
generic: each `--arm NAME=CKPT[:UPSAMPLE]` writes data/osmose_regions/sindh_test_<NAME>/prob,
reading every window once for all arms.

`fuse` averages two existing arm-alone prob dirs cell by cell (uint8 in, uint8 out) --
the deployed mean fusion 0.5*(P_S1 + P_S2). Both arms of an A/B are fused through this
same path, so the <=1/255 quantisation it adds is identical on both sides.

Usage:
  pixi run -e ml python scripts/infer_sindh_test_ckpts.py infer \
      --arm s1ctl_s0=data/models/stageA_v17_ctl_s1_seed0/<best>.ckpt --arm ...
  pixi run -e ml python scripts/infer_sindh_test_ckpts.py fuse --a s1ctl_s0 --b v12 --out s1ctl_s0_x_v12
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
log = logging.getLogger("infer_sindh_test_ckpts")
WIN, STRIDE = 224, 104
REGION = ROOT / "data/osmose_regions/sindh_test"
REG = ROOT / "data/osmose_regions"


def _write(path: Path, arr: np.ndarray, crs, transform) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0],
                       count=1, dtype="uint8", crs=crs, transform=transform,
                       compress="deflate", predictor=2) as dst:
        dst.write(arr, 1)


def cmd_infer(a) -> None:
    from subdetect.infer import load_model, predict_window

    arms = {}
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        ckpt, up = (rest.rsplit(":", 1) + ["1"])[:2] if rest.count(":") else (rest, "1")
        task, device = load_model(Path(ckpt))
        mods = task.hparams.get("model_args", {}).get("backbone_modalities", ["S2L2A"])
        arms[name] = (task, mods, device, int(up), REG / f"sindh_test_{name}" / "prob")
        log.info("arm %s: %s modalities=%s upsample=%s", name, ckpt, mods, up)
    hann = np.outer(np.hanning(WIN), np.hanning(WIN)).astype("float32") + 1e-3
    comp_dir = REGION / "composites"
    cells = sorted(p.name for p in comp_dir.iterdir() if p.is_dir())
    for i, cell in enumerate(cells, 1):
        if all((v[4] / f"{cell}.tif").exists() for v in arms.values()):
            continue
        s2_path, s1_path = comp_dir / cell / "composite_0.tif", comp_dir / cell / "composite_s1.tif"
        if not (s2_path.exists() and s1_path.exists()):
            log.warning("skipping %s (missing composite)", cell)
            continue
        with rasterio.open(s2_path) as src, rasterio.open(s1_path) as src1:
            H, W, transform, crs = src.height, src.width, src.transform, src.crs
            acc = {k: np.zeros((H, W), "float32") for k in arms}
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
                    for k, (task, mods, device, up, _) in arms.items():
                        p = predict_window(task, mods, device, s2_np, s1_np, upsample=up)
                        acc[k][r:r + h, c:c + w] += p[:h, :w] * hann[:h, :w]
                    wacc[r:r + h, c:c + w] += hann[:h, :w]
                    valid_any[r:r + h, c:c + w] |= (s2_np[:, :h, :w] > 0).any(axis=0)
        for k, v in arms.items():
            prob = np.where(wacc > 0, acc[k] / np.maximum(wacc, 1e-6), 0.0)
            prob[~valid_any] = 0.0
            _write(v[4] / f"{cell}.tif", (np.clip(prob, 0, 1) * 255).astype("uint8"), crs, transform)
        if i % 25 == 0 or i == len(cells):
            log.info("[%d/%d] %s", i, len(cells), cell)


def cmd_fuse(a) -> None:
    da, db = REG / f"sindh_test_{a.a}" / "prob", REG / f"sindh_test_{a.b}" / "prob"
    out = REG / f"sindh_test_{a.out}" / "prob"
    n = 0
    for fa in sorted(da.glob("*.tif")):
        fb = db / fa.name
        if not fb.exists():
            continue
        with rasterio.open(fa) as sa, rasterio.open(fb) as sb:
            x = (sa.read(1).astype(np.float32) + sb.read(1).astype(np.float32)) / 2.0
            _write(out / fa.name, np.round(x).astype("uint8"), sa.crs, sa.transform)
        n += 1
    log.info("fused %d cells -> %s", n, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("infer")
    p1.add_argument("--arm", action="append", required=True, help="NAME=CKPT[:UPSAMPLE]")
    p2 = sub.add_parser("fuse")
    p2.add_argument("--a", required=True)
    p2.add_argument("--b", required=True)
    p2.add_argument("--out", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"infer": cmd_infer, "fuse": cmd_fuse}[a.cmd](a)


if __name__ == "__main__":
    main()
