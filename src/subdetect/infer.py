"""Tiled inference over an AOI using the local composites.

For each composite cell, slides a 224 px window across its native UTM grid and overlap-
adds the predictions with a 2D Hann taper into one seamless probability raster (uint8) per
cell. A patch-size-coprime stride + Hann blending suppress window-seam / patch-grid
artefacts. When the checkpoint is dual-modality (backbone_modalities includes S1RTC), each
window is fed as {"S2L2A": ..., "S1RTC": ...} from the co-registered composite_s1.tif.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from subdetect.config import (
    CHIP_SIZE,
    MODEL_BANDS,
    S1_MEAN,
    S1_OFFSET_DB,
    S1_SCALE,
    S1_STD,
    Settings,
    resolve_aoi,
)
from subdetect.local_source import CompositeIndex

log = logging.getLogger(__name__)
_S1_MEAN = np.array(S1_MEAN, dtype="float32")[:, None, None]
_S1_STD = np.array(S1_STD, dtype="float32")[:, None, None]


def load_model(checkpoint: Path):
    import torch
    from terratorch.tasks import SemanticSegmentationTask

    # Recall-tuned checkpoints pickle a TverskyLoss criterion; PyTorch 2.6 defaults
    # weights_only=True, which rejects it. Allowlist our own class.
    try:
        import segmentation_models_pytorch as smp

        torch.serialization.add_safe_globals([smp.losses.TverskyLoss])
    except Exception:  # noqa: BLE001
        pass
    task = SemanticSegmentationTask.load_from_checkpoint(checkpoint, map_location="cpu")
    task.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return task.to(device), device


def _is_dual(task) -> bool:
    try:
        mods = task.hparams.get("model_args", {}).get("backbone_modalities", [])
        return "S1RTC" in mods
    except Exception:  # noqa: BLE001
        return False


def _standardize_s1(arr: np.ndarray) -> np.ndarray:
    nodata = arr <= 0
    db = arr.astype("float32") / S1_SCALE - S1_OFFSET_DB
    x = (db - _S1_MEAN) / _S1_STD
    x[nodata] = 0.0
    return x


def run_inference(
    aoi: str, checkpoint: Path, out_dir: Path, only_built: bool = True, limit: int = 0
) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import torch

    settings = Settings.load()
    _, cfg = resolve_aoi(aoi, settings)
    composed = Path("data/composites") / aoi
    if not (composed.exists() and any(composed.glob("composites/*/composite_0.tif"))):
        raise FileNotFoundError(f"No composites under {composed}; run compose --aoi {aoi} first")
    comp_idx = CompositeIndex(composed)
    task, device = load_model(checkpoint)
    dual = _is_dual(task)
    log.info("Inference on %s: %d cells, modalities=%s", aoi, len(comp_idx.index),
             "S2+S1" if dual else "S2")

    if dual:
        missing = [p for p in comp_idx.index.path
                   if not (Path(p).parent / "composite_s1.tif").exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} cells missing composite_s1.tif (e.g. {missing[:3]}); "
                "run compose --sensor s1 first"
            )
    out_dir = Path(out_dir) / aoi / "prob"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_bands = len(MODEL_BANDS)
    stride = 104  # NOT a multiple of the 16 px ViT patch: decorrelates patch-edge artefacts
    hann = np.outer(np.hanning(CHIP_SIZE), np.hanning(CHIP_SIZE)).astype("float32") + 1e-3

    tiles = list(comp_idx.index.path)
    if limit:
        tiles = tiles[:limit]
    windows_run = 0
    for tile_path in tqdm(tiles, desc="cells"):
        tile = Path(tile_path).parent.name
        out_tif = out_dir / f"{tile}.tif"
        if out_tif.exists():
            continue
        src1 = rasterio.open(Path(tile_path).parent / "composite_s1.tif") if dual else None
        with rasterio.open(tile_path) as src:
            H, W = src.height, src.width
            transform, crs = src.transform, src.crs
            acc = np.zeros((H, W), dtype="float32")
            wacc = np.zeros((H, W), dtype="float32")
            valid_any = np.zeros((H, W), dtype=bool)
            rows = sorted(set(list(range(0, max(H - CHIP_SIZE, 0) + 1, stride)) + [max(H - CHIP_SIZE, 0)]))
            cols = sorted(set(list(range(0, max(W - CHIP_SIZE, 0) + 1, stride)) + [max(W - CHIP_SIZE, 0)]))
            for r in rows:
                for c in cols:
                    win = Window(c, r, CHIP_SIZE, CHIP_SIZE)
                    arr = src.read(window=win, boundless=True, fill_value=0)[:n_bands]
                    h, w = min(CHIP_SIZE, H - r), min(CHIP_SIZE, W - c)
                    if (arr[:, :h, :w] > 0).mean() < 0.2:  # skip mostly-nodata windows
                        continue
                    s2 = torch.from_numpy(arr.astype("float32") / 10000.0)[None].to(device)
                    if src1 is not None:
                        s1raw = src1.read(window=win, boundless=True, fill_value=0)[:2]
                        s1 = torch.from_numpy(_standardize_s1(s1raw))[None].to(device)
                        x = {"S2L2A": s2, "S1RTC": s1}
                    else:
                        x = s2
                    with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
                        out = task(x)
                        logits = out.output if hasattr(out, "output") else out
                        prob = torch.softmax(logits, dim=1)[0, 1].float().cpu().numpy()
                    acc[r : r + h, c : c + w] += prob[:h, :w] * hann[:h, :w]
                    wacc[r : r + h, c : c + w] += hann[:h, :w]
                    valid_any[r : r + h, c : c + w] |= (arr[:, :h, :w] > 0).any(axis=0)
                    windows_run += 1
        prob_full = np.where(wacc > 0, acc / np.maximum(wacc, 1e-6), 0.0)
        prob_full[~valid_any] = 0.0
        with rasterio.open(
            out_tif, "w", driver="GTiff", width=W, height=H, count=1, dtype="uint8",
            crs=crs, transform=transform, compress="deflate", predictor=2,
        ) as dst:
            dst.write((np.clip(prob_full, 0, 1) * 255).astype("uint8"), 1)
        if src1 is not None:
            src1.close()
    log.info("Inference wrote %d cell rasters (%d windows) -> %s", len(tiles), windows_run, out_dir)
    return out_dir
