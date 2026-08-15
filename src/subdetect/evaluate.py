"""Evaluate a trained model on held-out chips.

Reports pixel IoU/F1 and, more importantly for recall-first triage, per-installation
recall bucketed by substation area and by OSM voltage class: an installation counts as
detected if any predicted positive pixel overlaps its polygon.

Handles both heads: the checkpoint's task type is sniffed (see infer.sniff_task_type), so
a fraction-regression checkpoint is scored on the same buckets, thresholding predicted
coverage instead of class probability. Per-installation recall by area bucket is the
cross-head comparison that means something; see the note on pixel IoU below.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
from rasterio import features as rio_features
from shapely.geometry import box

from subdetect.config import (
    S1_MEAN,
    S1_OFFSET_DB,
    S1_SCALE,
    S1_STD,
    Settings,
    geodesic_area_m2,
    resolve_aoi,
)
from subdetect.local_source import load_substation_labels

log = logging.getLogger(__name__)

AREA_BUCKETS = [(1000, 2000), (2000, 5000), (5000, 20000), (20000, np.inf)]
VOLT_BUCKETS = [(">=220kV", 220000, np.inf), ("66-220kV", 66000, 220000), ("<66kV/unknown", 0, 66000)]
_S1_MEAN = np.array(S1_MEAN, dtype="float32")[:, None, None]
_S1_STD = np.array(S1_STD, dtype="float32")[:, None, None]


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score of random positive > score of random negative)."""
    from scipy.stats import rankdata

    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_at(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    order = np.argsort(-scores, kind="stable")
    top = order[: min(k, len(order))]
    return float(y[top].mean()) if len(top) else float("nan")


def _load_s1(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        dn = src.read().astype("float32")
    nodata = dn <= 0
    x = (dn / S1_SCALE - S1_OFFSET_DB - _S1_MEAN) / _S1_STD
    x[nodata] = 0.0
    return x


def evaluate(aoi: str, checkpoint: Path, chips_dir: Path, threshold: float = 0.3,
             min_area_m2: float | None = None) -> pd.DataFrame:
    """`min_area_m2` overrides the settings label floor for which installations count
    as ground truth here (0 = every substation polygon, matching a no-floor training
    run); the settings floor otherwise silently gates out AREA_BUCKETS below it."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import torch

    from subdetect.infer import is_regression, load_model

    settings = Settings.load()
    if min_area_m2 is not None:
        settings.min_sub_area_m2 = float(min_area_m2)
    _, cfg = resolve_aoi(aoi, settings)
    labels = load_substation_labels(Path("data/labels") / aoi, settings.min_sub_area_m2)
    labels = labels[labels.role == "pos"].reset_index(drop=True)

    index = pd.read_parquet(Path(chips_dir) / aoi / "index.parquet")
    val = index[index.split == "val"]
    if val.empty:
        val = index.sample(frac=0.2, random_state=42)
    log.info("Evaluating on %d val chips", len(val))

    task, device = load_model(checkpoint)
    regression = is_regression(task)
    mods = task.hparams.get("model_args", {}).get("backbone_modalities", ["S2L2A"])
    dual = "S1RTC" in mods
    log.info("Checkpoint head: %s", "fraction (regression)" if regression else "segmentation")

    tp = fp = fn = 0
    det, mis = [], []  # (area, voltage) of GT installations
    for _, row in val.iterrows():
        with rasterio.open(row["image"]) as src:
            arr = src.read().astype("float32")
            transform, crs, shape = src.transform, src.crs, (src.height, src.width)
            chip_geo = box(*rasterio.warp.transform_bounds(crs, "EPSG:4326", *src.bounds))
        s2 = torch.from_numpy(arr / 10000.0)[None].to(device)
        if dual:
            s1 = torch.from_numpy(_load_s1(row["s1"]))[None].to(device)
            x = {m: (s2 if m == "S2L2A" else s1) for m in mods}
        else:
            x = s2
        with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
            out = task(x)
            logits = out.output if hasattr(out, "output") else out
            score = (logits[0] if logits.ndim == 3 else logits[0, 0]) if regression \
                else torch.softmax(logits, 1)[0, 1]
            pred = (score >= threshold).cpu().numpy()

        with rasterio.open(row["mask"]) as src:
            gt = src.read(1)
        if gt.dtype.kind == "f":
            # Fraction mask: a pixel is ground-truth positive when the substation covers
            # most of it, background when it covers none, and neither when it is a partial
            # boundary pixel -- scoring those as background would penalise exactly the
            # sub-pixel resolution the fraction target exists to supply, so they join the
            # ignore set. Pixel IoU stays comparable to a binary run only in the sense that
            # both count confidently-labelled pixels; the per-installation recall below is
            # the metric to compare across heads.
            pos, valid = gt >= 0.5, (gt >= 0.5) | (gt == 0)
        else:
            pos, valid = gt == 1, gt != -1
        tp += int((pred & pos & valid).sum())
        fp += int((pred & ~pos & valid).sum())
        fn += int((~pred & pos & valid).sum())

        here = labels[labels.geometry.intersects(chip_geo)]
        for _, inst in here.iterrows():
            area = inst.area_m2 if inst.area_m2 > 0 else geodesic_area_m2(inst.geometry)
            poly = gpd.GeoSeries([inst.geometry], crs="EPSG:4326").to_crs(crs).iloc[0]
            im = rio_features.rasterize([(poly, 1)], out_shape=shape, transform=transform,
                                        all_touched=True, dtype="uint8").astype(bool)
            if not im.any():
                continue
            (det if (pred & im).any() else mis).append((area, inst.voltage_v))

    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    log.info("Pixel IoU=%.3f F1=%.3f (tp=%d fp=%d fn=%d)", iou, f1, tp, fp, fn)

    da = pd.Series([a for a, _ in det], dtype=float)
    ma = pd.Series([a for a, _ in mis], dtype=float)
    dv = pd.Series([v for _, v in det], dtype=float)
    mv = pd.Series([v for _, v in mis], dtype=float)
    rows = []
    for lo, hi in AREA_BUCKETS:
        name = f"{lo}-{'inf' if hi == np.inf else int(hi)} m2"
        nd = int(((da >= lo) & (da < hi)).sum())
        nm = int(((ma >= lo) & (ma < hi)).sum())
        tot = nd + nm
        rows.append(dict(group=name, installations=tot, detected=nd,
                         recall=round(nd / tot, 3) if tot else float("nan")))
    for name, lo, hi in VOLT_BUCKETS:
        dsel = (dv >= lo) & (dv < hi) if lo > 0 else ((dv < hi) | dv.isna())
        msel = (mv >= lo) & (mv < hi) if lo > 0 else ((mv < hi) | mv.isna())
        nd, nm = int(dsel.sum()), int(msel.sum())
        tot = nd + nm
        rows.append(dict(group=name, installations=tot, detected=nd,
                         recall=round(nd / tot, 3) if tot else float("nan")))
    report = pd.DataFrame(rows)
    report.attrs["pixel_iou"] = iou
    report.attrs["pixel_f1"] = f1
    log.info("Per-installation recall:\n%s", report.to_string(index=False))
    return report
