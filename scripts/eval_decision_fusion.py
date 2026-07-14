"""Decision-level S1+S2 fusion: combine the two single-modality v4 arms' probability
maps on val chips and compare against each alone. No training required.

Combos: geometric mean, arithmetic mean, max, and s1-gated (S1 prob softly modulated
by S2 agreement). Reports pixel IoU at thr 0.3 plus best-threshold IoU, and
>=20k m2 installation recall at thr 0.3.
"""
import sys, logging
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.warp
import torch
from rasterio import features as rio_features
from shapely.geometry import box

sys.path.insert(0, "src")
from subdetect.local_source import load_substation_labels
from subdetect.config import geodesic_area_m2
from subdetect.evaluate import _load_s1
from subdetect.infer import load_model

logging.disable(logging.INFO)
CKPTS = {
    "s1only": "data/models/stageA_v4_s1only/terramind-sub-epoch=15-step=1376.ckpt",
    "s2only": "data/models/stageA_v4_s2only/terramind-sub-epoch=23-step=2064.ckpt",
}

idx = pd.read_parquet("data/chips/pakistan/index.parquet")
val = idx[idx.split == "val"]
labels = load_substation_labels(Path("data/labels/pakistan"), min_area_m2=20000.0)
big = labels[labels.role == "pos"]

# predict both models on all val chips once
probs = {k: [] for k in CKPTS}
gts, geo = [], []
for arm, ckpt in CKPTS.items():
    task, device = load_model(Path(ckpt))
    mods = task.hparams.get("model_args", {}).get("backbone_modalities", ["S2L2A"])
    dual = "S1RTC" in mods
    for _, row in val.iterrows():
        with rasterio.open(row["image"]) as src:
            arr = src.read().astype("float32")
            transform, crs, shape = src.transform, src.crs, (src.height, src.width)
            chip_geo = box(*rasterio.warp.transform_bounds(crs, "EPSG:4326", *src.bounds))
        s2 = torch.from_numpy(arr / 10000.0)[None].to(device)
        x = {m: (s2 if m == "S2L2A" else torch.from_numpy(_load_s1(row["s1"]))[None].to(device))
             for m in mods} if dual else s2
        with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
            out = task(x)
            logits = out.output if hasattr(out, "output") else out
            probs[arm].append(torch.softmax(logits, 1)[0, 1].float().cpu().numpy())
        if arm == "s1only":
            with rasterio.open(row["mask"]) as src:
                gts.append(src.read(1))
            geo.append((transform, crs, shape, chip_geo))
    del task
    torch.cuda.empty_cache()

def metrics(pmaps, thr=0.3):
    tp = fp = fn = 0
    det = mis = 0
    for p, gt, (transform, crs, shape, chip_geo) in zip(pmaps, gts, geo):
        pred = p >= thr
        valid = gt != -1
        tp += int((pred & (gt == 1) & valid).sum())
        fp += int((pred & (gt == 0) & valid).sum())
        fn += int((~pred & (gt == 1) & valid).sum())
        for _, inst in big[big.geometry.intersects(chip_geo)].iterrows():
            poly = gpd.GeoSeries([inst.geometry], crs="EPSG:4326").to_crs(crs).iloc[0]
            im = rio_features.rasterize([(poly, 1)], out_shape=shape, transform=transform,
                                        all_touched=True, dtype="uint8").astype(bool)
            if im.any():
                if (pred & im).any():
                    det += 1
                else:
                    mis += 1
    iou = tp / max(tp + fp + fn, 1)
    return iou, det, det + mis

combos = {
    "s1only": probs["s1only"],
    "s2only": probs["s2only"],
    "geometric": [np.sqrt(a * b) for a, b in zip(probs["s1only"], probs["s2only"])],
    "mean": [(a + b) / 2 for a, b in zip(probs["s1only"], probs["s2only"])],
    "max": [np.maximum(a, b) for a, b in zip(probs["s1only"], probs["s2only"])],
    "s1_gated": [a * (0.5 + 0.5 * b) for a, b in zip(probs["s1only"], probs["s2only"])],
}
print(f"{'combo':<12}{'IoU@0.3':>9}{'bestIoU':>9}{'@thr':>6}{'recall>=20k':>13}")
for name, pm in combos.items():
    iou03, det, n = metrics(pm, 0.3)
    best, bthr = max((metrics(pm, t)[0], t) for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    print(f"{name:<12}{iou03:>9.3f}{best:>9.3f}{bthr:>6.1f}{f'{det}/{n}':>13}")
