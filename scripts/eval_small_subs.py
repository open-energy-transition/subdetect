"""Do the v5 arms detect substations BELOW the 20k m2 training floor?

Small subs are ignore(-1) in training masks and absent from the standard eval.
Here: load all substation polygons (min_area=0), keep those < 20k m2 that
intersect val chips, and count per-model whether any predicted pixel falls
inside each. Detection criterion matches evaluate.py ((pred & poly).any()).
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
THRESHOLD = 0.3
ARMS = {
    "s1only": "data/models/stageA_v5_s1only/terramind-sub-epoch=24-step=6975.ckpt",
    "fusion": "data/models/stageA_v5_s1fusion/terramind-sub-epoch=45-step=12834.ckpt",
    "s2only": "data/models/stageA_v5_s2only/terramind-sub-epoch=40-step=11439.ckpt",
}
BUCKETS = [(1000, 2000), (2000, 5000), (5000, 10000), (10000, 20000)]

idx = pd.read_parquet("data/chips_v5/pakistan/index.parquet")
val = idx[idx.split == "val"]
labels = load_substation_labels(Path("data/labels/pakistan"), min_area_m2=0.0)
poly = labels[labels.role.isin(["pos", "small"])].copy()
poly["area"] = [a if a > 0 else geodesic_area_m2(g)
                for a, g in zip(poly.area_m2, poly.geometry)]
small = poly[poly.area < 20000]
print(f"{len(val)} val chips; {len(small)} small (<20k m2) sub polygons in AOI")

results = {}
for arm, ckpt in ARMS.items():
    task, device = load_model(Path(ckpt))
    mods = task.hparams.get("model_args", {}).get("backbone_modalities", ["S2L2A"])
    dual = "S1RTC" in mods
    det = []  # (area, hit)
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
            pred = (torch.softmax(logits, 1)[0, 1] >= THRESHOLD).cpu().numpy()
        here = small[small.geometry.intersects(chip_geo)]
        for _, inst in here.iterrows():
            p = gpd.GeoSeries([inst.geometry], crs="EPSG:4326").to_crs(crs).iloc[0]
            im = rio_features.rasterize([(p, 1)], out_shape=shape, transform=transform,
                                        all_touched=True, dtype="uint8").astype(bool)
            if im.any():
                det.append((inst.area, bool((pred & im).any())))
    results[arm] = det
    del task
    torch.cuda.empty_cache()

n = len(next(iter(results.values())))
print(f"\n{n} small-substation instances on val chips\n")
print(f"{'bucket':<16}" + "".join(f"{a:>10}" for a in ARMS))
for lo, hi in BUCKETS:
    line = f"{lo//1000}-{hi//1000}k m2".ljust(16)
    for arm in ARMS:
        d = [(a, h) for a, h in results[arm] if lo <= a < hi]
        line += (f"{sum(h for _, h in d)}/{len(d)}".rjust(10)) if d else f"{'—':>10}"
    print(line)
line = "all <20k".ljust(16)
for arm in ARMS:
    d = results[arm]
    line += f"{sum(h for _, h in d)}/{len(d)}".rjust(10)
print(line)
