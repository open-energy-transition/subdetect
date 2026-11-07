"""Training indexes and configs for the v18 arms (2026-11-05; docs/issues/v18-improvements.md).

Arms, each one variable away from its control, all on earthpv's v22 recipe (fixed budget,
1,000-step warmup then cosine to 1e-6, weight EMA, radiometric jitter, seed 1; the model is
the final weights):
  s2_recipe  v16 corpus (= v17 S2 control's), single-date S2      vs v17 ctl seeds (old recipe)
  s2_1d      the 27,965 chips the multi-date build kept, single-date S2, batch 4 x 4
  s2_4d      the same chips, 4 dates through TemporalWrapper max   vs s2_1d
  s1_recipe  v5 S1 corpus (= v17 S1 control's)                    vs v17 S1 ctl seeds
  s1_big     s1_recipe's chips plus every v16 chip with fetched S1  vs s1_recipe

Every training row whose chip footprint meets a test region (sindh_test, gujarat_test) is
dropped: TorchGeo puts chips on real substations everywhere (26 inside the Sindh holdout).
TorchGeo chips are stored south-up under a north-up transform; the multi-date stacks were
fetched north-up on that transform, so s2_4d points TorchGeo rows at north-up copies of their
masks (`mask[::-1]`). Single-date arms keep the stored (south-up image + mask) pairs.

Usage:
  pixi run python scripts/build_v18_indexes.py s2      # s2_recipe, s2_1d, s2_4d + configs
  pixi run python scripts/build_v18_indexes.py s1      # after fetch_s1_chips.py: s1_recipe, s1_big
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
log = logging.getLogger("build_v18_indexes")

OUT = ROOT / "data/v18"  # -> /home/tobi/subdetect-data/v18
V16 = ROOT / "data/chips_v16_honestburn/combined/index.parquet"
V5HB = ROOT / "data/chips_v5_honestburn/combined/index.parquet"
MD = ROOT / "data/chips_multidate_v11b/index.parquet"
S1X = ROOT / "data/chips_s1_extra/index.parquet"
REG = ROOT / "data/osmose_regions"
TEST_REGIONS = ("sindh_test", "gujarat_test")
PAD_DEG = 0.025          # chip half-width is 1.12 km; pad cells by ~2.5 km
FLIPPED = "torchgeo_substation"

BASE_TASK = {
    "model_factory": "EncoderDecoderFactory",
    "model_args": {
        "backbone": "terramind_v1_tiny", "backbone_pretrained": True,
        "necks": [{"name": "SelectIndices", "indices": [2, 5, 8, 11]},
                  {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                  {"name": "LearnedInterpolateToPyramidal"}],
        "decoder": "UNetDecoder", "decoder_channels": [256, 128, 64, 32],
        "num_classes": 2, "rescale": True,
    },
    "loss": "tversky", "tversky_args": {"alpha": 0.35, "beta": 0.65, "gamma": 1.33},
    "ignore_index": -1, "lr": 1.0e-4, "optimizer": "AdamW",
    "optimizer_hparams": {"weight_decay": 0.05},
    "freeze_backbone": False, "plot_on_val": False,
}
S2_BANDS = ["BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3",
            "NIR_BROAD", "NIR_NARROW", "SWIR_1", "SWIR_2"]
WARMUP = 1000


def test_region_union():
    geoms = []
    for r in TEST_REGIONS:
        cj = REG / r / "cells.json"
        if cj.exists():
            geoms += [box(*b).buffer(PAD_DEG) for b in json.loads(cj.read_text())["cells"].values()]
            continue
        comp = REG / r / "composites"
        if not comp.exists():
            raise SystemExit(f"test region {r} has neither cells.json nor composites")
        for c in sorted(p for p in comp.iterdir() if p.is_dir()):
            with rasterio.open(c / "composite_0.tif") as src:
                geoms.append(box(*transform_bounds(src.crs, "EPSG:4326", *src.bounds))
                             .buffer(PAD_DEG))
    return unary_union(geoms)


def drop_leaks(d: pd.DataFrame, region, name: str) -> pd.DataFrame:
    from shapely import points

    hit = region.contains(points(d["lon"].to_numpy(), d["lat"].to_numpy())) if len(d) else []
    hit = np.asarray(hit) & (d["split"].to_numpy() == "train")
    log.info("%s: dropping %d training chips inside a test region (%s)", name, int(hit.sum()),
             d[hit].groupby("aoi").size().to_dict())
    return d[~hit].reset_index(drop=True)


def northup_masks(rows: pd.DataFrame) -> pd.Series:
    """Write mask[::-1] for TorchGeo rows (their stored masks are south-up); return paths."""
    out_dir = OUT / "masks_northup"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for _, r in rows.iterrows():
        dst = out_dir / f"{r['chip_id']}.tif"
        if not dst.exists():
            with rasterio.open(ROOT / r["mask"]) as src:
                arr, prof = src.read(1), src.profile
            tmp = dst.with_name(dst.name + ".tmp")
            with rasterio.open(tmp, "w", **prof) as w:
                w.write(arr[::-1], 1)
            tmp.rename(dst)
        paths.append(str(dst))
    return pd.Series(paths, index=rows.index)


def steps_per_epoch(n_train: int, batch: int, accum: int) -> int:
    return math.ceil(math.ceil(n_train / batch) / accum)


def write_config(name: str, index: Path, n_train: int, *, modalities, batch: int, accum: int,
                 epochs: int, ema: float, n_revisits: int = 1, note: str = "") -> None:
    total = steps_per_epoch(n_train, batch, accum) * epochs
    task = json.loads(json.dumps(BASE_TASK))
    ma = task["model_args"]
    ma["backbone_modalities"] = list(modalities)
    ma["backbone_bands"] = ({"S1RTC": ["VV", "VH"]} if modalities == ["S1RTC"]
                            else {"S2L2A": S2_BANDS})
    if n_revisits > 1:
        ma["backbone_use_temporal"] = True
        ma["backbone_temporal_pooling"] = "max"
    task["scheduler"] = "SequentialLR"
    task["scheduler_hparams"] = {
        "interval": "step",
        "schedulers": {"LinearLR": {"start_factor": 0.01, "end_factor": 1.0,
                                    "total_iters": WARMUP},
                       "CosineAnnealingLR": {"T_max": total - WARMUP, "eta_min": 1.0e-6}},
        "milestones": [WARMUP],
    }
    data = {"index_path": str(index.relative_to(ROOT)), "batch_size": batch, "num_workers": 4,
            "modalities": list(modalities), "drop_page_cache": True,
            "jitter": {"p": 0.5, "gain": 0.07, "offset": 0.01}}
    if n_revisits > 1:
        data["n_revisits"] = n_revisits
    cfg = {"data": data, "task": task,
           "trainer": {"max_epochs": epochs, "accumulate_grad_batches": accum},
           "seed": 1, "early_stopping": False, "ema": {"decay": ema},
           "checkpoint_dir": f"data/models/v18_{name}", "monitor": "val/IoU_1",
           "monitor_mode": "max"}
    head = (f"# v18 arm {name} (scripts/build_v18_indexes.py; docs/issues/v18-improvements.md).\n"
            f"# earthpv's v22 recipe: {epochs} epochs x {total // epochs} optimizer steps = "
            f"{total} steps, {WARMUP} warmup,\n# cosine to 1e-6, EMA {ema}, jitter, seed 1; "
            f"the model is the final weights (last.ckpt).\n# {note}\n")
    path = ROOT / "configs" / f"terramind_sub_v18_{name}.yaml"
    path.write_text(head + yaml.safe_dump(cfg, sort_keys=False))
    (ROOT / "data/models").mkdir(exist_ok=True)
    log.info("%s: %d train rows, %d steps -> %s", name, n_train, total, path.name)


def cmd_s2(_a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    region = test_region_union()
    v16 = drop_leaks(pd.read_parquet(V16), region, "s2_recipe")
    p = OUT / "s2_recipe.parquet"
    v16.to_parquet(p)
    write_config("s2_recipe", p, int((v16.split == "train").sum()), modalities=["S2L2A"],
                 batch=8, accum=2, epochs=30, ema=0.9999,
                 note="v16 corpus (the v17 S2 control's), test-region chips dropped.")

    # v16 repeats ~1,025 Pakistan rows on purpose (2x in-domain weight); MD repeats them too, so
    # dedupe MD before the join or those chips would count 4x.
    md = (pd.read_parquet(MD)[["chip_id", "image"]].drop_duplicates("chip_id")
          .rename(columns={"image": "image_md"}))
    sub = v16.merge(md, on="chip_id", how="inner")
    log.info("multi-date subset: %d of %d v16 rows have a 4-date stack", len(sub), len(v16))
    one = sub.drop(columns=["image_md"])
    p1 = OUT / "s2_1d.parquet"
    one.to_parquet(p1)
    four = sub.copy()
    four["image"] = four["image_md"]
    four = four.drop(columns=["image_md"])
    tg = four["aoi"] == FLIPPED
    four.loc[tg, "mask"] = northup_masks(four[tg])
    four["n_revisits"] = 4
    p4 = OUT / "s2_4d.parquet"
    four.to_parquet(p4)
    n = int((one.split == "train").sum())
    write_config("s2_1d", p1, n, modalities=["S2L2A"], batch=4, accum=4, epochs=30, ema=0.9999,
                 note="Matched control for s2_4d: same chips, masks, batch; one median image.")
    write_config("s2_4d", p4, n, modalities=["S2L2A"], batch=4, accum=4, epochs=30, ema=0.9999,
                 n_revisits=4,
                 note="4 single dates (2025-11..2026-03) through TemporalWrapper max pooling.")


def cmd_s1(_a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    region = test_region_union()
    v5 = drop_leaks(pd.read_parquet(V5HB), region, "s1_recipe")
    p = OUT / "s1_recipe.parquet"
    v5.to_parquet(p)
    write_config("s1_recipe", p, int((v5.split == "train").sum()), modalities=["S1RTC"],
                 batch=8, accum=2, epochs=40, ema=0.999,
                 note="v5 S1 corpus (the v17 S1 control's), test-region chips dropped.")
    if _a.recipe_only:
        return
    s1x = pd.read_parquet(S1X)[["chip_id", "s1"]].rename(columns={"s1": "s1_x"})
    v16 = pd.read_parquet(V16)
    extra = v16[~v16.chip_id.isin(v5.chip_id)].merge(s1x, on="chip_id", how="left")
    extra["s1"] = extra["s1"].fillna(extra["s1_x"])
    extra = extra.drop(columns=["s1_x"])
    missing = extra["s1"].isna()
    log.info("s1_big: %d v16 chips beyond the v5 corpus, %d without S1 (dropped: %s)",
             len(extra), int(missing.sum()), extra[missing].groupby("aoi").size().to_dict())
    big = pd.concat([v5, drop_leaks(extra[~missing].reset_index(drop=True), region, "s1_big")],
                    ignore_index=True)
    p = OUT / "s1_big.parquet"
    big.to_parquet(p)
    log.info("s1_big: %s", big.groupby(["aoi", "split"]).size().to_dict())
    write_config("s1_big", p, int((big.split == "train").sum()), modalities=["S1RTC"],
                 batch=8, accum=2, epochs=30, ema=0.9999,
                 note="s1_recipe's chips plus every v16 chip (TorchGeo, hard negatives) with S1.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["s2", "s1"])
    ap.add_argument("--recipe-only", action="store_true", help="s1: s1_recipe only (no S1 fetch yet)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"s2": cmd_s2, "s1": cmd_s1}[a.cmd](a)


if __name__ == "__main__":
    main()
