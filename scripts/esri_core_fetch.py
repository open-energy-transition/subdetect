"""Stage 1 of the ESRI core-label study: Esri imagery for every target, on the chip grid.

For each row of data/esri_core/targets.parquet, read Esri World Imagery over the target's
crop window and resample it onto an integer sub-grid of the chip's OWN 10 m UTM grid
(factor 20 -> 0.5 m, or 10 -> 1 m for big sites). Working on the chip grid means the core
mask computed from this image block-averages straight back onto the training mask with no
further reprojection; the only remaining misalignment is Esri-vs-S2 geolocation, which the
core stage estimates per target by correlating against the chip's own S2 image.

Outputs (resumable; existing crops are skipped):
  data/esri_core/crops/<aoi>/<target_id>.png    RGBA, alpha=0 where Esri has no imagery
  data/esri_core/fetch_meta.parquet              res, zoom, coverage, imagery date/res

Usage:
  pixi run -e ml python scripts/esri_core_fetch.py --sample 120        # pilot
  pixi run -e ml python scripts/esri_core_fetch.py --all --workers 8   # full
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio import Affine
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subdetect.esri import EsriTiles, zoom_for  # noqa: E402

log = logging.getLogger("esri_core_fetch")
CORE = ROOT / "data/esri_core"
MAX_SIDE_PX = 1024


def sub_factor(row) -> int:
    """Sub-grid factor of the 10 m chip grid: 20 (0.5 m) unless that exceeds MAX_SIDE_PX."""
    side_px10 = max(row.r1 - row.r0, row.c1 - row.c0)
    for f in (20, 10, 5):
        if side_px10 * f <= MAX_SIDE_PX:
            return f
    return 5


def crop_grid(row, f: int):
    with rasterio.open(ROOT / row.mask) as src:
        tr, crs = src.transform, src.crs
    sub = tr * Affine.translation(row.c0, row.r0) * Affine.scale(1.0 / f)
    shape = ((row.r1 - row.r0) * f, (row.c1 - row.c0) * f)
    return sub, crs, shape


def fetch_one(row, tiles: EsriTiles, with_meta: bool) -> dict:
    """Fetch at the zoom matching the sub-grid, falling back up to two zooms coarser when
    Esri has no imagery (placeholder tiles) there. JPEG when fully covered, else RGBA PNG
    so the no-data alpha survives."""
    base = CORE / "crops" / row.aoi / row.target_id
    f = sub_factor(row)
    res = 10.0 / f
    z0 = zoom_for(res, row.lat)
    rec = dict(target_id=row.target_id, factor=f, res_m=res)
    existing = [p for p in (base.with_suffix(".jpg"), base.with_suffix(".png")) if p.exists()]
    if existing:
        im = np.asarray(Image.open(existing[0]))
        cov = float((im[..., 3] > 0).mean()) if im.ndim == 3 and im.shape[2] == 4 else 1.0
        rec.update(path=str(existing[0]), coverage=cov,
                   gray_std=float(im[..., :3].mean(-1).std()))
    else:
        sub, crs, shape = crop_grid(row, f)
        best = None
        for z in (z0, z0 - 1, z0 - 2):
            rgba = tiles.read_grid(sub, crs, shape, z)
            cov = float((rgba[..., 3] > 0).mean())
            if best is None or cov > best[1] + 0.05:
                best = (rgba, cov, z)
            if cov >= 0.9:
                break
        rgba, cov, z = best
        out = base.with_suffix(".jpg" if cov >= 0.999 else ".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".part" + out.suffix)
        if out.suffix == ".jpg":
            Image.fromarray(rgba[..., :3], "RGB").save(tmp, quality=90)
        else:
            Image.fromarray(rgba, "RGBA").save(tmp, compress_level=3)
        tmp.replace(out)
        g = rgba[..., :3].mean(-1)[rgba[..., 3] > 0]
        rec.update(path=str(out), zoom=z, zoom_fallback=z0 - z, coverage=cov,
                   gray_std=float(g.std()) if g.size else 0.0)
    if with_meta:
        rec.update(tiles.metadata(row.lon, row.lat))
    return rec


def pick(targets: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified pilot: OSM AOIs and TorchGeo continents, small and large sites."""
    rng = np.random.default_rng(seed)
    t = targets.copy()
    t["size"] = pd.cut(t.n_px, [0, 20, 60, 200, 1e9], labels=["<2k", "2-6k", "6-20k", ">20k"])
    t["stratum"] = t.aoi.where(t.aoi != "torchgeo_substation",
                               "tg_" + pd.cut(t.lon, [-180, -30, 60, 180],
                                              labels=["americas", "eur_afr", "asia"]).astype(str))
    groups = list(t.groupby(["stratum", "size"], observed=True))
    per = max(1, n // len(groups))
    picks = [g.sample(min(per, len(g)), random_state=int(rng.integers(1e9))) for _, g in groups]
    return pd.concat(picks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, default=CORE / "targets.parquet")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--aoi", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    t = pd.read_parquet(a.targets)
    if a.aoi:
        t = t[t.aoi.isin(a.aoi)]
    if a.sample:
        t = pick(t, a.sample, a.seed)
    elif not a.all and not a.aoi:
        raise SystemExit("pass --sample N, --aoi ..., or --all")
    meta_p = CORE / "fetch_meta.parquet"
    old = pd.read_parquet(meta_p) if meta_p.exists() else pd.DataFrame(columns=["target_id"])
    done = set(old.target_id)
    # Random order, so a partially finished fetch is still a geographically fair sample
    # (the TorchGeo index is sorted by filename, i.e. by latitude).
    todo = t[~t.target_id.isin(done)].sample(frac=1.0, random_state=a.seed)
    log.info("%d targets, %d already fetched, %d to do", len(t), len(t) - len(todo), len(todo))

    tiles = EsriTiles(CORE / "tiles", workers=a.workers)
    recs, t0, fails = [], time.time(), 0

    def flush():
        nonlocal old, recs
        if recs:
            old = pd.concat([old, pd.DataFrame(recs)], ignore_index=True) \
                .drop_duplicates("target_id", keep="last")
            tmp = meta_p.with_suffix(".part")
            old.to_parquet(tmp)
            tmp.replace(meta_p)
            recs = []

    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch_one, r, tiles, not a.no_meta): r.target_id
                for r in todo.itertuples(index=False)}
        for i, fu in enumerate(tqdm(as_completed(futs), total=len(futs), mininterval=30), 1):
            try:
                recs.append(fu.result())
            except Exception as e:  # noqa: BLE001
                fails += 1
                log.warning("%s failed: %s", futs[fu], e)
            if i % 100 == 0:
                flush()
                log.info("%d/%d done, %.2f targets/s, %d failed", i, len(futs),
                         i / (time.time() - t0), fails)
    flush()
    log.info("finished: %d fetched, %d failed", len(todo) - fails, fails)


if __name__ == "__main__":
    main()
