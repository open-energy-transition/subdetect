"""Stage 3 of the ESRI core-label study: rewrite chip masks from the core probability maps.

Per target (one substation group in one chip, see esri_core_targets.py):
  core_hi  = probe P >= --thr on the Esri sub-grid, cleaned (opening, hole fill, small
             components dropped)
  frac10   = core_hi block-averaged back to the chip's 10 m grid
  positive = group pixels (dilated --dilate px, 0-pixels only) with frac10 >= 0.5
  ignore   = group pixels that are not positive          (the property, not the core)
Statuses: `core` (rewritten), `no_core` (no pixel reaches 0.5 -> whole group ignore),
`no_imagery` (coverage < 0.7 / no prob map -> original label kept).

Pixels outside every group (background, existing ignore discs/small polygons) are never
touched. Masks are written in the SOURCE chip orientation (TorchGeo chips flipped back),
and an index is written alongside, repointing only the rewritten chips.

Usage:
  pixi run -e ml python scripts/esri_core_labels.py --index data/chips_v16_honestburn/combined/index.parquet \
      --out-root data/chips_v17_core [--aoi pakistan india_pilot] [--ids-only FILE] [--preview DIR]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from esri_core_targets import GROUP_DILATE_PX  # noqa: E402
from subdetect.core_labels import is_flipped, write_label  # noqa: E402

CORE = ROOT / "data/esri_core"
P = 14


def core_hi_from_prob(p: np.ndarray, shape: tuple[int, int], res_m: float, thr: float,
                      min_comp_m2: float = 100.0) -> np.ndarray:
    """Patch-grid probability -> cleaned boolean core on the (H, W) sub-grid."""
    valid = np.isfinite(p)
    pz = np.where(valid, p, 0.0).astype(np.float32)
    hi = ndimage.zoom(pz, P, order=1)[: shape[0], : shape[1]]
    if hi.shape != shape:  # crops are padded up to /14 before DINOv2, never down
        hi = np.pad(hi, ((0, shape[0] - hi.shape[0]), (0, shape[1] - hi.shape[1])), mode="edge")
    core = hi >= thr
    r = max(1, int(round(2.0 / res_m)))
    core = ndimage.binary_opening(core, iterations=r)
    core = ndimage.binary_fill_holes(core)
    lab, n = ndimage.label(core)
    if n:
        sizes = ndimage.sum(core, lab, range(1, n + 1)) * res_m * res_m
        core = np.isin(lab, 1 + np.flatnonzero(sizes >= min_comp_m2))
    return core


def _chip(job) -> tuple[list[dict], dict | None]:
    (aoi, chip_id, mask_path, targets, out_root, thr, dilate, preview) = job
    fl = is_flipped(aoi)
    with rasterio.open(ROOT / mask_path) as src:
        m = src.read(1)
    m = m[::-1].copy() if fl else m.copy()
    pos = m == 1
    lab, _ = ndimage.label(ndimage.binary_dilation(pos, iterations=GROUP_DILATE_PX),
                           structure=np.ones((3, 3)))
    new = m.copy()
    stats = []
    for t in targets:
        grp = pos & (lab == t["group"])
        rec = dict(target_id=t["target_id"], aoi=aoi, chip_id=chip_id, orig_px=int(grp.sum()))
        pp = CORE / "prob" / aoi / f"{t['target_id']}.npz"
        if t["coverage"] < 0.7 or not pp.exists():
            rec.update(status="no_imagery", core_px=int(grp.sum()), ignore_px=0)
            stats.append(rec)
            continue
        p = np.load(pp)["p"].astype(np.float32)
        f = 20  # prob maps are on the 0.5 m sub-grid for every crop (esri_core_extract.SUB)
        h10, w10 = t["r1"] - t["r0"], t["c1"] - t["c0"]
        core = core_hi_from_prob(p, (h10 * f, w10 * f), 10.0 / f, thr)
        frac = core.reshape(h10, f, w10, f).mean(axis=(1, 3))
        win = (slice(t["r0"], t["r1"]), slice(t["c0"], t["c1"]))
        g_win = grp[win]
        allow = ndimage.binary_dilation(g_win, iterations=dilate) if dilate else g_win
        allow &= (g_win | (m[win] == 0))
        newpos = allow & (frac >= 0.5)
        sub = new[win]
        sub[g_win] = -1
        sub[newpos] = 1
        new[win] = sub
        n_core = int(newpos.sum())
        rec.update(status="core" if n_core else "no_core", core_px=n_core,
                   ignore_px=int((g_win & ~newpos).sum()), added_px=int((newpos & ~g_win).sum()),
                   core_frac_mean=float(frac[g_win].mean()) if g_win.any() else 0.0)
        stats.append(rec)
        if preview:
            o = Path(preview) / aoi / f"{t['target_id']}.npz"
            o.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(o, core=core, label=new[win])
    out = Path(out_root) / "masks" / aoi / f"{chip_id}.tif"
    write_label(ROOT / mask_path, out, new, fl)
    return stats, {"aoi": aoi, "chip_id": chip_id, "mask": str(out.relative_to(ROOT))
                   if out.is_relative_to(ROOT) else str(out),
                   "sub_pixels": int((new == 1).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--aoi", nargs="*", default=None, help="only rewrite chips of these AOIs")
    ap.add_argument("--ids-only", type=Path, default=None, help="only these target_ids (dev/QC)")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--preview", type=Path, default=None,
                    help="also write per-target core/label npz for esri_core_qc.py --core")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    t = pd.read_parquet(CORE / "targets.parquet")
    meta = pd.read_parquet(CORE / "fetch_meta.parquet")[["target_id", "factor", "coverage"]]
    t = t.merge(meta, on="target_id", how="left")
    t["coverage"] = t.coverage.fillna(0.0)
    t["factor"] = t.factor.fillna(20)
    if a.aoi:
        t = t[t.aoi.isin(a.aoi)]
    if a.ids_only:
        ids = {l.strip() for l in a.ids_only.read_text().splitlines() if l.strip()}
        t = t[t.chip_id.isin({i.rsplit("_", 1)[0] for i in ids})]
    jobs = []
    for (aoi, cid, mp), g in t.groupby(["aoi", "chip_id", "mask"]):
        jobs.append((aoi, cid, mp, g.to_dict("records"), str(a.out_root), a.thr, a.dilate,
                     str(a.preview) if a.preview else None))
    print(f"{len(jobs)} chips, {len(t)} targets to rewrite")
    stats, remap = [], {}
    with ProcessPoolExecutor(a.workers) as ex:
        for s, r in tqdm(ex.map(_chip, jobs, chunksize=16), total=len(jobs), mininterval=20):
            stats.extend(s)
            if r:
                remap[(r["aoi"], r["chip_id"])] = r
    st = pd.DataFrame(stats)
    a.out_root.mkdir(parents=True, exist_ok=True)
    st.to_parquet(a.out_root / "core_stats.parquet")

    idx = pd.read_parquet(a.index)
    keys = list(zip(idx.aoi, idx.chip_id))
    hit = np.array([k in remap for k in keys])
    idx.loc[hit, "mask"] = [remap[k]["mask"] for k, h in zip(keys, hit) if h]
    idx.loc[hit, "sub_pixels"] = [remap[k]["sub_pixels"] for k, h in zip(keys, hit) if h]
    (a.out_root / "combined").mkdir(parents=True, exist_ok=True)
    idx.to_parquet(a.out_root / "combined" / "index.parquet")

    print(st.status.value_counts().to_string())
    for aoi, g in st.groupby("aoi"):
        o = g.orig_px.sum()
        print(f"{aoi:22s} targets {len(g):6d}  orig px {o:9,d}  -> core {g.core_px.sum():9,d} "
              f"({100 * g.core_px.sum() / max(o, 1):5.1f}%), ignore {g.ignore_px.sum():9,d} "
              f"({100 * g.ignore_px.sum() / max(o, 1):5.1f}%)  no_core {int((g.status == 'no_core').sum())}")
    print(f"index: {int(hit.sum())} rows repointed -> {a.out_root / 'combined' / 'index.parquet'}")


if __name__ == "__main__":
    main()
