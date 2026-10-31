"""Render QC mosaics for the ESRI core-label study.

Each panel pair = [Esri crop at 0.5-2 m | S2 RGB of the same window at 10 m]. The original
training mask is outlined in YELLOW; the refined core (when --core is given and a core mask
exists) in CYAN, and pixels the refinement turns to ignore are tinted magenta on the S2
panel. Panels are labelled with a running number so a human grader can refer to them.

Usage:
  pixi run -e ml python scripts/esri_core_qc.py --ids-from data/esri_core/fetch_meta.parquet \
      --out data/esri_core/qc/pilot --per 12
  pixi run -e ml python scripts/esri_core_qc.py --core data/esri_core/cores --sample 96 \
      --out data/esri_core/qc/round1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from subdetect.core_labels import read_north_up  # noqa: E402
CORE = ROOT / "data/esri_core"
BIG, SMALL, PAD, LBL = 400, 200, 6, 18


def _edges(m: np.ndarray, width: int = 2) -> np.ndarray:
    m = m.astype(bool)
    e = m & ~ndimage.binary_erosion(m, iterations=1)
    return ndimage.binary_dilation(e, iterations=width - 1) if width > 1 else e


def _resize_mask(m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127


def load_crop(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".png":
        return np.asarray(Image.open(p).convert("RGB"))
    return np.asarray(Image.open(p).convert("RGB"))


def panel(t, meta, core_dir: Path | None) -> Image.Image:
    rgb = load_crop(meta.path)
    H, W = rgb.shape[:2]
    s = BIG / max(H, W)
    size = (max(1, int(W * s)), max(1, int(H * s)))
    esri = np.asarray(Image.fromarray(rgb).resize(size, Image.BILINEAR)).copy()
    fl = bool(t.get("flipped", False))
    m = read_north_up(ROOT / t["mask"], fl, t.r0, t.r1, t.c0, t.c1, bands=1)
    s2 = read_north_up(ROOT / t["image"], fl, t.r0, t.r1, t.c0, t.c1, bands=[3, 2, 1]).astype("float32")
    s2 = np.moveaxis(s2, 0, -1)
    lo, hi = np.percentile(s2[s2 > 0], [2, 98]) if (s2 > 0).any() else (0, 1)
    s2 = (np.clip((s2 - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
    ssize = (max(1, int(size[0] * SMALL / BIG)), max(1, int(size[1] * SMALL / BIG)))
    s2 = np.asarray(Image.fromarray(s2).resize(ssize, Image.NEAREST)).copy()

    orig = m == 1
    esri[_edges(_resize_mask(orig, size), 2)] = (255, 230, 0)
    s2[_edges(_resize_mask(orig, ssize), 1)] = (255, 230, 0)
    if core_dir is not None:
        cp = core_dir / t.aoi / f"{t.target_id}.npz"
        if cp.exists():
            z = np.load(cp)
            core_hi = z["core"]  # sub-grid core mask already in the S2 frame
            esri[_edges(_resize_mask(core_hi, size), 2)] = (0, 255, 255)
            new = z["label"]       # 10 m label window: 1 core, -1 ignore, 0 bg
            ign = _resize_mask(new == -1, ssize)
            s2[ign] = (0.5 * s2[ign] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
            s2[_edges(_resize_mask(new == 1, ssize), 1)] = (0, 255, 255)
    w = BIG + SMALL + PAD
    im = Image.new("RGB", (w, BIG + LBL), (30, 30, 30))
    im.paste(Image.fromarray(esri), (0, LBL))
    im.paste(Image.fromarray(s2), (BIG + PAD, LBL))
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, default=CORE / "targets.parquet")
    ap.add_argument("--meta", type=Path, default=CORE / "fetch_meta.parquet")
    ap.add_argument("--core", type=Path, default=None, help="core dir (npz per target)")
    ap.add_argument("--ids", type=Path, default=None, help="text file of target_ids, in order")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per", type=int, default=12)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    t = pd.read_parquet(a.targets).set_index("target_id", drop=False)
    meta = pd.read_parquet(a.meta).set_index("target_id", drop=False)
    ids = list(meta.index)
    if a.core is not None:
        ids = [i for i in ids if (a.core / t.loc[i, "aoi"] / f"{i}.npz").exists()]
    if a.ids is not None:
        ids = [l.strip() for l in a.ids.read_text().splitlines() if l.strip()]
    if a.sample:
        rng = np.random.default_rng(a.seed)
        ids = list(rng.choice(ids, size=min(a.sample, len(ids)), replace=False))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "ids.txt").write_text("\n".join(ids) + "\n")

    rows = (a.per + a.cols - 1) // a.cols
    pw, ph = BIG + SMALL + PAD, BIG + LBL
    for k in range(0, len(ids), a.per):
        sheet = Image.new("RGB", (a.cols * (pw + PAD), rows * (ph + PAD)), (0, 0, 0))
        d = ImageDraw.Draw(sheet)
        for j, tid in enumerate(ids[k:k + a.per]):
            tr, mr = t.loc[tid], meta.loc[tid]
            x, y = (j % a.cols) * (pw + PAD), (j // a.cols) * (ph + PAD)
            try:
                sheet.paste(panel(tr, mr, a.core), (x, y))
            except Exception as e:  # noqa: BLE001
                d.text((x + 4, y + 30), f"ERR {e}", fill=(255, 0, 0))
            date = str(mr.get("img_date", ""))[:4]
            d.text((x + 4, y + 2), f"#{k + j}  {tr.aoi[:10]}  {tr.n_px * 100:,.0f} m2  "
                   f"{mr.res_m:.1f}m  esri {date}  {tid}", fill=(255, 255, 255))
        sheet.save(a.out / f"sheet_{k // a.per:03d}.jpg", quality=88)
    print(f"{len(ids)} targets -> {a.out} ({(len(ids) + a.per - 1) // a.per} sheets)")


if __name__ == "__main__":
    main()
