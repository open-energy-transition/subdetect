"""Build the core-label review page (docs/issues/esri-core-labels.md).

  pixi run -e ml python scripts/esri_core_review_page.py [out_dir]   # default data/esri_core_review (on the aidisc)

Open <out_dir>/index.html in a browser. Opened from disk, grades stay in that browser and
"Export grades (CSV)" saves them; the same page published as a claude.ai artifact saves
them to the artifact's store instead (collection `reviews`).

Set A = the frozen G0.4 QC sample (60, with Claude's grades). Set B = a fresh random sample
of the S2 training corpus, stratified by source, excluding every target used for dev, QC,
the supplementary check or probe training.

Per target: e/<id>.jpg (plain Esri crop, <=640 px) and m/<id>.png (overlay channels at the
same size: R = OSM polygon edge, G = Esri core edge, B = 10 m pixels that became ignore).
S2 RGB (10 m, data URI) and the old/new 10 m label windows go into data.json. The new label
is read from the real training masks (chips_v17_core) and checked against a recomputation.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path("/run/media/tobi/aidisc/subdetect")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from esri_core_labels import core_hi_from_prob  # noqa: E402
from subdetect.core_labels import read_north_up  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/esri_core_review"
TEMPLATE = ROOT / "scripts/templates/esri_core_review.html"
CORE = ROOT / "data/esri_core"
THR, F, MAXPX = 0.76, 20, 640
SEED = 20261105

t = pd.read_parquet(CORE / "targets.parquet").set_index("target_id", drop=False)
meta = pd.read_parquet(CORE / "fetch_meta.parquet").set_index("target_id", drop=False)
st = pd.read_parquet(ROOT / "data/chips_v17_core/core_stats.parquet").set_index("target_id")

qc_ids = [l.strip() for l in (CORE / "qc_ids.txt").read_text().splitlines() if l.strip()]
_g = [l.rstrip("\n").split(",", 3) for l in (CORE / "qc/g04/grades.csv").read_text().splitlines()[1:] if l.strip()]
grades = pd.DataFrame(_g, columns=["panel", "target_id", "grade", "note"]).set_index("target_id")  # notes hold commas
excl = set(qc_ids)
for f in ["dev_all_ids.txt", "supp_small_nocore_ids.txt"]:
    excl |= {l.strip() for l in (CORE / f).read_text().splitlines() if l.strip()}
probe = np.load(CORE / "probe/probe_data.npz", allow_pickle=True)
excl |= {str(i) for i in probe["ids"]}

rng = np.random.default_rng(SEED)
pool = st[(st.status != "no_imagery") & ~st.index.isin(excl)]
rand_ids = []
for aoi, n in [("torchgeo_substation", 24), ("india_pilot", 8), ("pakistan", 8)]:
    ids = pool.index[pool.aoi == aoi].to_numpy()
    rand_ids += list(rng.choice(ids, size=n, replace=False))

SOURCE = {"torchgeo_substation": "TorchGeo", "india_pilot": "India", "pakistan": "Pakistan",
          "pakistan_hardneg": "Pakistan"}


def edges(m: np.ndarray, w: int = 2) -> np.ndarray:
    e = m & ~ndimage.binary_erosion(m)
    return ndimage.binary_dilation(e, iterations=w - 1) if w > 1 else e


def resize(m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127


def enc(lbl: np.ndarray) -> str:
    return "".join("1" if v == 1 else ("i" if v == -1 else "0") for v in lbl.ravel())


(OUT / "e").mkdir(parents=True, exist_ok=True)
(OUT / "m").mkdir(parents=True, exist_ok=True)
rows, mismatch = [], 0
for set_name, ids in [("qc", qc_ids), ("random", rand_ids)]:
    for k, tid in enumerate(ids):
        tr, mr, sr = t.loc[tid], meta.loc[tid], st.loc[tid]
        fl = bool(tr.get("flipped", False))
        h10, w10 = int(tr.r1 - tr.r0), int(tr.c1 - tr.c0)
        old = read_north_up(ROOT / tr["mask"], fl, tr.r0, tr.r1, tr.c0, tr.c1, bands=1).astype(np.int16)
        new_path = ROOT / "data/chips_v17_core/masks" / tr.aoi / f"{tr.chip_id}.tif"
        new = read_north_up(new_path, fl, tr.r0, tr.r1, tr.c0, tr.c1, bands=1).astype(np.int16)
        old = np.where(old == 255, 0, old)
        new = np.where(new == 255, -1, new)  # uint8 storage of -1

        rgb = np.asarray(Image.open(mr.path).convert("RGB"))
        H, W = rgb.shape[:2]
        s = min(1.0, MAXPX / max(H, W))
        size = (max(1, round(W * s)), max(1, round(H * s)))
        Image.fromarray(rgb).resize(size, Image.LANCZOS).save(OUT / "e" / f"{tid}.jpg", quality=84)

        pp = CORE / "prob" / tr.aoi / f"{tid}.npz"
        core = np.zeros((h10 * F, w10 * F), bool)
        if pp.exists():
            core = core_hi_from_prob(np.load(pp)["p"].astype(np.float32), (h10 * F, w10 * F), 10.0 / F, THR)
        # check: recomputed core blocks inside this target's group == positives in the training mask
        frac = core.reshape(h10, F, w10, F).mean(axis=(1, 3))
        grp = (old == 1) & ((new == 1) | (new == -1))
        rec_pos = grp & (frac >= 0.5)
        if int(((new == 1) & grp).sum()) != int(rec_pos.sum()):
            mismatch += 1

        # RGBA sprite, three layers stacked vertically: OSM edge, core edge, newly ignored px
        ew, eh = size
        ov = np.zeros((3 * eh, ew, 4), np.uint8)
        ov[:eh][edges(resize(old == 1, size), 2)] = (255, 214, 0, 255)
        ov[eh:2 * eh][edges(resize(core, size), 2)] = (0, 229, 255, 255)
        ov[2 * eh:][resize((new == -1) & (old != -1), size)] = (255, 43, 214, 120)
        Image.fromarray(ov, "RGBA").save(OUT / "m" / f"{tid}.png", optimize=True)

        s2 = read_north_up(ROOT / tr["image"], fl, tr.r0, tr.r1, tr.c0, tr.c1, bands=[3, 2, 1]).astype("float32")
        s2 = np.moveaxis(s2, 0, -1)
        lo, hi = np.percentile(s2[s2 > 0], [2, 98]) if (s2 > 0).any() else (0, 1)
        s2 = (np.clip((s2 - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(s2).save(buf, "PNG", optimize=True)

        g = grades.loc[tid] if (set_name == "qc" and tid in grades.index) else None
        date = str(mr.get("img_date", ""))
        rows.append(dict(
            id=tid, set=set_name, n=k + 1, source=SOURCE.get(tr.aoi, tr.aoi),
            area=int(tr.n_px) * 100, res=float(mr.res_m), year=date[:4] if date[:4].isdigit() else "",
            lat=round(float(tr.lat), 5), lon=round(float(tr.lon), 5),
            status=str(sr.status), orig=int(sr.orig_px), core=int(sr.core_px), ign=int(sr.ignore_px),
            w10=w10, h10=h10, ew=size[0], eh=size[1],
            s2="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            old=enc(old), new=enc(new),
            claude=(str(g.grade) if g is not None else ""),
            claude_note=(str(g.note) if g is not None and isinstance(g.note, str) else ""),
        ))
        print(set_name, k + 1, tid, sr.status, size, flush=True)

data = json.dumps(rows, separators=(",", ":"))
(OUT / "data.json").write_text(data)
# The template is artifact-shaped (no document skeleton); wrap it for opening from disk.
head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style></head><body>')
page = TEMPLATE.read_text().replace("__DATA__", data.replace("</", "<\\/"))
(OUT / "index.html").write_text(head + page + "</body></html>\n")
# label-check: recomputed core blocks vs the training mask, counted over every group in the
# window; a window that holds another target's group can differ without being wrong.
print(f"{len(rows)} targets -> {OUT / 'index.html'} (label-check differences: {mismatch})")
