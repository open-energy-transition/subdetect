"""Stage 2 of the ESRI core-label study: find the equipment core inside each label.

Method (weak supervision, no hand labels): DINOv2 ViT-B/14 (reg4) patch features on each
Esri crop (~7 m patches at 0.5 m/px), and a linear probe trained to separate patches INSIDE
the original training masks from patches well OUTSIDE them, pooled over thousands of sites.
The probe can only learn what is consistently inside-not-outside: the switchyard itself
(gantries, busbars, transformers on gravel) almost never occurs outside a substation, so it
scores high; the fields, lawns, parking and staff housing that bloated polygons sweep in
look like the land outside, so they score low even though they were labelled positive.
That is exactly the separation the study needs, and it is learned from the noisy labels
themselves rather than asserted.

  probe-data  features + per-patch labels for a stratified subset -> data/esri_core/probe/
  train       fit the probe (held-out sites -> patch AUC), save probe.pt
  apply       run DINOv2 + probe over every fetched crop -> per-target core prob map (npz)

The label rewrite (prob map -> 10 m core / ignore mask) is a separate step
(scripts/esri_core_labels.py) so thresholds can be revisited without recomputing features.

Usage:
  pixi run -e ml python scripts/esri_core_extract.py probe-data --n 3000
  pixi run -e ml python scripts/esri_core_extract.py train
  pixi run -e ml python scripts/esri_core_extract.py apply [--aoi pakistan india_pilot]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subdetect.core_labels import read_north_up  # noqa: E402

log = logging.getLogger("esri_core_extract")
CORE = ROOT / "data/esri_core"
PROBE = CORE / "probe"
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"
P = 14            # ViT patch size (px)
LAYER = 8         # transformer block whose patch tokens are used (1-based). Measured on 600
                  # held-out-split sites: inside-vs-near-ring AUC 0.942/0.945/0.952/0.946/0.947
                  # for blocks 4/6/8/10/12 -- depth barely matters, 8 is best and 1/3 cheaper.
NEAR_M = 60.0     # outside patches within this distance of the label are the "near ring"
NEAR_W = 3.0      # ...and weigh this much more as negatives: they share the switchyard's
                  # context, so only LOCAL appearance can separate them from the inside
TILE = 37 * P     # 518 px tiles for big crops
SUB = 20          # every crop is processed on the 0.5 m sub-grid (20x the 10 m chip grid)
OUT_GAP_M = 10.0  # patches within this distance of the mask are neither inside nor outside
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------- feature extraction
class Dino:
    def __init__(self, device: str | None = None):
        import timm
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp32 on purpose: the GTX 1060 (Pascal) runs fp16 at a fraction of its fp32 rate --
        # measured 557 vs 102 ms per 518 px tile.
        self.m = timm.create_model(MODEL, pretrained=True, dynamic_img_size=True).eval().to(self.device)
        self.npre = self.m.num_prefix_tokens

    def _run(self, rgb: np.ndarray) -> np.ndarray:
        """(h, w, 3) uint8 with h, w multiples of 14 -> (h/14, w/14, 768) float32."""
        torch = self.torch
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)
        with torch.no_grad():
            f = self.m.forward_intermediates(x, indices=[LAYER - 1], output_fmt="NCHW",
                                             intermediates_only=True, norm=False,
                                             stop_early=True)[0]
        return f[0].float().permute(1, 2, 0).cpu().numpy()

    def features(self, rgb: np.ndarray) -> np.ndarray:
        """Patch features over a whole crop of any size (edge-padded to /14, tiled at 518)."""
        H, W = rgb.shape[:2]
        Hp, Wp = -(-H // P) * P, -(-W // P) * P
        img = np.pad(rgb, ((0, Hp - H), (0, Wp - W), (0, 0)), mode="edge")
        out = np.zeros((Hp // P, Wp // P, 768), np.float32)
        for y in range(0, Hp, TILE):
            for x in range(0, Wp, TILE):
                t = img[y:y + TILE, x:x + TILE]
                out[y // P:(y + t.shape[0]) // P, x // P:(x + t.shape[1]) // P] = self._run(t)
        return out


def load_rgba(path: str, factor: int = 20) -> np.ndarray:
    """Esri crop resampled to the 0.5 m sub-grid (factor 20 of the 10 m chip grid).

    Large sites were fetched at 1 m or 2 m (factor 10 / 5) to bound crop size, but the probe
    is scale-specific: on a 1 m crop a 60,000 m2 switchyard scored as background (dev QC,
    2026-10-31). Upsampling to 0.5 m puts every crop at the scale the probe was fit on.
    """
    import cv2

    im = Image.open(path)
    if im.mode == "RGBA":
        a = np.asarray(im)
    else:
        rgb = np.asarray(im.convert("RGB"))
        a = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    up = SUB // int(factor)
    if up > 1:
        H, W = a.shape[:2]
        rgb = cv2.resize(a[..., :3], (W * up, H * up), interpolation=cv2.INTER_LINEAR)
        al = cv2.resize(a[..., 3], (W * up, H * up), interpolation=cv2.INTER_NEAREST)
        a = np.dstack([rgb, al])
    return a


def patch_mean(a: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Mean of a (H, W) array over the 14 px patch grid (edge-padded like the image)."""
    H, W = a.shape
    a = np.pad(a.astype(np.float32), ((0, gh * P - H), (0, gw * P - W)), mode="edge")
    return a.reshape(gh, P, gw, P).mean(axis=(1, 3))


def target_arrays(t, meta):
    """Esri RGBA crop + the original mask on the same sub-grid (both north-up)."""
    rgba = load_rgba(meta.path, int(meta.factor))
    m10 = read_north_up(ROOT / t["mask"], bool(t.flipped), t.r0, t.r1, t.c0, t.c1, bands=1)
    m_hi = np.kron(m10, np.ones((SUB, SUB), dtype=m10.dtype))
    assert m_hi.shape == rgba.shape[:2], (m_hi.shape, rgba.shape, t.target_id)
    return rgba, m10, m_hi, SUB


def patch_labels(m_hi: np.ndarray, alpha: np.ndarray, gh: int, gw: int, res_m: float):
    """1 inside (>=90% covered by a positive), 2 near ring (OUT_GAP_M..NEAR_M from any
    labelled pixel), 0 far (> NEAR_M), -1 otherwise (boundary, ignore regions, no imagery)."""
    pos = m_hi == 1
    cov = patch_mean(pos, gh, gw)
    dist = ndimage.distance_transform_edt(m_hi == 0) * res_m
    dp = patch_mean(dist, gh, gw)
    valid = patch_mean(alpha > 0, gh, gw) > 0.95
    lab = np.full((gh, gw), -1, np.int8)
    lab[(cov >= 0.9) & valid] = 1
    lab[(dp > OUT_GAP_M) & (dp <= NEAR_M) & valid] = 2
    lab[(dp > NEAR_M) & valid] = 0
    return lab, cov


# ---------------------------------------------------------------- stages
def _load():
    t = pd.read_parquet(CORE / "targets.parquet").set_index("target_id", drop=False)
    meta = pd.read_parquet(CORE / "fetch_meta.parquet").set_index("target_id", drop=False)
    meta = meta[meta.coverage.fillna(0) >= 0.7]
    return t.loc[t.index.intersection(meta.index)], meta


def cmd_probe_data(a) -> None:
    t, meta = _load()
    rng = np.random.default_rng(a.seed)
    # stratify: every OSM site is eligible, TorchGeo subsampled across continents
    t = t.assign(strat=np.where(t.aoi.isin(["pakistan", "india_pilot"]), t.aoi,
                                "tg" + pd.cut(t.lon, [-180, -30, 60, 180], labels=["A", "E", "S"]).astype(str)))
    picks = []
    for _, g in t.groupby("strat"):
        picks.append(g.sample(min(len(g), a.n // max(1, t.strat.nunique())),
                              random_state=int(rng.integers(1e9))))
    sel = pd.concat(picks)
    log.info("probe sites: %s", sel.strat.value_counts().to_dict())
    dino = Dino()
    X, Y, S = [], [], []
    for i, (tid, r) in enumerate(sel.iterrows()):
        rgba, _, m_hi, f = target_arrays(r, meta.loc[tid])
        feat = dino.features(rgba[..., :3])
        gh, gw = feat.shape[:2]
        lab, _ = patch_labels(m_hi, rgba[..., 3], gh, gw, 10.0 / f)
        ii = np.flatnonzero(lab.ravel() == 1)
        nn = np.flatnonzero(lab.ravel() == 2)
        oo = np.flatnonzero(lab.ravel() == 0)
        # cap per site so big sites don't dominate
        ii = rng.choice(ii, min(len(ii), 200), replace=False) if len(ii) else ii
        nn = rng.choice(nn, min(len(nn), max(100, 2 * len(ii))), replace=False) if len(nn) else nn
        oo = rng.choice(oo, min(len(oo), max(50, len(ii))), replace=False) if len(oo) else oo
        k = np.concatenate([ii, nn, oo])
        X.append(feat.reshape(-1, feat.shape[-1])[k].astype(np.float16))
        Y.append(lab.ravel()[k])
        S.append(np.full(len(k), i, np.int32))
        if i % 200 == 0:
            log.info("%d/%d sites", i, len(sel))
    PROBE.mkdir(parents=True, exist_ok=True)
    np.savez(PROBE / "probe_data.npz", X=np.concatenate(X), y=np.concatenate(Y),
             site=np.concatenate(S), ids=np.array(sel.index), strat=np.array(sel.strat))
    log.info("saved %d patches from %d sites", sum(len(y) for y in Y), len(sel))


def cmd_train(a) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    d = np.load(a.data or (PROBE / "probe_data.npz"), allow_pickle=True)
    # X stays float16: a full float32 copy plus its standardised twin is ~10 GB for 1.65M x 768,
    # more than this box can spare next to earthpv. Stats are accumulated in chunks instead.
    X, y, site = d["X"], d["y"].astype(np.int8), d["site"]
    rng = np.random.default_rng(a.seed)
    sites = np.unique(site)
    va = np.isin(site, rng.choice(sites, len(sites) // 5, replace=False))
    # the frozen G0.4 QC sample must stay unseen by the probe (pre-registration): its sites
    # are dropped from both train and held-out, so they can't leak in or inflate the AUC
    if a.exclude and Path(a.exclude).exists():
        ex = {l.strip() for l in Path(a.exclude).read_text().splitlines() if l.strip()}
        bad = np.flatnonzero(np.isin(d["ids"], list(ex)))
        y = np.where(np.isin(site, bad), -1, y).astype(np.int8)
        log.info("excluded %d QC sites (%d patches) from probe training",
                 len(bad), int(np.isin(site, bad).sum()))
    va &= y >= 0
    s1 = np.zeros(X.shape[1]); s2 = np.zeros(X.shape[1])
    idx = np.flatnonzero(~va)
    for k in range(0, len(idx), 100_000):
        c = X[idx[k:k + 100_000]].astype(np.float64)
        s1 += c.sum(0); s2 += (c * c).sum(0)
    mu = (s1 / len(idx)).astype(np.float32)
    sd = (np.sqrt(np.maximum(s2 / len(idx) - (s1 / len(idx)) ** 2, 0)) + 1e-6).astype(np.float32)
    tr = ~va & (y >= 0)
    if tr.sum() > a.max_train:  # sklearn works in float64: 1.65M x 768 would be ~10 GB
        keep = rng.choice(np.flatnonzero(tr), a.max_train, replace=False)
        tr = np.zeros_like(tr)
        tr[keep] = True
    w = np.where(y == 2, NEAR_W, 1.0)
    clf = LogisticRegression(C=a.C, max_iter=400, class_weight="balanced")
    clf.fit((X[tr].astype(np.float32) - mu) / sd, (y[tr] == 1).astype(int), sample_weight=w[tr])
    p = clf.predict_proba((X[va].astype(np.float32) - mu) / sd)[:, 1]
    yv = y[va]
    log.info("held-out sites: AUC inside-vs-near %.4f  inside-vs-far %.4f  inside-vs-all %.4f",
             roc_auc_score(yv[yv != 0] == 1, p[yv != 0]), roc_auc_score(yv[yv != 2] == 1, p[yv != 2]),
             roc_auc_score(yv == 1, p))
    log.info("mean p: inside %.2f near %.2f far %.2f", p[yv == 1].mean(), p[yv == 2].mean(), p[yv == 0].mean())
    strat = d["strat"] if "strat" in d else None
    if strat is not None:
        for s_ in np.unique(strat):
            m = np.isin(site[va], np.flatnonzero(strat == s_))
            yy, pp = yv[m], p[m]
            if m.sum() and (yy == 1).any() and (yy != 1).any():
                log.info("  %-12s n=%6d AUC all %.4f  near %.4f", s_, m.sum(), roc_auc_score(yy == 1, pp),
                         roc_auc_score(yy[yy != 0] == 1, pp[yy != 0]) if (yy == 2).any() else float("nan"))
    w_, b_ = clf.coef_[0].astype(np.float32), float(clf.intercept_[0])
    import torch
    torch.save(dict(w=w_, b=b_, mu=mu, sd=sd, model=MODEL, layer=LAYER), PROBE / "probe.pt")
    log.info("saved probe -> %s", PROBE / "probe.pt")


def cmd_apply(a) -> None:
    import torch

    t, meta = _load()
    if a.aoi:
        t = t[t.aoi.isin(a.aoi)]
    if a.ids:
        ids = [l.strip() for l in Path(a.ids).read_text().splitlines() if l.strip()]
        t = t.loc[t.index.intersection(ids)]
    pr = torch.load(PROBE / "probe.pt", weights_only=False)
    assert pr.get("layer") == LAYER, f"probe trained on block {pr.get('layer')}, extractor uses {LAYER}"
    w = (pr["w"] / pr["sd"]).astype(np.float32)
    b = float(pr["b"] - (pr["mu"] / pr["sd"]) @ pr["w"])
    dino = Dino()
    out_dir = CORE / "prob"
    todo = [tid for tid, r in t.iterrows() if not (out_dir / r.aoi / f"{tid}.npz").exists()]
    log.info("%d targets, %d to do", len(t), len(todo))
    t0 = time.time()
    for i, tid in enumerate(todo, 1):
        r = t.loc[tid]
        rgba = load_rgba(meta.loc[tid].path, int(meta.loc[tid].factor))
        feat = dino.features(rgba[..., :3])
        logit = feat @ w + b
        p = 1.0 / (1.0 + np.exp(-logit))
        p[patch_mean(rgba[..., 3] > 0, *p.shape) < 0.5] = np.nan
        o = out_dir / r.aoi / f"{tid}.npz"
        o.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(o, p=p.astype(np.float16))
        if i % 500 == 0:
            log.info("%d/%d  %.1f/s", i, len(todo), i / (time.time() - t0))
    log.info("done %d", len(todo))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("probe-data")
    p1.add_argument("--n", type=int, default=3000)
    p1.add_argument("--seed", type=int, default=0)
    p2 = sub.add_parser("train")
    p2.add_argument("--C", type=float, default=0.05)
    p2.add_argument("--max-train", type=int, default=500_000)
    p2.add_argument("--data", default=None, help="probe data npz (default probe/probe_data.npz)")
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--exclude", default=str(CORE / "qc_ids.txt"),
                    help="target_ids whose sites are never used by the probe (frozen QC sample)")
    p3 = sub.add_parser("apply")
    p3.add_argument("--aoi", nargs="*", default=None)
    p3.add_argument("--ids", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"probe-data": cmd_probe_data, "train": cmd_train, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
