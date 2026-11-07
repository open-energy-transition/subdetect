"""Probability rasters for any checkpoints on any field-evaluation region (v18, 2026-11-05).

infer_sindh_test_ckpts.py generalised: `--region` picks data/osmose_regions/<region>/ (its
composites/<cell>/composite_{0,s1}.tif), outputs go to data/osmose_regions/<region>_<arm>/prob,
same 224 px windows, stride 104, Hann blending and valid-pixel rule, so a sindh_test row from
here matches the historical ones. Each arm is `NAME=CKPT[|MODE[|UPSAMPLE]]`:

  single    one window of the median composite (every production model)
  datemax   a single-date model run on each of the cell's 4 cached dates, max over dates (the
            no-retrain test of whether the median composite throws signal away)
  temporal  a TemporalWrapper model (configs/terramind_sub_v18_s2_4d.yaml) on the 4 dates

Dates come from `fetch-dates` (4 least-cloudy dates per cell on the cell's own composite grid,
2025-11-01..2026-03-15, build_multidate_chips.fetch_dates_geobox: the training stacks' fetch).
A cell with fewer than 4 clear dates falls back to its composite (datemax: one date; temporal:
T = 1) and is listed in <region>/dates/fallback.txt; a deployment would do the same.

Usage:
  pixi run python scripts/infer_region_ckpts.py fetch-dates --region gujarat_test
  pixi run -e ml python scripts/infer_region_ckpts.py infer --region gujarat_test \
      --arm "v12=data/models/<v12>.ckpt|single|2" --arm "v12dmax=data/models/<v12>.ckpt|datemax|2"
  pixi run python scripts/infer_region_ckpts.py fuse --region gujarat_test --a v12 --b v5 --out v12_x_v5
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
log = logging.getLogger("infer_region_ckpts")

# A stalled HTTP read otherwise hangs a fetch for good (2026-11-05: three S1 sockets idle 12 min
# with no error). GDAL drops a transfer under 1 kB/s for 60 s and retries it, like earthpv's
# compose._harden_python_network.
for _k, _v in {"GDAL_HTTP_TIMEOUT": "180", "GDAL_HTTP_LOW_SPEED_TIME": "60",
               "GDAL_HTTP_LOW_SPEED_LIMIT": "1024", "GDAL_HTTP_MAX_RETRY": "4",
               "GDAL_HTTP_RETRY_DELAY": "3", "GDAL_HTTP_TCP_KEEPALIVE": "YES"}.items():
    os.environ.setdefault(_k, _v)
WIN, STRIDE = 224, 104
REG = ROOT / "data/osmose_regions"
DATES_ROOT = Path("/home/tobi/subdetect-data/region_dates")
S2_WINDOW = ("2025-11-01", "2026-03-15")
N_DATES = 4


def _write(path: Path, arr: np.ndarray, crs, transform, dtype="uint8") -> None:
    """Atomic and durable (a crash must never leave a partial tif under its final name)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(tmp, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                       count=arr.shape[0], dtype=dtype, crs=crs, transform=transform,
                       compress="deflate", predictor=2) as dst:
        dst.write(arr)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _cells(region: str) -> list[str]:
    comp = REG / region / "composites"
    return sorted(p.name for p in comp.iterdir()
                  if (p / "composite_0.tif").exists() and (p / "composite_s1.tif").exists())


def cmd_fetch_dates(a) -> None:
    from odc.geo.geobox import GeoBox

    # The training stacks' own fetch (identical selection and masking). A lazy per-date variant
    # read ~20% less on the shared link but did not reproduce its pixels (2026-11-06), so the
    # evaluation stacks use the same code as the training ones.
    from build_multidate_chips import fetch_dates_geobox

    out = DATES_ROOT / a.region
    out.mkdir(parents=True, exist_ok=True)
    link = REG / a.region / "dates"
    if not link.exists():
        link.symlink_to(out)
    cells = _cells(a.region)
    short = set((out / "fallback.txt").read_text().split()) if (out / "fallback.txt").exists() else set()
    for i, cell in enumerate(cells, 1):
        dst = out / f"{cell}.tif"
        if dst.exists() or (cell in short and not a.retry_short):
            continue
        with rasterio.open(REG / a.region / "composites" / cell / "composite_0.tif") as src:
            gbox = GeoBox((src.height, src.width), src.transform, src.crs)
            crs, transform = src.crs, src.transform
        b = gbox.extent.to_crs("EPSG:4326").boundingbox
        try:
            dates = fetch_dates_geobox((b.left, b.bottom, b.right, b.top), gbox, S2_WINDOW, N_DATES)
        except Exception as e:  # noqa: BLE001 - logged, retried on the next run
            log.warning("%s: %s", cell, e)
            continue
        if len(dates) < N_DATES:
            short.add(cell)
            (out / "fallback.txt").write_text("\n".join(sorted(short)) + "\n")
            log.info("%s: only %d clear dates -> composite fallback", cell, len(dates))
            continue
        _write(dst, np.concatenate(dates, axis=0), crs, transform, dtype="uint16")
        if i % 10 == 0 or i == len(cells):
            log.info("[%d/%d] dates %s", i, len(cells), cell)


def _parse_arm(spec: str):
    name, rest = spec.split("=", 1)
    parts = rest.split("|")
    ckpt = parts[0]
    mode = parts[1] if len(parts) > 1 and parts[1] else "single"
    up = int(parts[2]) if len(parts) > 2 else 1
    if mode not in ("single", "datemax", "temporal"):
        raise SystemExit(f"unknown mode {mode!r} in {spec!r}")
    return name, ckpt, mode, up


def cmd_infer(a) -> None:
    from subdetect.infer import load_model, predict_window

    arms = {}
    for spec in a.arm:
        name, ckpt, mode, up = _parse_arm(spec)
        task, device = load_model(Path(ckpt))
        mods = task.hparams.get("model_args", {}).get("backbone_modalities", ["S2L2A"])
        arms[name] = (task, mods, device, up, mode, REG / f"{a.region}_{name}" / "prob")
        log.info("arm %s: %s mode=%s modalities=%s upsample=%s", name, ckpt, mode, mods, up)
    hann = np.outer(np.hanning(WIN), np.hanning(WIN)).astype("float32") + 1e-3
    comp_dir, dates_dir = REG / a.region / "composites", REG / a.region / "dates"
    cells = _cells(a.region)
    need_dates = any(v[4] != "single" for v in arms.values())
    for i, cell in enumerate(cells, 1):
        if all((v[5] / f"{cell}.tif").exists() for v in arms.values()):
            continue
        stack = None
        if need_dates and (dates_dir / f"{cell}.tif").exists():
            with rasterio.open(dates_dir / f"{cell}.tif") as ds:
                stack = ds.read().astype("float32")  # (T*10, H, W)
            stack = stack.reshape(-1, 10, *stack.shape[-2:])  # (T, 10, H, W)
        elif need_dates:
            log.info("%s: no date stack, composite fallback", cell)
        with rasterio.open(comp_dir / cell / "composite_0.tif") as src, \
                rasterio.open(comp_dir / cell / "composite_s1.tif") as src1:
            H, W, transform, crs = src.height, src.width, src.transform, src.crs
            acc = {k: np.zeros((H, W), "float32") for k in arms}
            wacc = np.zeros((H, W), "float32")
            valid_any = np.zeros((H, W), bool)
            rows = sorted(set(list(range(0, max(H - WIN, 0) + 1, STRIDE)) + [max(H - WIN, 0)]))
            cols = sorted(set(list(range(0, max(W - WIN, 0) + 1, STRIDE)) + [max(W - WIN, 0)]))
            for r in rows:
                for c in cols:
                    win = Window(c, r, WIN, WIN)
                    s2_np = src.read(window=win, boundless=True, fill_value=0).astype("float32")[:10]
                    h, w = min(WIN, H - r), min(WIN, W - c)
                    if (s2_np[:, :h, :w] > 0).mean() < 0.2:
                        continue
                    s1_np = src1.read(window=win, boundless=True, fill_value=0).astype("float32")[:2]
                    dates = None
                    if stack is not None:
                        dates = np.zeros((stack.shape[0], 10, WIN, WIN), "float32")
                        dates[:, :, :h, :w] = stack[:, :, r:r + h, c:c + w]
                    for k, (task, mods, device, up, mode, _) in arms.items():
                        if mode == "single" or dates is None and mode == "datemax":
                            p = predict_window(task, mods, device, s2_np, s1_np, upsample=up)
                        elif mode == "datemax":
                            p = np.max([predict_window(task, mods, device, d, s1_np, upsample=up)
                                        for d in dates], axis=0)
                        else:  # temporal: (10, T, H, W); T = 1 on a fallback cell
                            x = (dates.transpose(1, 0, 2, 3) if dates is not None
                                 else s2_np[:, None])
                            p = predict_window(task, mods, device, np.ascontiguousarray(x), None)
                        acc[k][r:r + h, c:c + w] += p[:h, :w] * hann[:h, :w]
                    wacc[r:r + h, c:c + w] += hann[:h, :w]
                    valid_any[r:r + h, c:c + w] |= (s2_np[:, :h, :w] > 0).any(axis=0)
        for k, v in arms.items():
            prob = np.where(wacc > 0, acc[k] / np.maximum(wacc, 1e-6), 0.0)
            prob[~valid_any] = 0.0
            _write(v[5] / f"{cell}.tif", (np.clip(prob, 0, 1) * 255).astype("uint8"), crs,
                   transform)
        if i % 25 == 0 or i == len(cells):
            log.info("[%d/%d] %s", i, len(cells), cell)


def cmd_fuse(a) -> None:
    da = REG / f"{a.region}_{a.a}" / "prob"
    db = REG / f"{a.region}_{a.b}" / "prob"
    out = REG / f"{a.region}_{a.out}" / "prob"
    n = 0
    for fa in sorted(da.glob("*.tif")):
        fb = db / fa.name
        if not fb.exists():
            continue
        with rasterio.open(fa) as sa, rasterio.open(fb) as sb:
            x = (sa.read(1).astype(np.float32) + sb.read(1).astype(np.float32)) / 2.0
            _write(out / fa.name, np.round(x).astype("uint8"), sa.crs, sa.transform)
        n += 1
    log.info("fused %d cells -> %s", n, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p0 = sub.add_parser("fetch-dates")
    p0.add_argument("--region", required=True)
    p0.add_argument("--retry-short", action="store_true")
    p1 = sub.add_parser("infer")
    p1.add_argument("--region", required=True)
    p1.add_argument("--arm", action="append", required=True, help="NAME=CKPT[|MODE[|UPSAMPLE]]")
    p2 = sub.add_parser("fuse")
    p2.add_argument("--region", required=True)
    p2.add_argument("--a", required=True)
    p2.add_argument("--b", required=True)
    p2.add_argument("--out", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"fetch-dates": cmd_fetch_dates, "infer": cmd_infer, "fuse": cmd_fuse}[a.cmd](a)


if __name__ == "__main__":
    main()
