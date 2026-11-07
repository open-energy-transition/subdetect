"""Sentinel-1 RTC VV/VH chips for every training chip that has none (v18, 2026-11-05).

The S1 arm is the stronger one (v5_s1only alone: AUC 0.796, 52 sindh_test hits) but trains on
the 4,730 Pakistan/India chips that carry S1, while the S2 arm gained 26,496 TorchGeo chips plus
2,600 hard negatives. This fetches S1 for those chips so S1 can train on the same corpus.

Per chip: the chip's own image grid (transform, CRS, shape) becomes the odc GeoBox, and
`subdetect.imagery.s1_composite` (the function behind every composite_s1.tif) takes the
dry-season median over up to `--max-items` scenes and encodes it as uint16 DN. TorchGeo chips
are stored SOUTH-UP under a north-up transform (memory subdetect-torchgeo-flipped-chips), so
their S1, fetched north-up on that transform, is flipped `[:, ::-1]` before writing, to sit
on the stored image and mask. `check-flip` verifies that on real chips before a run.

Resumable (an existing tif is done; writes go through a temp name). A chip with no scenes or
under `--min-valid` valid pixels is logged in failures.csv and retried on the next run.

Usage:
  pixi run python scripts/fetch_s1_chips.py check-flip --n 12
  pixi run python scripts/fetch_s1_chips.py fetch --workers 2
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
log = logging.getLogger("fetch_s1_chips")

# A stalled HTTP read otherwise hangs a fetch for good (2026-11-05: three S1 sockets idle 12 min
# with no error). GDAL drops a transfer under 1 kB/s for 60 s and retries it, like earthpv's
# compose._harden_python_network.
for _k, _v in {"GDAL_HTTP_TIMEOUT": "180", "GDAL_HTTP_LOW_SPEED_TIME": "60",
               "GDAL_HTTP_LOW_SPEED_LIMIT": "1024", "GDAL_HTTP_MAX_RETRY": "4",
               "GDAL_HTTP_RETRY_DELAY": "3", "GDAL_HTTP_TCP_KEEPALIVE": "YES"}.items():
    os.environ.setdefault(_k, _v)

BASE_INDEX = ROOT / "data/chips_v16_honestburn/combined/index.parquet"
OUT = ROOT / "data/chips_s1_extra"          # -> /home/tobi/subdetect-data/chips_s1_extra
WINDOW = ("2025-11-01", "2026-03-15")        # configs/aoi.yaml s1_window, as for every S1 chip
FLIPPED_AOIS = {"torchgeo_substation"}


def _geobox(image_path: str):
    from odc.geo.geobox import GeoBox

    with rasterio.open(image_path) as src:
        return GeoBox((src.height, src.width), src.transform, src.crs), src.profile


def _bbox_wgs84(gbox) -> tuple[float, float, float, float]:
    b = gbox.extent.to_crs("EPSG:4326").boundingbox
    return (b.left, b.bottom, b.right, b.top)


def fetch_one(row, max_items: int, flip: bool):
    """(2, H, W) uint16 S1 DN on the chip's stored orientation, or None."""
    from subdetect.imagery import s1_composite

    gbox, profile = _geobox(row["image"])
    res = s1_composite(_bbox_wgs84(gbox), WINDOW, max_items=max_items, geobox=gbox)
    if res is None:
        return None, profile
    arr = res[0]
    if arr.shape[1:] != (profile["height"], profile["width"]):
        raise RuntimeError(f"shape {arr.shape} != chip {profile['height']}x{profile['width']}")
    if flip:
        arr = arr[:, ::-1, :].copy()
    return arr, profile


def _write(path: Path, arr: np.ndarray, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with rasterio.open(tmp, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                       count=2, dtype="uint16", crs=profile["crs"],
                       transform=profile["transform"], nodata=0, compress="deflate",
                       predictor=2) as dst:
        dst.write(arr)
    os.replace(tmp, path)


def todo_rows(index: Path) -> pd.DataFrame:
    d = pd.read_parquet(index)
    return d[d["s1"].isna()].reset_index(drop=True)


def cmd_fetch(a) -> None:
    rows = todo_rows(Path(a.index))
    out_dir = Path(a.out_dir)
    (out_dir / "s1").mkdir(parents=True, exist_ok=True)
    rows = rows[~rows["chip_id"].map(lambda c: (out_dir / "s1" / f"{c}.tif").exists())]
    # Hard negatives first (2,600: they keep the corpus's negative share whatever stops the
    # run), then TorchGeo in a fixed random order, so a partial fetch is a random subset.
    rows = rows.sample(frac=1.0, random_state=0)
    rows = pd.concat([rows[rows.aoi != "torchgeo_substation"],
                      rows[rows.aoi == "torchgeo_substation"]], ignore_index=True)
    if a.limit:
        rows = rows.head(a.limit)
    log.info("%d chips to fetch (workers %d, max_items %d)", len(rows), a.workers, a.max_items)
    fail_csv = out_dir / "failures.csv"
    t0, done, failed = time.time(), 0, 0

    def work(row):
        path = out_dir / "s1" / f"{row['chip_id']}.tif"
        try:
            arr, profile = fetch_one(row, a.max_items, row["aoi"] in FLIPPED_AOIS)
        except Exception as e:  # noqa: BLE001 - logged, retried on the next run
            return row["chip_id"], f"error: {type(e).__name__}: {str(e)[:200]}"
        if arr is None:
            return row["chip_id"], "no scenes"
        valid = float((arr > 0).all(axis=0).mean())
        if valid < a.min_valid:
            return row["chip_id"], f"valid {valid:.2f}"
        _write(path, arr, profile)
        return row["chip_id"], None

    import dask

    dask.config.set(num_workers=max(1, 8 // a.workers))
    records = [r for _, r in rows.iterrows()]
    with ThreadPoolExecutor(a.workers) as ex:
        for i in range(0, len(records), a.workers * 4):  # bounded: never queue every task
            for cid, err in ex.map(work, records[i:i + a.workers * 4]):
                if err:
                    failed += 1
                    with open(fail_csv, "a") as f:
                        f.write(f"{time.strftime('%F %T')},{cid},{err}\n")
                else:
                    done += 1
            n = done + failed
            if n and (n % 200 < a.workers * 4 or n == len(records)):
                rate = (time.time() - t0) / n
                log.info("[%d/%d] %d written, %d failed, %.1f s/chip, eta %.1f h", n,
                         len(records), done, failed, rate, rate * (len(records) - n) / 3600)
    write_index(Path(a.index), out_dir)


def write_index(index: Path, out_dir: Path) -> None:
    rows = todo_rows(index)
    paths = rows["chip_id"].map(lambda c: out_dir / "s1" / f"{c}.tif")
    have = paths.map(Path.exists)
    rows = rows[have].copy()
    rows["s1"] = paths[have].map(str)
    rows[["chip_id", "aoi", "s1"]].to_parquet(out_dir / "index.parquet")
    log.info("index: %d of %d chips have S1 -> %s", len(rows), len(have), out_dir / "index.parquet")


def cmd_check_flip(a) -> None:
    """For TorchGeo chips, correlate the fetched S1 against the stored S2 in both orientations.
    S1 VH and S2 SWIR both rise over built-up land and fall over water, so the right
    orientation should correlate clearly better."""
    rows = todo_rows(Path(a.index))
    rows = rows[rows["aoi"] == "torchgeo_substation"].sample(a.n, random_state=a.seed)
    wins = 0
    for _, row in rows.iterrows():
        arr, _ = fetch_one(row, a.max_items, flip=False)  # north-up, as fetched
        if arr is None:
            log.info("%s: no scenes", row["chip_id"])
            continue
        with rasterio.open(row["image"]) as src:
            swir = src.read(9).astype("float32")  # B11, stored orientation
        vh = arr[1].astype("float32")

        def corr(x):
            ok = (x > 0) & (swir > 0)
            return float(np.corrcoef(x[ok], swir[ok])[0, 1])

        c_as, c_flip = corr(vh), corr(vh[::-1])
        wins += c_flip > c_as
        log.info("%s lat %.2f: corr as fetched %.3f, flipped %.3f", row["chip_id"], row["lat"],
                 c_as, c_flip)
    log.info("flipped orientation correlates better on %d of %d chips", wins, len(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "check-flip", "index"):
        p = sub.add_parser(name)
        p.add_argument("--index", default=str(BASE_INDEX))
        p.add_argument("--out-dir", default=str(OUT))
        p.add_argument("--max-items", type=int, default=16)
    sub.choices["fetch"].add_argument("--workers", type=int, default=2)
    sub.choices["fetch"].add_argument("--limit", type=int, default=0)
    sub.choices["fetch"].add_argument("--min-valid", type=float, default=0.9)
    sub.choices["check-flip"].add_argument("--n", type=int, default=12)
    sub.choices["check-flip"].add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("rasterio", "distributed", "urllib3", "azure"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if a.cmd == "fetch":
        cmd_fetch(a)
    elif a.cmd == "check-flip":
        cmd_check_flip(a)
    else:
        write_index(Path(a.index), Path(a.out_dir))


if __name__ == "__main__":
    main()
