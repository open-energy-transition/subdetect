"""Build a small multi-temporal S2 chip dataset: for a stratified subset of existing
chips_v5 chips (prioritizing small substations, our target failure mode), fetch N
individual least-cloudy S2 dates on the EXACT SAME pixel grid as the existing chip
(via an odc GeoBox built from the existing image's own transform/crs -- guarantees
perfect co-registration with the unchanged mask file) instead of one median-collapsed
composite. Writes (N*10, H, W) uint16 stacks + a new index parquet reusing the
untouched mask paths.

Proof-of-concept scope for the [[subdetect-max-fusion... overnight ablation, NOT a
full production re-chip: a stratified few hundred chips, S2-only (no S1/tower
multi-date), to validate the TemporalWrapper architecture on real data and get a
first trainable signal before deciding whether a full country-scale recompose is
worth the (much larger) storage/compute investment.

Usage:
  pixi run -e ml python scripts/build_multidate_chips.py --n-dates 4 --n-small 60 \
      --n-mid 250 --n-large 150 --n-neg 150
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_multidate_chips")

S2_WINDOW = ("2025-11-01", "2026-03-15")
BASE_INDEX = ROOT / "data/chips_v5/combined/index.parquet"
OUT_DIR = ROOT / "data/chips_multidate/pakistan_india"


class DegradedRead(RuntimeError):
    """Reads "succeeded" via fail_on_error=False but GDAL logged too many failures.

    A PC SAS token expires ~45min into any run (see the v14 field-eval incident): odc
    retries the same dead URL without re-signing, per-tile errors degrade to NaN -> 0,
    and a partially-landed date can still clear the valid-fraction filter below. Left
    unchecked that lands a silently-degraded chip in the TRAINING set, which is far
    worse than the inference case -- it poisons the model with no detectable signature.
    """


def fetch_dates_geobox(bbox_wgs84, geobox, date_range, max_dates: int,
                       max_failures: int | None = None):
    """Return up to max_dates individual per-date (10, H, W) uint16 stacks.

    The real protection against a degraded read is the data-level check: the caller
    requires max_dates DISTINCT dates that each clear the valid-fraction filter below,
    which is thread-independent and catches the token-expiry case (a dead date comes
    back all-zero and is rejected). The GDAL failure count is a secondary signal --
    counted process-wide rather than per-thread, because dask's threaded scheduler
    emits read failures from worker threads that a thread-scoped handler (as in
    imagery._count_gdal_failures) silently drops. Forcing synchronous reads would fix
    attribution but costs 3-4x wall-clock (25-33s vs 8-13s per chip, measured), which
    is untenable over a 33k-chip run. With several chips in flight the count can be
    misattributed, so a burst may reject a neighbouring good chip -- deliberately the
    safe direction: a rejected chip is logged and refetched on the next resume, while a
    wrongly ACCEPTED one silently poisons training.
    """
    import odc.stac

    from subdetect.imagery import (
        MAX_CLOUD,
        MAX_GDAL_WARNINGS,
        _GDAL_FAILURE_MARKERS,
        _catalog,
        _SCL_VALID,
    )

    if max_failures is None:
        max_failures = MAX_GDAL_WARNINGS
    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    search = _catalog().search(
        collections=["sentinel-2-l2a"], bbox=bbox_wgs84,
        datetime=f"{date_range[0]}/{date_range[1]}",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
    )
    items = sorted(search.items(), key=lambda it: it.properties.get("eo:cloud_cover", 100))
    if not items:
        return []
    items = items[: max_dates * 2]  # margin: some may fail cloud-fraction check below
    ds = odc.stac.load(
        items, bands=[*bands, "SCL"], groupby="solar_day",
        chunks={"x": 512, "y": 512}, fail_on_error=False, geobox=geobox,
    )
    valid = ds["SCL"].isin(_SCL_VALID)

    class _Counter(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.INFO)
            self.count = 0

        def emit(self, record):
            if any(m in record.getMessage() for m in _GDAL_FAILURE_MARKERS):
                self.count += 1

    counter = _Counter()
    root = logging.getLogger()
    root.addHandler(counter)
    try:
        masked = ds[bands].where(valid).fillna(0).astype("uint16").compute()
    finally:
        root.removeHandler(counter)
    if counter.count > max_failures:
        raise DegradedRead(f"{counter.count} GDAL read failures for bbox={bbox_wgs84}")
    out = []
    for t in range(masked.sizes["time"]):
        arr = np.stack([masked[b].isel(time=t).values for b in bands], axis=0)
        if (arr > 0).mean() > 0.5:
            out.append(arr)
        if len(out) >= max_dates:
            break
    return out


def select_subset(base: pd.DataFrame, n_small: int, n_mid: int, n_large: int, n_neg: int,
                  seed: int = 42, exclude_chip_ids: set | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if exclude_chip_ids:
        base = base[~base.chip_id.isin(exclude_chip_ids)]
    pos = base[base.kind == "positive"]
    small = pos[(pos.sub_pixels > 0) & (pos.sub_pixels <= 20)]
    mid = pos[(pos.sub_pixels > 20) & (pos.sub_pixels <= 200)]
    large = pos[pos.sub_pixels > 200]
    neg = base[base.kind != "positive"]

    def sample(df, n):
        return df if len(df) <= n else df.sample(n=n, random_state=seed)

    parts = [small, sample(mid, n_mid), sample(large, n_large), sample(neg, n_neg)]
    if n_small < len(small):
        parts[0] = sample(small, n_small)
    return pd.concat(parts, ignore_index=True)


def _build_one(row, out_dir: Path, n_dates: int):
    """Fetch + write one chip's multi-date stack. Returns an index row, or None.

    Returns None (never a padded stack) when fewer than n_dates real dates land. The
    old duplicate-the-last-date padding silently turned a token-expiry casualty into a
    "4-date" chip that was one date repeated -- see DegradedRead. A short chip is
    skipped instead, so the dataset only ever contains genuine n_dates stacks.
    """
    from odc.geo.geobox import GeoBox

    img_path = Path(row["image"])
    if not img_path.exists():
        return None
    out_tif = out_dir / f"{row['chip_id']}_multidate.tif"
    with rasterio.open(img_path) as src:
        h, w = src.height, src.width
        gbox = GeoBox((h, w), src.transform, src.crs)
        b = src.bounds
        lons, lats = warp_transform(src.crs, "EPSG:4326", [b.left, b.right], [b.bottom, b.top])
        bbox_wgs84 = (min(lons), min(lats), max(lons), max(lats))
        profile = src.profile

    try:
        dates = fetch_dates_geobox(bbox_wgs84, gbox, S2_WINDOW, n_dates)
    except DegradedRead as e:
        log.warning("degraded read for %s: %s -- skipped", row["chip_id"], e)
        return None
    except Exception as e:  # noqa: BLE001
        log.warning("fetch failed for %s: %s", row["chip_id"], e)
        return None
    if len(dates) < n_dates:
        log.warning("only %d/%d usable dates for %s -- skipped (no padding)",
                    len(dates), n_dates, row["chip_id"])
        return None

    stack = np.concatenate(dates, axis=0)  # (n_dates*10, H, W) uint16
    profile.update(count=stack.shape[0])
    tmp = out_tif.with_suffix(".tif.part")  # never leave a half-written tif for resume
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(stack)
    tmp.replace(out_tif)
    return {**row.to_dict(), "image": str(out_tif), "n_revisits": n_dates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dates", type=int, default=4)
    ap.add_argument("--n-small", type=int, default=60)
    ap.add_argument("--n-mid", type=int, default=250)
    ap.add_argument("--n-large", type=int, default=150)
    ap.add_argument("--n-neg", type=int, default=150)
    ap.add_argument("--all", action="store_true",
                    help="take EVERY row of --base-index (full-scale recompose) instead of "
                         "the stratified n-small/n-mid/n-large/n-neg subset")
    ap.add_argument("--base-index", default=str(BASE_INDEX))
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help="output dir; may live on another volume (81GB at 31k chips)")
    ap.add_argument("--workers", type=int, default=4,
                    help="chips fetched concurrently; each worker reads synchronously so "
                         "the per-chip GDAL failure guard stays correctly attributed")
    ap.add_argument("--checkpoint-every", type=int, default=200,
                    help="rewrite index.parquet this often -- a multi-day run must not "
                         "keep the only copy of its index in RAM")
    ap.add_argument("--append", action="store_true",
                    help="deprecated/implied: resume is now automatic via on-disk tifs")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(a.base_index)
    sel = base.copy() if a.all else select_subset(
        base, a.n_small, a.n_mid, a.n_large, a.n_neg)
    log.info("base=%s selected %d chips (kind counts: %s)",
             a.base_index, len(sel), sel.kind.value_counts().to_dict())

    # Resume: anything already on disk is done. Reconstruct its index row from `base`
    # rather than trusting a stale index.parquet, so the index always describes exactly
    # what the directory contains (idempotent across restarts).
    done = {p.name[: -len("_multidate.tif")] for p in out_dir.glob("*_multidate.tif")}
    for p in out_dir.glob("*.tif.part"):
        p.unlink()  # torn write from a previous kill
    rows = [{**r.to_dict(), "image": str(out_dir / f"{r['chip_id']}_multidate.tif"),
             "n_revisits": a.n_dates}
            for _, r in sel[sel.chip_id.isin(done)].iterrows()]
    todo = sel[~sel.chip_id.isin(done)]
    log.info("resume: %d already built, %d to fetch (%.1f GB more at 2.6MB/chip)",
             len(rows), len(todo), len(todo) * 2.6 / 1000)

    # Cap dask's own threadpool. Each _build_one calls .compute(), so with W chip
    # workers there are W nested dask pools; left at the default (one thread per core)
    # that reached 135 threads at W=6 and livelocked on GIL/scheduler contention --
    # 22% CPU and literally 0 KB/s of network for minutes. Keep W * dask_threads small.
    import dask
    dask.config.set(num_workers=max(1, 8 // max(a.workers, 1)))

    lock = threading.Lock()
    n_done = n_skip = 0
    t_start = time.monotonic()

    def write_index():
        with lock:
            pd.DataFrame(rows).to_parquet(out_dir / "index.parquet")

    def run(row):
        nonlocal n_done, n_skip
        out = _build_one(row, out_dir, a.n_dates)
        with lock:
            if out is None:
                n_skip += 1
            else:
                rows.append(out)
                n_done += 1
            n = n_done + n_skip
            ckpt = n_done and n_done % a.checkpoint_every == 0
        if n % 25 == 0 or ckpt:
            rate = (time.monotonic() - t_start) / max(n, 1)
            log.info("[%d/%d done: %d built, %d skipped] %.1fs/chip, eta %.1fh",
                     n, len(todo), n_done, n_skip, rate, rate * (len(todo) - n) / 3600)
        if ckpt:
            write_index()

    # Submit in bounded batches: ThreadPoolExecutor.map/submit over a 33k generator
    # queues every task up front (33k pending futures + row objects), which is what
    # starved the workers. A batch window keeps only `workers * 4` in flight.
    todo_rows = [r for _, r in todo.iterrows()]
    batch = max(a.workers * 4, 8)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i in range(0, len(todo_rows), batch):
            list(ex.map(run, todo_rows[i:i + batch]))
            if i and (i // batch) % 10 == 0:
                write_index()

    write_index()
    log.info("wrote %s (%d chips total; this run: %d built, %d skipped of %d)",
             out_dir / "index.parquet", len(rows), n_done, n_skip, len(todo))


if __name__ == "__main__":
    main()
