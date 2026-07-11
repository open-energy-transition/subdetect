"""Re-rank a candidates.parquet with Sentinel-1 VH backscatter (metal prior).

Samples dry-season S1 RTC VV/VH (Planetary Computer) at each candidate centroid,
adds s1_vh_db / s1_vv_db / s1_prior columns and rank_score_s1 = rank_score * s1_prior.
Resumable: samples are appended to <candidates dir>/s1_samples_partial.csv as they
arrive; on restart, already-sampled candidates are skipped.

Usage: pixi run python scripts/rerank_s1.py data/predictions_v3b/india_pilot/candidates.parquet
"""
import sys, threading, warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.warp
from rasterio.windows import from_bounds
import planetary_computer, pystac_client

DATE = "2025-11-01/2026-03-15"
WINDOW_M = 60
WORKERS = 6
# logistic prior on VH: centered between substation (-11.4) and bare-land (-15.8) medians
VH_CENTER, VH_SCALE = -13.6, 1.5

cand_path = Path(sys.argv[1])
out_dir = cand_path.parent
partial = out_dir / "s1_samples_partial.csv"

cands = gpd.read_parquet(cand_path)
cands = cands.reset_index(drop=True)
pts = cands.geometry.centroid
todo = set(range(len(cands)))
if partial.exists():
    done = pd.read_csv(partial)
    todo -= set(done["idx"].astype(int))
    print(f"resume: {len(done)} sampled, {len(todo)} to go", flush=True)

_tl = threading.local()
def catalog():
    if not hasattr(_tl, "cat"):
        _tl.cat = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace)
    return _tl.cat

def sample(idx):
    lon, lat = float(pts.x[idx]), float(pts.y[idx])
    try:
        items = list(catalog().search(collections=["sentinel-1-rtc"],
                                      intersects={"type": "Point", "coordinates": [lon, lat]},
                                      datetime=DATE, max_items=2).items())
    except Exception:
        return idx, np.nan, np.nan
    vv_l, vh_l = [], []
    for it in items:
        try:
            vals = {}
            for pol in ("vv", "vh"):
                with rasterio.open(it.assets[pol].href) as src:
                    xs, ys = rasterio.warp.transform("EPSG:4326", src.crs, [lon], [lat])
                    win = from_bounds(xs[0]-WINDOW_M, ys[0]-WINDOW_M,
                                      xs[0]+WINDOW_M, ys[0]+WINDOW_M, src.transform)
                    a = src.read(1, window=win, boundless=True, fill_value=0).astype("float32")
                    a = a[a > 0]
                    if a.size < 10:
                        vals = None; break
                    vals[pol] = float(10*np.log10(a.mean()))
            if vals:
                vv_l.append(vals["vv"]); vh_l.append(vals["vh"])
        except Exception:
            continue
    if not vv_l:
        return idx, np.nan, np.nan
    return idx, float(np.median(vv_l)), float(np.median(vh_l))

lock = threading.Lock()
n_done = 0
if todo:
    if not partial.exists():
        partial.write_text("idx,s1_vv_db,s1_vh_db\n")
    with ThreadPoolExecutor(WORKERS) as ex, open(partial, "a") as f:
        futs = [ex.submit(sample, i) for i in sorted(todo)]
        for fut in as_completed(futs):
            idx, vv, vh = fut.result()
            with lock:
                f.write(f"{idx},{vv},{vh}\n"); f.flush()
                n_done += 1
                if n_done % 100 == 0:
                    print(f"{n_done}/{len(todo)} sampled", flush=True)

s1 = pd.read_csv(partial).drop_duplicates("idx").set_index("idx").sort_index()
cands["s1_vv_db"] = s1["s1_vv_db"].reindex(range(len(cands))).values
cands["s1_vh_db"] = s1["s1_vh_db"].reindex(range(len(cands))).values
vh = cands["s1_vh_db"].to_numpy(float)
prior = 1.0 / (1.0 + np.exp(-(vh - VH_CENTER) / VH_SCALE))
prior[np.isnan(vh)] = 0.5  # no data -> neutral
cands["s1_prior"] = prior.round(4)
cands["rank_score_s1"] = (cands["rank_score"].to_numpy(float) * prior).round(4)
cands = cands.sort_values("rank_score_s1", ascending=False).reset_index(drop=True)

out_pq = out_dir / "candidates_s1.parquet"
out_gj = out_dir / "candidates_s1.geojson"
cands.to_parquet(out_pq)
cands.to_file(out_gj, driver="GeoJSON")
n = cands["s1_vh_db"].notna().sum()
print(f"wrote {out_pq} and {out_gj} ({n}/{len(cands)} sampled)")
print("VH dB by status:\n", cands.groupby("status")["s1_vh_db"].describe().round(1)[["count","mean","50%"]])
for name, d in (("all", cands),):
    for topn in (100, 400):
        t_old = cands.sort_values("rank_score", ascending=False).head(topn)
        t_new = cands.head(topn)
        print(f"top{topn}: known by rank_score={int((t_old.status=='known').sum())} "
              f"vs by rank_score_s1={int((t_new.status=='known').sum())}")
