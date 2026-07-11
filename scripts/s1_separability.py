"""Does Sentinel-1 backscatter separate real substations from bare-land FPs?

Real set:  v2_india candidates with status=known (model detections on OSM subs).
FP set:    the manually-reviewed top-400 new leads (user: ~1/258 real -> treat as FPs).
For each centroid, read a small VV/VH window from 1-2 dry-season S1 RTC scenes
(Planetary Computer) and compare dB statistics.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.warp
from rasterio.windows import from_bounds
import planetary_computer, pystac_client

ROOT = Path("/run/media/tobi/aidisc/subdetect")
OUT = Path("/tmp/claude-1000/-run-media-tobi-aidisc-substation-seg/0b416b8b-a654-4340-a785-184c8ada0564/scratchpad/s1_sep")
OUT.mkdir(exist_ok=True)
WINDOW_M = 60          # half-size of sample box around centroid
DATE = "2025-11-01/2026-03-15"
MAX_PER_CLASS = 150

cat = pystac_client.Client.open("https://planetarycatalog.microsoft.com/api/stac/v1"
                                if False else "https://planetarycomputer.microsoft.com/api/stac/v1",
                                modifier=planetary_computer.sign_inplace)

known = gpd.read_parquet(ROOT / "data/predictions_v2_india/pakistan/candidates.parquet")
known = known[known.status == "known"].sample(min(MAX_PER_CLASS, 226), random_state=42)
fp_file = None
for cand in ["data/predictions_v2_india/pakistan/pakistan_sub_new_leads_top400.geojson",
             "data/exports/pakistan_sub_new_leads_top400.geojson"]:
    if (ROOT / cand).exists():
        fp_file = ROOT / cand
        break
if fp_file is None:
    hits = list(ROOT.rglob("*new_leads_top400*.geojson"))
    fp_file = hits[0] if hits else None
print("FP set file:", fp_file)
fps = gpd.read_file(fp_file)
if len(fps) > MAX_PER_CLASS:
    fps = fps.sample(MAX_PER_CLASS, random_state=42)

def sample_point(lon, lat):
    """Return (vv_db, vh_db) mean over WINDOW at point, median across up to 2 scenes."""
    try:
        items = list(cat.search(collections=["sentinel-1-rtc"],
                                intersects={"type": "Point", "coordinates": [lon, lat]},
                                datetime=DATE, max_items=2).items())
    except Exception:
        return None
    vals = []
    for it in items:
        try:
            row = {}
            for pol in ("vv", "vh"):
                with rasterio.open(it.assets[pol].href) as src:
                    xs, ys = rasterio.warp.transform("EPSG:4326", src.crs, [lon], [lat])
                    win = from_bounds(xs[0]-WINDOW_M, ys[0]-WINDOW_M, xs[0]+WINDOW_M, ys[0]+WINDOW_M,
                                      src.transform)
                    a = src.read(1, window=win, boundless=True, fill_value=0).astype("float32")
                    a = a[a > 0]
                    if a.size < 10:
                        row = None; break
                    row[pol] = float(10*np.log10(a.mean()))
            if row:
                vals.append(row)
        except Exception as e:
            print("read error:", type(e).__name__, str(e)[:120], flush=True)
            continue
    if not vals:
        return None
    vv = float(np.median([v["vv"] for v in vals]))
    vh = float(np.median([v["vh"] for v in vals]))
    return vv, vh

def run(gdf, label):
    out = []
    pts = gdf.geometry.centroid
    for i, (lon, lat) in enumerate(zip(pts.x, pts.y)):
        r = sample_point(lon, lat)
        if r:
            out.append({"label": label, "lon": lon, "lat": lat, "vv_db": r[0], "vh_db": r[1]})
        if (i+1) % 25 == 0:
            print(f"{label}: {i+1}/{len(gdf)} sampled, {len(out)} ok", flush=True)
    return out

rows = run(known, "substation") + run(fps, "bare_fp")
import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(OUT / "s1_samples.csv", index=False)

print("\n== S1 RTC dry-season backscatter (dB, 120 m box) ==")
print(df.groupby("label")[["vv_db", "vh_db"]].describe().round(1).T)

# simple separability: AUC via rank stats
from scipy.stats import mannwhitneyu
for col in ("vv_db", "vh_db"):
    a = df[df.label == "substation"][col]; b = df[df.label == "bare_fp"][col]
    u, p = mannwhitneyu(a, b, alternative="greater")
    auc = u / (len(a) * len(b))
    print(f"{col}: AUC(sub>fp)={auc:.3f}  p={p:.1e}  median sub={a.median():.1f} fp={b.median():.1f}")
