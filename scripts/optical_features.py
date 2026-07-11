"""Per-candidate optical features from local S2 composites: roof-vs-gravel test.

For each candidate polygon, samples the containing composite cell and computes
mean visible brightness, NDVI, and within-polygon brightness texture (std).
Validates: known substations vs the S1-top-150 new leads (user-reviewed ~industrial).

Usage: pixi run python scripts/optical_features.py <candidates parquet> <aoi>
"""
import sys, glob
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.warp
from rasterio import features as rio_features

cand_path, aoi = Path(sys.argv[1]), sys.argv[2]
cands = gpd.read_parquet(cand_path).reset_index(drop=True)

# index composite cells by bounds (WGS84)
cells = []
for p in glob.glob(f"data/composites/{aoi}/composites/*/composite_0.tif"):
    with rasterio.open(p) as src:
        b = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        cells.append((b, p))

def cell_for(lon, lat):
    for b, p in cells:
        if b[0] <= lon <= b[2] and b[1] <= lat <= b[3]:
            return p
    return None

bright = np.full(len(cands), np.nan)   # mean of B02..B04 reflectance
ndvi = np.full(len(cands), np.nan)
tex = np.full(len(cands), np.nan)      # std of visible brightness within polygon
DN_OFFSET, SCALE = 1000.0, 10000.0

pts = cands.geometry.centroid
for i, (geom, lon, lat) in enumerate(zip(cands.geometry, pts.x, pts.y)):
    p = cell_for(lon, lat)
    if p is None:
        continue
    with rasterio.open(p) as src:
        g = rasterio.warp.transform_geom("EPSG:4326", src.crs, geom.__geo_interface__)
        try:
            mask = rio_features.geometry_mask([g], out_shape=(src.height, src.width),
                                              transform=src.transform, invert=True)
        except Exception:
            continue
        if mask.sum() < 4:
            continue
        rows, cols = np.where(mask)
        r0, r1, c0, c1 = rows.min(), rows.max()+1, cols.min(), cols.max()+1
        win = rasterio.windows.Window(c0, r0, c1-c0, r1-r0)
        arr = src.read([1, 2, 3, 7], window=win).astype("float32")  # B02,B03,B04,B08
        m = mask[r0:r1, c0:c1] & (arr > 0).all(axis=0)
        if m.sum() < 4:
            continue
        refl = np.clip((arr - DN_OFFSET) / SCALE, 0, 1)
        vis = refl[:3].mean(axis=0)
        red, nir = refl[2], refl[3]
        bright[i] = float(vis[m].mean())
        tex[i] = float(vis[m].std())
        nd = (nir - red) / np.maximum(nir + red, 1e-6)
        ndvi[i] = float(nd[m].mean())

cands["opt_bright"] = np.round(bright, 4)
cands["opt_tex"] = np.round(tex, 4)
cands["opt_ndvi"] = np.round(ndvi, 4)
out = cand_path.parent / (cand_path.stem.replace("_bld", "") + "_opt.parquet")
cands.to_parquet(out)
print("wrote", out)

from scipy.stats import mannwhitneyu
kn = cands[cands.status == "known"].dropna(subset=["opt_bright"])
key = "rank_score_s1" if "rank_score_s1" in cands else "rank_score"
fp = cands[cands.status == "new"].sort_values(key, ascending=False).head(150).dropna(subset=["opt_bright"])
for col in ("opt_bright", "opt_tex", "opt_ndvi"):
    u, p = mannwhitneyu(kn[col], fp[col])
    print(f"{col}: AUC={u/(len(kn)*len(fp)):.3f} known median {kn[col].median():.3f} vs fp {fp[col].median():.3f}")
