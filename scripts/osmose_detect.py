"""Osmose-guided regional substation detection — end to end.

For a country/state/province (Osmose country code):
  1. Fetch Osmose 'unfinished major power line' issues (item 7040, class 2).
  2. Drop endpoints within --sub-dist-m (default 700 m) of any OSM substation
     (fetched live from Overpass for the region — works anywhere, no local labels).
  3. Select all 0.1 deg cells within --search-km (default 10 km) of a surviving endpoint.
  4. Compose Sentinel-2 + Sentinel-1 dry-season composites for those cells (resumable).
  5. Tiled dual-model inference with the established best stack:
     P = P_S1only * (0.5 + 0.5 * P_S2only)   (soft optical gate; re-validated on the
     v5 s1only/s2only arms via scripts/eval_decision_fusion.py: bestIoU 0.238 @ thr 0.3,
     recall>=20k 15/18 on pakistan val. NOTE: plain "mean" fusion scored higher
     (bestIoU 0.254 @ thr 0.5, same 15/18 recall) on the same v5 arms -- not yet
     adopted here because the seed/grow hysteresis below was tuned around s1_gated's
     0.5 plateau property, so switching formulas needs a field re-validation pass
     first, not just a val-chip check.)
  6. Polygonize with hysteresis (seed 0.4, grow 0.2 -- seed kept below the 0.5
     fusion plateau so S1-only detections survive; validated on sindh_test +
     yunnan pilot vs the old single 0.3 threshold: fewer fragments, +1 recovered
     substation each, AUC unchanged), flag (not drop) candidates below the 20k m2
     area floor, rank by confidence (= component max prob; robust alternatives
     tested and not better) x endpoint-proximity x line-proximity priors (the
     line prior is the big one: sindh_test AUC 0.769 -> 0.967); write
     review-ready GeoJSON with conf_p90/conf_mean/n_pixels/line_dist_m/
     below_floor as extra review columns, below-floor rows sunk to the bottom.

Usage:
  pixi run -e ml python scripts/osmose_detect.py --region punjab_in --country india_punjab
  pixi run -e ml python scripts/osmose_detect.py --region sindh --country pakistan \
      --bbox 66.0,24.0,69.5,26.5 --dry-run      # fetch/filter/cell-plan only

Outputs under data/osmose_regions/<region>/:
  endpoints.geojson   composites/<cell>/composite_{0,s1}.tif   prob/<cell>.tif   leads.geojson
"""
import argparse, logging, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rasterio
import rasterio.warp
from rasterio.windows import Window
from shapely.geometry import Point, box

sys.path.insert(0, "src")
from subdetect.imagery import annual_composite, s1_composite
from subdetect.config import Settings, geodesic_area_m2
from subdetect.roi import CELL_DEG
from subdetect.postprocess import polygonize_chips_v2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("osmose_detect")

OSMOSE_API = "https://osmose.openstreetmap.fr/api/0.3/issues"
OVERPASS_API = "https://overpass-api.de/api/interpreter"
UA = {"User-Agent": "subdetect-osmose-detect/0.1 (OpenEnergyTransition; substation mapping research)"}
EQ = "EPSG:6933"
S1_CKPT = "data/models/stageA_v5_s1only/terramind-sub-epoch=24-step=6975.ckpt"
S2_CKPT = "data/models/stageA_v5_s2only/terramind-sub-epoch=40-step=11439.ckpt"
WIN = 224


# ---------------------------------------------------------------- step 1: Osmose
def fetch_osmose(country: str, bbox=None, item=7040, klass="2", tile_deg=2.0, limit=500):
    """Country query first (possibly truncated) to find the extent, then bbox-tiled."""
    if bbox is None:
        r = requests.get(OSMOSE_API, params={"item": item, "class": klass, "country": country,
                                             "limit": limit, "status": "open"},
                         headers=UA, timeout=60)
        r.raise_for_status()
        seed = [(i["lon"], i["lat"]) for i in r.json().get("issues", [])]
        if not seed:
            return pd.DataFrame()
        lons, lats = zip(*seed)
        bbox = (min(lons) - 0.5, min(lats) - 0.5, max(lons) + 0.5, max(lats) + 0.5)
        log.info("country=%s extent from seed query: %s", country, [round(b, 2) for b in bbox])
    rows, lon = [], bbox[0]
    while lon < bbox[2]:
        lat = bbox[1]
        while lat < bbox[3]:
            tb = f"{lon},{lat},{min(lon+tile_deg, bbox[2])},{min(lat+tile_deg, bbox[3])}"
            r = requests.get(OSMOSE_API, params={"item": item, "class": klass, "status": "open",
                                                 "limit": limit, "bbox": tb, "country": country},
                             headers=UA, timeout=60)
            r.raise_for_status()
            issues = r.json().get("issues", [])
            if len(issues) >= limit:
                log.warning("tile %s hit the %d cap", tb, limit)
            rows += [{"osmose_id": i["id"], "lat": i["lat"], "lon": i["lon"]} for i in issues]
            time.sleep(0.5)
            lat += tile_deg
        lon += tile_deg
    df = pd.DataFrame(rows).drop_duplicates("osmose_id") if rows else pd.DataFrame(rows)
    log.info("country=%s: %d unique open issues", country, len(df))
    return df


# ------------------------------------------------------------- step 2: Overpass
def fetch_power_lines(bbox) -> gpd.GeoDataFrame:
    """OSM power=line ways in bbox as LineStrings (for the line-proximity prior)."""
    from shapely.geometry import LineString

    q = f"""[out:json][timeout:180];
    way["power"="line"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    out geom;"""
    r = requests.post(OVERPASS_API, data={"data": q}, headers=UA, timeout=240)
    r.raise_for_status()
    geoms = []
    for el in r.json().get("elements", []):
        coords = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(coords) >= 2:
            geoms.append(LineString(coords))
    log.info("Overpass: %d power lines in region", len(geoms))
    return gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")


def fetch_substations(bbox) -> gpd.GeoDataFrame:
    """All OSM power=substation centers in bbox via Overpass (any size, incl. nodes)."""
    q = f"""[out:json][timeout:180];
    nwr["power"="substation"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    out center;"""
    r = requests.post(OVERPASS_API, data={"data": q}, headers=UA, timeout=240)
    r.raise_for_status()
    pts = []
    for el in r.json().get("elements", []):
        c = el.get("center", el)
        if "lat" in c and "lon" in c:
            pts.append(Point(c["lon"], c["lat"]))
    log.info("Overpass: %d substations in region", len(pts))
    return gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326")


# ------------------------------------------------------- steps 4-5: compose+infer
def compose_cell(cell_dir: Path, cbox, s2_window, s1_window):
    from odc.geo.geobox import GeoBox
    s2_tif = cell_dir / "composite_0.tif"
    s1_tif = cell_dir / "composite_s1.tif"
    try:
        if not s2_tif.exists():
            res = annual_composite(cbox, date_range=s2_window)
            if res is None:
                return False
            _write(s2_tif, *res)
        if not s1_tif.exists():
            with rasterio.open(s2_tif) as b:
                gbox = GeoBox((b.height, b.width), b.transform, b.crs)
            res = s1_composite(cbox, date_range=s1_window, geobox=gbox)
            if res is None:
                return False
            _write(s1_tif, *res)
    except Exception as e:  # noqa: BLE001 — one bad cell must not kill the run
        log.warning("cell %s failed: %s", cell_dir.name, e)
        return False
    return True


def _write(tif, arr, transform, crs):
    tif.parent.mkdir(parents=True, exist_ok=True)
    tmp = tif.with_suffix(".tif.tmp")
    with rasterio.open(tmp, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                       count=arr.shape[0], dtype="uint16", crs=crs, transform=transform,
                       compress="deflate", predictor=2) as dst:
        dst.write(arr)
    tmp.rename(tif)


def gated_inference(cells, comp_dir: Path, prob_dir: Path, delete_composites=False):
    import torch
    from subdetect.infer import load_model, _standardize_s1

    prob_dir.mkdir(parents=True, exist_ok=True)
    tasks = {}
    for name, ckpt in (("s1", S1_CKPT), ("s2", S2_CKPT)):
        task, device = load_model(Path(ckpt))
        mods = task.hparams.get("model_args", {}).get("backbone_modalities", ["S2L2A"])
        tasks[name] = (task, mods)
    hann = np.outer(np.hanning(WIN), np.hanning(WIN)).astype("float32") + 1e-3
    stride = 104

    def predict(task, mods, s2_np, s1_np):
        s2_t = torch.from_numpy(s2_np / 10000.0)[None].to(device)
        x = ({m: (s2_t if m == "S2L2A" else torch.from_numpy(_standardize_s1(s1_np))[None].to(device))
              for m in mods} if "S1RTC" in mods else s2_t)
        with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
            out = task(x)
            logits = out.output if hasattr(out, "output") else out
            return torch.softmax(logits, 1)[0, 1].float().cpu().numpy()

    for cell in cells:
        out_tif = prob_dir / f"{cell}.tif"
        if out_tif.exists():
            continue
        s2_path = comp_dir / cell / "composite_0.tif"
        s1_path = comp_dir / cell / "composite_s1.tif"
        if not (s2_path.exists() and s1_path.exists()):
            continue
        with rasterio.open(s2_path) as src, rasterio.open(s1_path) as src1:
            H, W, transform, crs = src.height, src.width, src.transform, src.crs
            acc = np.zeros((H, W), "float32"); wacc = np.zeros((H, W), "float32")
            valid_any = np.zeros((H, W), bool)
            rows = sorted(set(list(range(0, max(H - WIN, 0) + 1, stride)) + [max(H - WIN, 0)]))
            cols = sorted(set(list(range(0, max(W - WIN, 0) + 1, stride)) + [max(W - WIN, 0)]))
            for r in rows:
                for c in cols:
                    win = Window(c, r, WIN, WIN)
                    s2_np = src.read(window=win, boundless=True, fill_value=0).astype("float32")[:10]
                    h, w = min(WIN, H - r), min(WIN, W - c)
                    if (s2_np[:, :h, :w] > 0).mean() < 0.2:
                        continue
                    s1_np = src1.read(window=win, boundless=True, fill_value=0).astype("float32")[:2]
                    p1 = predict(*tasks["s1"], s2_np, s1_np)
                    p2 = predict(*tasks["s2"], s2_np, s1_np)
                    prob = p1 * (0.5 + 0.5 * p2)  # soft optical gate
                    acc[r:r+h, c:c+w] += prob[:h, :w] * hann[:h, :w]
                    wacc[r:r+h, c:c+w] += hann[:h, :w]
                    valid_any[r:r+h, c:c+w] |= (s2_np[:, :h, :w] > 0).any(axis=0)
        prob_full = np.where(wacc > 0, acc / np.maximum(wacc, 1e-6), 0.0)
        prob_full[~valid_any] = 0.0
        with rasterio.open(out_tif, "w", driver="GTiff", width=W, height=H, count=1,
                           dtype="uint8", crs=crs, transform=transform,
                           compress="deflate", predictor=2) as dst:
            dst.write((np.clip(prob_full, 0, 1) * 255).astype("uint8"), 1)
        if delete_composites:
            shutil.rmtree(comp_dir / cell, ignore_errors=True)
        log.info("inferred %s", cell)


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, help="Output name under data/osmose_regions/")
    ap.add_argument("--country", required=True, help="Osmose country code (e.g. india_punjab)")
    ap.add_argument("--bbox", help="lon1,lat1,lon2,lat2 (else derived from the issues)")
    ap.add_argument("--sub-dist-m", type=float, default=700.0)
    ap.add_argument("--search-km", type=float, default=10.0)
    ap.add_argument("--hi", type=float, default=0.4,
                    help="Hysteresis seed threshold (keep below the 0.5 fusion plateau)")
    ap.add_argument("--lo", type=float, default=0.2, help="Hysteresis grow threshold")
    ap.add_argument("--line-decay-m", type=float, default=500.0,
                    help="Line-proximity prior decay (m); exp(-line_dist/this)")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent compose cells")
    ap.add_argument("--limit-cells", type=int, default=0)
    ap.add_argument("--tile-deg", type=float, default=2.0, help="Osmose fetch tile size")
    ap.add_argument("--batch-cells", type=int, default=0,
                    help="Compose+infer in batches of N cells (0 = all at once)")
    ap.add_argument("--delete-composites", action="store_true",
                    help="Remove a cell's composites once its prob raster is written")
    ap.add_argument("--dry-run", action="store_true", help="Stop after cell planning")
    a = ap.parse_args()

    settings = Settings.load()
    out = Path("data/osmose_regions") / a.region
    out.mkdir(parents=True, exist_ok=True)
    bbox = tuple(float(x) for x in a.bbox.split(",")) if a.bbox else None

    issues = fetch_osmose(a.country, bbox, tile_deg=a.tile_deg)
    if issues.empty:
        log.info("no issues; done"); return
    pts = gpd.GeoDataFrame(issues, geometry=[Point(xy) for xy in zip(issues.lon, issues.lat)],
                           crs="EPSG:4326")
    ext = pts.total_bounds
    subs = fetch_substations((ext[0]-0.2, ext[1]-0.2, ext[2]+0.2, ext[3]+0.2))
    if not subs.empty:
        near = gpd.sjoin_nearest(pts.to_crs(EQ), subs.to_crs(EQ)[["geometry"]],
                                 how="left", distance_col="sub_dist_m")
        near = near[~near.index.duplicated(keep="first")]
        pts["sub_dist_m"] = near["sub_dist_m"].round(1).values
    else:
        pts["sub_dist_m"] = 1e9
    keep = pts[pts.sub_dist_m > a.sub_dist_m].reset_index(drop=True)
    log.info("%d/%d endpoints are >%.0f m from any mapped substation",
             len(keep), len(pts), a.sub_dist_m)
    keep.to_file(out / "endpoints.geojson", driver="GeoJSON")
    if keep.empty:
        return

    # cell plan: 0.1 deg cells intersecting endpoint buffers
    buf = keep.to_crs(EQ).buffer(a.search_km * 1000).to_crs("EPSG:4326").union_all()
    cells = {}
    b = buf.bounds
    for ix in range(int(np.floor(b[0] / CELL_DEG)), int(np.ceil(b[2] / CELL_DEG))):
        for iy in range(int(np.floor(b[1] / CELL_DEG)), int(np.ceil(b[3] / CELL_DEG))):
            cb = (ix * CELL_DEG, iy * CELL_DEG, (ix + 1) * CELL_DEG, (iy + 1) * CELL_DEG)
            if box(*cb).intersects(buf):
                cells[f"{ix:05d}_{iy:05d}"] = cb
    names = sorted(cells)
    if a.limit_cells:
        names = names[: a.limit_cells]
    log.info("cell plan: %d cells (~%.1f GB S2+S1, ~%.0f min compose at 4 workers)",
             len(names), len(names) * 0.038, len(names) * 2.2 / a.workers * a.workers / 4)
    if a.dry_run:
        return

    comp_dir = out / "composites"
    prob_dir = out / "prob"
    s2w = tuple(settings.raw.get("s2_window", ("2025-11-01", "2026-03-15")))
    s1w = tuple(settings.raw.get("s1_window", ("2025-11-01", "2026-03-15")))
    todo = [n for n in names if not (prob_dir / f"{n}.tif").exists()]
    bs = a.batch_cells or max(len(todo), 1)
    batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]
    cell_ex = ThreadPoolExecutor(a.workers)
    pre_ex = ThreadPoolExecutor(1)

    def compose_batch(batch):
        return list(cell_ex.map(lambda n: compose_cell(comp_dir / n, cells[n], s2w, s1w), batch))

    fut = pre_ex.submit(compose_batch, batches[0]) if batches else None
    for bi, batch in enumerate(batches):
        oks = fut.result()
        log.info("batch %d/%d: composed %d/%d cells", bi + 1, len(batches), sum(oks), len(batch))
        if bi + 1 < len(batches):
            fut = pre_ex.submit(compose_batch, batches[bi + 1])
        gated_inference(batch, comp_dir, prob_dir, delete_composites=a.delete_composites)
    cell_ex.shutdown()
    pre_ex.shutdown()

    cands = polygonize_chips_v2(out / "prob", lo=a.lo, hi=a.hi)
    if cands.empty:
        log.info("no candidates above threshold"); return
    cands["confidence"] = cands.conf_max  # ranking score; p90/mean stay as review columns
    cands["area_m2"] = [geodesic_area_m2(g) for g in cands.geometry]
    # Recall-first: below-floor candidates are flagged and sunk to the bottom of the
    # review order, not dropped (sindh_test: the hard drop cost 1 of 36 detected
    # sub-20k-m2 substations for 46 mostly-FP components saved).
    cands["below_floor"] = cands.area_m2 < settings.min_sub_area_m2
    cu = cands.to_crs(EQ)
    ku = keep.to_crs(EQ)
    dists, n_eps = [], []
    for g in cu.geometry:
        ds = ku.distance(g)
        dists.append(float(ds.min()))
        n_eps.append(int((ds < a.search_km * 1000).sum()))
    cands["endpoint_dist_m"] = np.round(dists, 1)
    cands["n_endpoints_in_radius"] = n_eps
    # Line-proximity prior: substations sit ON power lines (sindh_test: 75% of real
    # hits at line_dist=0, non-hits median 1.8 km; conf*exp(-d/500) lifts AUC
    # 0.769 -> 0.967, P@20 0.70 -> 0.90). 500 m decay, not the tested-best 250 m,
    # hedges against incompletely mapped lines in exactly the regions Osmose flags.
    ext = cands.total_bounds
    try:
        power_lines = fetch_power_lines((ext[0] - 0.1, ext[1] - 0.1, ext[2] + 0.1, ext[3] + 0.1))
    except Exception as e:  # noqa: BLE001 — prior is optional, ranking must not die
        log.warning("power-line fetch failed (%s); ranking without line prior", e)
        power_lines = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if not power_lines.empty:
        near = gpd.sjoin_nearest(cu, power_lines.to_crs(EQ)[["geometry"]],
                                 how="left", distance_col="line_dist_m")
        near = near[~near.index.duplicated(keep="first")]
        cands["line_dist_m"] = near["line_dist_m"].round(1).values
        line_prior = np.exp(-cands.line_dist_m.fillna(0).values / a.line_decay_m)
    else:
        cands["line_dist_m"] = np.nan
        line_prior = 1.0
    cands["rank_score"] = (cands.confidence.fillna(0)
                           * np.exp(-np.array(dists) / 2000.0) * line_prior).round(4)
    cands = cands.sort_values(["below_floor", "rank_score"],
                              ascending=[True, False]).reset_index(drop=True)
    cands.to_file(out / "leads.geojson", driver="GeoJSON")
    log.info("wrote %s: %d leads (%d below the %d m2 floor, sunk not dropped; "
             "%d within %.1f km of an endpoint)",
             out / "leads.geojson", len(cands), int(cands.below_floor.sum()),
             int(settings.min_sub_area_m2),
             int((cands.endpoint_dist_m < a.search_km * 1000).sum()), a.search_km)


if __name__ == "__main__":
    main()
