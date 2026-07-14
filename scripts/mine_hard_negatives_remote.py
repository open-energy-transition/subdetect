"""Mine hard-negative training chips from terrain far from any known building.

Complements `mine_hard_negatives_yunnan.py` (which mines the model's own low-confidence
detections) with an inference-independent strategy: the documented false-positive
pattern for this model is bare/exposed natural land confused for a substation's gravel
yard -- i.e. remote terrain, not built-up area. Sampling points that are far from any
known building (VIDA Google+Microsoft Open Buildings, via DuckDB httpfs, bbox/cell-
scoped so no country-wide download) and far from any real substation directly targets
that terrain type, and needs no prior inference pass over the region.

Usage: python scripts/mine_hard_negatives_remote.py [n] --region yunnan --iso3 CHN
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from subdetect.chips import _bbox_poly, _burn_mask, _chip_bbox, _chip_id, _crop, _write_tif  # noqa: E402
from subdetect.config import CHIP_SIZE, MODEL_BANDS, S1_BANDS, Settings  # noqa: E402
from subdetect.local_source import CompositeIndex  # noqa: E402
from osmose_detect import fetch_substations  # noqa: E402

EQ = "EPSG:6933"
VIDA_URL = (
    "https://data.source.coop/vida/google-microsoft-open-buildings/"
    "geoparquet/by_country/country_iso={iso3}/{iso3}.parquet"
)
BUILDING_FAR_M = 1000.0     # "far from any building" -- remote/uninhabited terrain
SUB_CLEARANCE_M = 1500.0    # matches chips.py's LINE_NEG_CLEARANCE_M convention
DEDUP_DEG = 0.02            # ~2 km, matches the other mining scripts


def _fetch_building_points(cells_bounds, iso3: str) -> gpd.GeoDataFrame:
    """VIDA building centers over the union of cell bboxes; points only (no polygon
    decode) since only nearest-distance is needed -- cheap even at cell scale."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    url = VIDA_URL.format(iso3=iso3)
    xmin, ymin, xmax, ymax = cells_bounds
    sql = f"""
        SELECT (bbox.xmin + bbox.xmax) / 2 AS lon, (bbox.ymin + bbox.ymax) / 2 AS lat
        FROM read_parquet('{url}')
        WHERE bbox.xmin <= {xmax} AND bbox.xmax >= {xmin}
          AND bbox.ymin <= {ymax} AND bbox.ymax >= {ymin}
    """
    df = con.execute(sql).df()
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    )


def main(n_target: int, region: str, iso3: str) -> None:
    settings = Settings.load()
    region_dir = ROOT / "data" / "osmose_regions" / region
    comp_idx = CompositeIndex(region_dir)
    coverage = comp_idx.coverage
    b = coverage.bounds
    print(f"Coverage: {len(comp_idx.index)} composited cells, bounds={[round(v, 3) for v in b]}")

    buildings = _fetch_building_points((b[0], b[1], b[2], b[3]), iso3)
    print(f"Fetched {len(buildings)} VIDA building points over the coverage bbox")

    subs = fetch_substations((b[0] - 0.05, b[1] - 0.05, b[2] + 0.05, b[3] + 0.05))
    print(f"Fetched {len(subs)} real OSM substations for the clearance + ignore mask")

    # Dense candidate grid over the coverage (~300 m spacing), then filter.
    rng = np.random.default_rng(42)
    minx, miny, maxx, maxy = b
    grid_deg = 300.0 / 111320.0
    xs = np.arange(minx, maxx, grid_deg)
    ys = np.arange(miny, maxy, grid_deg)
    gx, gy = np.meshgrid(xs, ys)
    jitter = grid_deg * 0.4
    lons = gx.ravel() + rng.uniform(-jitter, jitter, gx.size)
    lats = gy.ravel() + rng.uniform(-jitter, jitter, gy.size)
    cand = gpd.GeoDataFrame(
        {"lon": lons, "lat": lats}, geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326"
    )
    cand = cand[cand.within(coverage)].reset_index(drop=True)
    print(f"{len(cand)} candidate grid points inside composited coverage")

    cand_eq = cand.to_crs(EQ)
    if not buildings.empty:
        near_b = gpd.sjoin_nearest(cand_eq, buildings.to_crs(EQ)[["geometry"]],
                                   how="left", distance_col="building_dist_m")
        near_b = near_b[~near_b.index.duplicated(keep="first")]
        cand["building_dist_m"] = near_b["building_dist_m"].round(1).values
    else:
        cand["building_dist_m"] = np.inf
    if not subs.empty:
        near_s = gpd.sjoin_nearest(cand_eq, subs.to_crs(EQ)[["geometry"]],
                                   how="left", distance_col="sub_dist_m")
        near_s = near_s[~near_s.index.duplicated(keep="first")]
        cand["sub_dist_m"] = near_s["sub_dist_m"].round(1).values
    else:
        cand["sub_dist_m"] = np.inf

    remote = cand[(cand.building_dist_m >= BUILDING_FAR_M)
                  & (cand.sub_dist_m >= SUB_CLEARANCE_M)].reset_index(drop=True)
    print(f"{len(remote)} candidates >= {BUILDING_FAR_M:.0f} m from any building and "
          f">= {SUB_CLEARANCE_M:.0f} m from any known substation")

    cell = ((remote.lon / DEDUP_DEG).round().astype(int).astype(str) + "_"
            + (remote.lat / DEDUP_DEG).round().astype(int).astype(str))
    remote = remote.loc[~cell.duplicated()].reset_index(drop=True)
    if len(remote) > n_target:
        remote = remote.sample(n=n_target, random_state=1).reset_index(drop=True)
    print(f"Mining {len(remote)} hard-negative chips after ~2 km dedup + cap")

    out_dir = ROOT / "data" / "chips" / f"{region}_hardneg_remote"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "s1").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    labels = subs.assign(area_m2=0.0, voltage_v=float("nan"), role="node")
    n_bands = len(MODEL_BANDS)

    records = []
    for _, row in tqdm(list(remote.iterrows()), total=len(remote), desc="remote_hard_negatives"):
        lon, lat = float(row.lon), float(row.lat)
        cid = _chip_id(lon, lat)
        img_path = out_dir / "images" / f"{cid}.tif"
        s1_path = out_dir / "s1" / f"{cid}.tif"
        mask_path = out_dir / "masks" / f"{cid}.tif"
        try:
            if not (img_path.exists() and mask_path.exists()):
                res = comp_idx.read_window(_chip_bbox(lon, lat))
                if res is None:
                    continue
                arr, transform, crs = res
                arr = arr[:n_bands]
                arr, ox, oy = _crop(arr, CHIP_SIZE)
                transform = transform * rasterio.Affine.translation(ox, oy)
                win_labels = labels[labels.geometry.intersects(_bbox_poly(lon, lat))]
                mask = _burn_mask(win_labels, transform, crs, arr.shape[-2:],
                                  settings.min_sub_area_m2, settings.node_ignore_radius_m)
                _write_tif(img_path, arr, transform, crs, "uint16")
                _write_tif(mask_path, mask.astype("int16"), transform, crs, "int16")
            if not s1_path.exists():
                s1res = comp_idx.read_window(_chip_bbox(lon, lat), "composite_s1.tif")
                if s1res is None:
                    continue
                s1arr, s1t, s1crs = s1res
                s1arr = s1arr[: len(S1_BANDS)]
                s1arr, ox, oy = _crop(s1arr, CHIP_SIZE)
                s1t = s1t * rasterio.Affine.translation(ox, oy)
                _write_tif(s1_path, s1arr, s1t, s1crs, "uint16")
        except Exception as e:  # noqa: BLE001 -- one bad chip must not kill the run
            print(f"chip {cid} failed: {e}")
            continue
        with rasterio.open(mask_path) as m:
            band = m.read(1)
        records.append(dict(chip_id=cid, lon=lon, lat=lat, kind="hard_negative_remote",
                             tile=None, split="train", sub_pixels=int((band == 1).sum()),
                             building_dist_m=row.building_dist_m, sub_dist_m=row.sub_dist_m,
                             image=str(img_path), s1=str(s1_path), mask=str(mask_path)))

    index = pd.DataFrame(records)
    index_path = out_dir / "index.parquet"
    index.to_parquet(index_path)
    n_actually_positive = int((index.sub_pixels > 0).sum()) if len(index) else 0
    print(f"Wrote {len(index)} remote hard-negative chips -> {index_path}")
    if n_actually_positive:
        print(f"  note: {n_actually_positive} accidentally overlap a real substation label "
              "(burned correctly as ignore, harmless but not a true hard negative)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=300)
    ap.add_argument("--region", default="yunnan")
    ap.add_argument("--iso3", default="CHN")
    a = ap.parse_args()
    main(a.n, a.region, a.iso3)
