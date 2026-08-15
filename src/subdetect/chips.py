"""Sample training chips: S2 (+S1) composites + rasterized substation masks.

Chip mix (all constrained to the composite coverage):
- positive        one per substation polygon >= min area, jittered so the target is
                  NOT centred (anti center-bias: else the model fires once per window
                  at inference -> a grid of false positives at the stride spacing).
- line_negative   points sampled along power lines, >= 1.5 km from any substation —
                  hard negatives matching the inference domain (the ROI is the corridor).
- background      uniform random.

Mask: 1 = substation polygon >= min area, 0 = background, -1 = ignore (sub-threshold
polygons, substation nodes as discs, and power=plant perimeters — unmapped switchyards).

With `fraction=True` the mask is instead float32 sub-pixel coverage in [0, 1] (-1 still
ignore), the target of the regression head — see `_burn_fraction`.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rio_features
from tqdm import tqdm

from subdetect.config import CHIP_SIZE, MODEL_BANDS, S1_BANDS, Settings, resolve_aoi
from subdetect.local_source import CompositeIndex, load_lines, load_substation_labels

log = logging.getLogger(__name__)

CHIP_M = CHIP_SIZE * 10.0  # chip edge in metres
EQ_AREA = "EPSG:6933"
LINE_NEG_SPACING_M = 3000.0   # sample a candidate negative every ~3 km along lines
LINE_NEG_CLEARANCE_M = 1500.0  # ...but only if this far from any substation


def _chip_id(lon: float, lat: float) -> str:
    return hashlib.sha1(f"{lon:.5f}_{lat:.5f}".encode()).hexdigest()[:12]


def _chip_bbox(lon: float, lat: float) -> tuple[float, float, float, float]:
    dy = CHIP_M / 2 / 111320.0
    dx = CHIP_M / 2 / (111320.0 * np.cos(np.radians(lat)))
    return (lon - dx, lat - dy, lon + dx, lat + dy)


def _bbox_poly(lon: float, lat: float):
    from shapely.geometry import box

    return box(*_chip_bbox(lon, lat))


def _jitter(lon: float, lat: float, rng: np.random.Generator, m: float = 900.0) -> tuple[float, float]:
    jit = m / 111320.0
    jx = rng.uniform(-jit, jit) / np.cos(np.radians(lat))
    jy = rng.uniform(-jit, jit)
    return lon + jx, lat + jy


def sample_chip_centers(
    labels: gpd.GeoDataFrame, lines: gpd.GeoDataFrame, coverage,
    rng: np.random.Generator, min_area: float, limit: int, voltage_weight: int = 1,
) -> pd.DataFrame:
    """`voltage_weight > 1` oversamples substations with a valid OSM voltage tag: each
    tagged polygon gets `voltage_weight` jittered positive chips instead of 1, giving them
    more representation in training without excluding untagged substations outright (which
    caused severe data-starvation regression when tried -- see conversation/memory)."""
    pos_polys = labels[(labels.role == "pos")]
    rows = []
    for _, r in pos_polys.iterrows():
        c = r.geometry.centroid
        lon, lat = _jitter(c.x, c.y, rng)
        tagged = bool(pd.notna(r.voltage_v) and r.voltage_v > 0)
        rows.append((lon, lat, "positive", tagged))
    pos = pd.DataFrame(rows, columns=["lon", "lat", "kind", "tagged"])
    if not pos.empty:
        cell = (pos.lon / 0.02).round().astype(int).astype(str) + "_" + (
            pos.lat / 0.02).round().astype(int).astype(str)
        pos = pos.loc[~cell.duplicated()].reset_index(drop=True)
    if voltage_weight > 1 and not pos.empty:
        extra = []
        for _, r in pos[pos.tagged].iterrows():
            for _ in range(voltage_weight - 1):
                lon, lat = _jitter(r.lon, r.lat, rng, m=300.0)
                extra.append((lon, lat, "positive", True))
        if extra:
            pos = pd.concat([pos, pd.DataFrame(extra, columns=pos.columns)], ignore_index=True)
    pos = pos.drop(columns="tagged")
    n_pos = max(len(pos), 1)

    # Line negatives: vertices of lines densified to LINE_NEG_SPACING_M, kept only where
    # far from any substation (poly or node), then jittered. Matches the corridor domain.
    line_rows = []
    if lines is not None and not lines.empty:
        subs_geom = labels[labels.role.isin(["pos", "small", "node"])]
        subs_buf = (
            subs_geom.to_crs(EQ_AREA).buffer(LINE_NEG_CLEARANCE_M).union_all()
            if not subs_geom.empty else None
        )
        lm = lines.to_crs(EQ_AREA)
        pts_m = []
        for geom in lm.geometry:
            if geom.is_empty:
                continue
            dense = geom.segmentize(LINE_NEG_SPACING_M)
            coords = (dense.coords if dense.geom_type == "LineString"
                      else [c for g in dense.geoms for c in g.coords])
            pts_m.extend(coords[::1])
        if pts_m:
            gp = gpd.GeoSeries(gpd.points_from_xy([p[0] for p in pts_m], [p[1] for p in pts_m]),
                               crs=EQ_AREA)
            if subs_buf is not None:
                gp = gp[~gp.within(subs_buf)]
            gp = gp.to_crs("EPSG:4326")
            for p in gp:
                lon, lat = _jitter(p.x, p.y, rng)
                line_rows.append((lon, lat, "line_negative"))
    near = pd.DataFrame(line_rows, columns=["lon", "lat", "kind"])
    if len(near) > int(0.7 * n_pos):
        near = near.sample(n=int(0.7 * n_pos), random_state=1).reset_index(drop=True)

    minx, miny, maxx, maxy = coverage.bounds
    m = int(0.3 * n_pos) or 1
    rand = pd.DataFrame({"lon": rng.uniform(minx, maxx, m), "lat": rng.uniform(miny, maxy, m),
                         "kind": "background"})

    out = pd.concat([pos, near, rand], ignore_index=True)
    inside = gpd.GeoSeries(gpd.points_from_xy(out.lon, out.lat), crs="EPSG:4326").within(coverage)
    out = out[inside.values].reset_index(drop=True)
    if limit and len(out) > limit:
        keep = pd.concat([
            out[out.kind == "positive"].head(max(limit // 2, 1)),
            out[out.kind == "line_negative"].head(max(limit // 4, 1)),
            out[out.kind == "background"].head(max(limit // 4, 1)),
        ])
        out = keep.head(limit).reset_index(drop=True)
    return out


def _burn_mask(labels: gpd.GeoDataFrame, transform, crs, shape, min_area: float,
               node_radius_m: float, all_touched: bool = True) -> np.ndarray:
    """Rasterize substation labels. pos polys -> 1; small/plant polys + node discs -> -1.

    `all_touched=True` (the historical default, and what every checkpoint through v14 was
    trained on) burns any pixel the polygon touches, which inflates a positive by roughly
    its perimeter in pixels: measured on 502 real installations, 1.70x in the 1,000-2,000 m2
    bucket against 1.13x at >=20,000 m2. `all_touched=False` burns pixels by centre
    containment instead, which measures 0.98-1.00x at every size with no installation losing
    its mask entirely -- see docs/model-lineage.md. Only the POSITIVE burn is switchable:
    ignore regions stay `all_touched=True`, where a halo is conservative rather than a bias.
    """
    mask = np.zeros(shape, dtype="int16")
    lab = labels.to_crs(crs)
    ignore = []
    big = []
    for g, a, role in zip(lab.geometry, labels.area_m2, labels.role):
        if g.is_empty:
            continue
        if role == "node":
            ignore.append((g.buffer(node_radius_m), 1))
        elif role in ("small", "plant") and g.geom_type in ("Polygon", "MultiPolygon"):
            ignore.append((g, 1))
        elif role == "pos" and a >= min_area and g.geom_type in ("Polygon", "MultiPolygon"):
            big.append((g, 1))
    if ignore:
        ign = rio_features.rasterize(ignore, out_shape=shape, transform=transform,
                                     fill=0, all_touched=True, dtype="uint8")
        mask[ign == 1] = -1
    if big:
        pos = rio_features.rasterize(big, out_shape=shape, transform=transform,
                                     fill=0, all_touched=all_touched, dtype="uint8")
        mask[pos == 1] = 1
    return mask


def _burn_fraction(labels: gpd.GeoDataFrame, transform, crs, shape, node_radius_m: float,
                   factor: int = 10) -> np.ndarray:
    """Sub-pixel substation coverage fraction per 10 m pixel, in [0, 1], with -1 = ignore.

    Rasterize every substation polygon (`pos` AND sub-floor `small` -- a real substation
    below the size floor is exactly what this target exists to represent) at `factor`x
    resolution (~1 m by default), then block-mean back down to the native grid.

    This is the whole point of the fraction target. `_burn_mask` rasterizes with
    `all_touched=True`, which inflates a polygon by roughly its perimeter in pixels: a
    1,500 m2 substation (~4 px across) is burned ~2x too large, a 200,000 m2 one only
    ~1.09x. The binary head is therefore trained on a footprint whose bias varies with
    size across the exact range where recall collapses. `all_touched=False` at the hi-res
    grid keeps the area estimate honest -- `all_touched=True` would reintroduce the same
    halo at 1 m.

    Ignore semantics carry over from `_burn_mask`: substation nodes (discs) and power=plant
    perimeters are unknown geometry, not background, so they are burned as -1 -- but only
    where no polygon coverage was measured, so a mapped switchyard inside a plant perimeter
    keeps its fraction.
    """
    lab = labels.to_crs(crs)
    polys = [
        (g, 1) for g, role in zip(lab.geometry, labels.role)
        if not g.is_empty and role in ("pos", "small")
        and g.geom_type in ("Polygon", "MultiPolygon")
    ]
    if polys:
        hi_shape = (shape[0] * factor, shape[1] * factor)
        hi_transform = transform * rasterio.Affine.scale(1.0 / factor)
        hi = rio_features.rasterize(polys, out_shape=hi_shape, transform=hi_transform,
                                    fill=0, all_touched=False, dtype="uint8")
        frac = hi.reshape(shape[0], factor, shape[1], factor).mean(axis=(1, 3)).astype("float32")
    else:
        frac = np.zeros(shape, dtype="float32")

    ignore = []
    for g, role in zip(lab.geometry, labels.role):
        if g.is_empty:
            continue
        if role == "node":
            ignore.append((g.buffer(node_radius_m), 1))
        elif role == "plant" and g.geom_type in ("Polygon", "MultiPolygon"):
            ignore.append((g, 1))
    if ignore:
        ign = rio_features.rasterize(ignore, out_shape=shape, transform=transform,
                                     fill=0, all_touched=True, dtype="uint8")
        frac[(ign == 1) & (frac <= 0)] = -1.0
    return frac


def _transform_bbox_poly(transform, crs, shape):
    """The chip's footprint in EPSG:4326, from its own geotransform."""
    from shapely.geometry import box

    minx, maxy = transform * (0, 0)
    maxx, miny = transform * (shape[1], shape[0])
    return gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=crs).to_crs("EPSG:4326").iloc[0]


def reburn_mask(src_mask_path: Path, out_path: Path | None, labels: gpd.GeoDataFrame,
                node_radius_m: float, fraction: bool = True, min_area: float = 0.0,
                factor: int = 10, all_touched: bool = True) -> np.ndarray:
    """Re-burn one already-built chip's mask from `labels`, reusing the chip's own
    geotransform: as a float32 coverage fraction (`fraction=True`) or as a binary class
    mask with a chosen `all_touched`.

    This is how an existing chip corpus is re-targeted without re-sampling chip centres or
    re-reading composites, so a run trains on byte-identical imagery with an identical
    train/val split and the label representation is the only variable against a baseline
    trained on the same index -- which is the point of both arms in
    `scripts/build_reburn_chip_index.py`. Passing `fraction=False, all_touched=True`
    reproduces the mask the chip was originally built with, which is how that script's
    `--verify` proves the label settings match before re-burning anything.

    `labels` must carry a spatial index (geopandas builds one lazily on first use).
    Pass `out_path=None` to compute without writing.
    """
    with rasterio.open(src_mask_path) as src:
        transform, crs, shape = src.transform, src.crs, (src.height, src.width)
    bbox = _transform_bbox_poly(transform, crs, shape)
    win = labels.iloc[labels.sindex.query(bbox, predicate="intersects")]
    if fraction:
        mask = _burn_fraction(win, transform, crs, shape, node_radius_m, factor=factor)
        dtype = "float32"
    else:
        mask = _burn_mask(win, transform, crs, shape, min_area, node_radius_m,
                          all_touched=all_touched).astype("int16")
        dtype = "int16"
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_tif(out_path, mask, transform, crs, dtype)
    return mask


def _write_tif(path: Path, arr: np.ndarray, transform, crs, dtype: str) -> None:
    arr = arr if arr.ndim == 3 else arr[None]
    # predictor 3 is the floating-point predictor; 2 (horizontal) is for integers and
    # silently costs compression ratio on float32 fraction masks.
    predictor = 3 if dtype.startswith("float") else 2
    with rasterio.open(
        path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
        dtype=dtype, crs=crs, transform=transform, compress="deflate", predictor=predictor,
    ) as dst:
        dst.write(arr)


def _crop(arr: np.ndarray, size: int) -> tuple[np.ndarray, int, int]:
    y, x = arr.shape[-2], arr.shape[-1]
    oy, ox = max((y - size) // 2, 0), max((x - size) // 2, 0)
    arr = arr[..., oy : oy + size, ox : ox + size]
    if arr.shape[-2] < size or arr.shape[-1] < size:
        pad = [(0, 0)] * (arr.ndim - 2) + [(0, size - arr.shape[-2]), (0, size - arr.shape[-1])]
        arr = np.pad(arr, pad)
    return arr, ox, oy


def _tile_of(lon: float, lat: float, comp_idx: CompositeIndex) -> str | None:
    from shapely.geometry import Point

    hit = comp_idx.index[comp_idx.index.contains(Point(lon, lat))]
    if hit.empty:
        return None
    return Path(hit.iloc[0].path).parent.name


def _split_of(lon: float, lat: float, cfg: dict) -> str:
    vb = cfg.get("val_bbox")
    if vb and (vb[0] <= lon <= vb[2]) and (vb[1] <= lat <= vb[3]):
        return "val"
    return "train"


def build_chips(
    aoi: str, labels_dir: Path, out_dir: Path, limit: int = 0, with_s1: bool = False,
    with_tower: bool = False,
    min_area_m2: float | None = None, prefer_refined: bool = True, voltage_only: bool = False,
    voltage_weight: int = 1, fraction: bool = False,
) -> Path:
    """`fraction=True` burns continuous sub-pixel coverage-fraction masks (float32, the
    regression target -- see `_burn_fraction`) instead of binary class masks, into a
    parallel `<aoi>_fraction/` tree so the segmentation chips are untouched. To convert an
    *existing* corpus instead (the matched-baseline path), see
    `scripts/build_reburn_chip_index.py`.

    `min_area_m2` overrides the settings label floor for TRAINING masks only
    (0 = every substation polygon becomes class 1, none ignored for size). The
    settings floor stays authoritative for candidates/eval elsewhere.

    `prefer_refined=False` forces raw (unrefined) OSM labels even if a refined parquet
    exists, for a clean baseline comparison. `voltage_only=True` restricts positive
    supervision to substations with a valid OSM `voltage` tag (see
    `local_source.load_substation_labels`). `voltage_weight` oversamples voltage-tagged
    substations at the chip level instead (see `sample_chip_centers`) -- the less
    destructive alternative to `voltage_only`, which starved the training set."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings.load()
    if min_area_m2 is not None:
        settings.min_sub_area_m2 = float(min_area_m2)
    _, cfg = resolve_aoi(aoi, settings)
    out_dir = Path(out_dir) / (f"{aoi}_fraction" if fraction else aoi)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    if with_s1:
        (out_dir / "s1").mkdir(parents=True, exist_ok=True)
    if with_tower:
        (out_dir / "tower").mkdir(parents=True, exist_ok=True)

    composed = Path("data/composites") / aoi
    if not (composed.exists() and any(composed.glob("composites/*/composite_0.tif"))):
        raise FileNotFoundError(
            f"No composites under {composed}; run `subdetect compose --aoi {aoi}` first"
        )
    comp_idx = CompositeIndex(composed)
    coverage = comp_idx.coverage

    labels_dir = Path(labels_dir) / aoi
    labels = load_substation_labels(labels_dir, min_area_m2=settings.min_sub_area_m2,
                                     prefer_refined=prefer_refined, voltage_only=voltage_only)
    labels = labels[labels.geometry.centroid.within(coverage)].reset_index(drop=True)
    lines = load_lines(labels_dir)
    lines = lines[lines.geometry.intersects(coverage)].reset_index(drop=True)
    log.info("AOI %s: %d composite tiles, labels %s, %d lines in coverage", aoi,
             len(comp_idx.index), labels.role.value_counts().to_dict(), len(lines))

    rng = np.random.default_rng(42)
    centers = sample_chip_centers(labels, lines, coverage, rng, settings.min_sub_area_m2, limit,
                                   voltage_weight=voltage_weight)
    log.info("Sampling %d chips (%s)%s", len(centers), centers.kind.value_counts().to_dict(),
             " with S1" if with_s1 else "")
    n_bands = len(MODEL_BANDS)

    def _build_one(row) -> dict | None:
        cid = _chip_id(row.lon, row.lat)
        img_path = out_dir / "images" / f"{cid}.tif"
        mask_path = out_dir / "masks" / f"{cid}.tif"
        s1_path = out_dir / "s1" / f"{cid}.tif" if with_s1 else None
        tower_path = out_dir / "tower" / f"{cid}.tif" if with_tower else None
        tile = _tile_of(row.lon, row.lat, comp_idx)
        try:
            if not (img_path.exists() and mask_path.exists()):
                res = comp_idx.read_window(_chip_bbox(row.lon, row.lat))
                if res is None:
                    return None
                arr, transform, crs = res
                arr = arr[:n_bands]
                arr, ox, oy = _crop(arr, CHIP_SIZE)
                transform = transform * rasterio.Affine.translation(ox, oy)
                win_labels = labels[labels.geometry.intersects(_bbox_poly(row.lon, row.lat))]
                _write_tif(img_path, arr, transform, crs, "uint16")
                if fraction:
                    mask = _burn_fraction(win_labels, transform, crs, arr.shape[-2:],
                                          settings.node_ignore_radius_m)
                    _write_tif(mask_path, mask, transform, crs, "float32")
                else:
                    mask = _burn_mask(win_labels, transform, crs, arr.shape[-2:],
                                      settings.min_sub_area_m2, settings.node_ignore_radius_m)
                    _write_tif(mask_path, mask.astype("int16"), transform, crs, "int16")
            if with_s1 and s1_path is not None and not s1_path.exists():
                try:
                    s1res = comp_idx.read_window(_chip_bbox(row.lon, row.lat), "composite_s1.tif")
                except FileNotFoundError:
                    return None  # S1 not composited for this cell yet
                if s1res is None:
                    return None
                s1arr, s1t, s1crs = s1res
                s1arr = s1arr[: len(S1_BANDS)]
                s1arr, ox, oy = _crop(s1arr, CHIP_SIZE)
                s1t = s1t * rasterio.Affine.translation(ox, oy)
                _write_tif(s1_path, s1arr, s1t, s1crs, "uint16")
            if with_tower and tower_path is not None and not tower_path.exists():
                try:
                    tres = comp_idx.read_window(_chip_bbox(row.lon, row.lat), "composite_tower.tif")
                except FileNotFoundError:
                    return None  # tower prior not composited for this cell yet
                if tres is None:
                    return None
                tarr, tt, tcrs = tres
                tarr = tarr[:1]
                tarr, ox, oy = _crop(tarr, CHIP_SIZE)
                tt = tt * rasterio.Affine.translation(ox, oy)
                _write_tif(tower_path, tarr, tt, tcrs, "uint16")
        except Exception as e:  # noqa: BLE001 — one bad chip must not kill the run
            log.warning("chip %s failed: %s", cid, e)
            return None
        with rasterio.open(mask_path) as m:
            band = m.read(1)
        rec = dict(chip_id=cid, lon=row.lon, lat=row.lat, kind=row.kind, tile=tile,
                   split=_split_of(row.lon, row.lat, cfg),
                   sub_pixels=int((band > 0).sum() if fraction else (band == 1).sum()),
                   image=str(img_path), s1=str(s1_path) if s1_path else None,
                   tower=str(tower_path) if tower_path else None,
                   mask=str(mask_path))
        if fraction:
            # Σ coverage over the chip: chip-level substation area in units of 100 m2.
            # `sub_pixels` counts any partially covered pixel, so the two together separate
            # "one big yard" from "a scatter of boundary pixels".
            rec["sub_frac_sum"] = float(band[band > 0].sum())
        return rec

    records = [r for r in (_build_one(row) for _, row in tqdm(
        list(centers.iterrows()), desc="chips")) if r]
    index = pd.DataFrame(records)
    index_path = out_dir / "index.parquet"
    index.to_parquet(index_path)
    log.info("Wrote %d chips (%d with substation, %d val) -> %s", len(index),
             int((index.sub_pixels > 0).sum()) if len(index) else 0,
             int((index.split == "val").sum()) if len(index) else 0, index_path)
    return index_path
