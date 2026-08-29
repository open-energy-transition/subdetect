"""OSM -> small run-of-river hydropower SITE labels for northern Pakistan.

Written 2026-08-29 to answer a feasibility question: can this pipeline, which detects
transmission substations, be pointed at the small hydro plants mapped across
Gilgit-Baltistan / KPK / Chitral / Swat / AJK?

The blocking fact is size. The median mapped hydro plant polygon in northern Pakistan is
~535 m2 -- about five Sentinel-2 pixels -- while `data/eval_results/field_eval.csv` reports
recall 0.000 in the 1,000-2,000 m2 bucket for all 23 substation runs, with p_max *exactly*
0.000 inside those polygons. The powerhouse alone is therefore below a size class where
this model demonstrably produces no response at all, and the penstock is sub-pixel in width
and usually tagged `tunnel=yes`.

So the positive built here is not the powerhouse. It is the **site complex**: the dissolved
75 m buffer of plant + generator + penstock/pressurised conduit belonging to one physical
installation. That is ~30,000 m2 -- the >=20,000 m2 bucket, where measured recall is 0.714 --
and it is what a ranked-lead product actually needs, since a lead needs a location, not a
footprint. `core_area_m2` preserves the honest unbuffered number alongside it.

Deliberately a standalone script rather than a change to `src/subdetect/osm.py`: that
module's four parquets feed the production substation lineage, and its `_EXPORT_CONFIG` is
a module-level dict whose mutation is the change most likely to silently alter the
substation label set. `scripts/build_settlement_hardneg_chips.py` already establishes the
"second osmium extraction alongside the main one" idiom against the same cached PBF.

Read-only with respect to everything that exists: writes only under
`data/labels/pakistan_hydro/` and `data/eval_results/`, and asserts the production labels
are untouched.

    pixi run python scripts/build_hydro_labels.py --refresh-pbf
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely.ops import nearest_points, substring, unary_union
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subdetect.config import geodesic_area_m2  # noqa: E402

log = logging.getLogger("build_hydro_labels")

OSM_DIR = ROOT / "data" / "osm"
OUT_DIR = ROOT / "data" / "labels" / "pakistan_hydro"
RESULTS_DIR = ROOT / "data" / "eval_results"
PROD_LABELS = ROOT / "data" / "labels" / "pakistan"
COMPOSITES = ROOT / "data" / "composites" / "pakistan" / "composites"
PROB_V9 = ROOT / "data" / "predictions_v9_fused" / "pakistan" / "prob_mean"

GEOFABRIK_PK = "https://download.geofabrik.de/asia/pakistan-latest.osm.pbf"

# Northern Pakistan study window. East edge runs to 77.5 so the Ladakh overlap in the
# Geofabrik PK extract stays *visible* (flagged as east_of_76) rather than silently
# dropped -- admin1_PAK.geojson cannot separate it, its Gilgit-Baltistan polygon extends
# to lon 77.815 (the Pakistani claim).
NORTH_BBOX = (70.5, 33.0, 77.5, 37.5)

# The pakistan AOI's snapped 0.1 deg lattice origin (roi._snapped_origin of its bbox
# against grid_origin). Cell names must match data/composites/pakistan/composites/.
CELL_DEG = 0.1
GRID_MINX, GRID_MINY = 60.8405158324398, 23.7089663393252

# Filenames owned by the substation lineage. Never write any of these here: chips.py
# densifies `lines.parquet` at 3 km spacing into `line_negative` training chips, so a
# penstock landing in that file would be trained as *background*.
FORBIDDEN_NAMES = {
    "lines.parquet",
    "plants.parquet",
    "substations_poly.parquet",
    "substations_node.parquet",
    "substations_poly_refined.parquet",
}

# osmium tags-filter has no AND semantics, so filter broadly and AND in pandas below.
# waterway=stream is deliberately absent: tens of thousands of features, no gain here.
FILTERS = [
    "nwr/power=plant",
    "nwr/power=generator",
    "w/waterway=penstock,pressurised,canal,ditch,drain",
    "w/usage=penstock,headrace,tailrace",
    "w/man_made=pipeline",
]

# The tags that must survive `osmium export`. The production config keeps only
# ["power","voltage","name","substation","operator"], which is exactly why the cached
# plants.parquet cannot tell hydro from solar. Exact keys, no wildcards.
EXPORT_CONFIG = {
    "attributes": {"type": "@type", "id": "@id"},
    "linear_tags": True,
    "area_tags": True,
    "include_tags": [
        "power",
        "plant:source", "plant:method", "plant:type", "plant:output:electricity",
        "generator:source", "generator:method", "generator:type",
        "generator:output:electricity",
        "waterway", "usage", "man_made", "substance", "pipeline",
        "location", "tunnel", "layer", "intermittent",
        "name", "name:en", "operator", "voltage", "start_date", "ref", "wikidata",
    ],
}

# Columns that must exist after the export round-trip. Their absence means a stale
# intermediate was reused despite a changed filter -- the failure mode the config hash in
# the filenames is there to prevent, asserted here as a backstop.
REQUIRED_COLS = {
    "power", "plant:source", "generator:source", "waterway",
    "usage", "man_made", "substance", "plant:output:electricity",
}

HYDRO_SRC = re.compile(r"hydro|run[-_ ]?of[-_ ]?(?:the[-_ ])?river", re.I)
WATER_METHOD = re.compile(r"water", re.I)
CONDUIT_USAGE = re.compile(r"penstock|headrace|tailrace", re.I)
WATER_SUBSTANCE = re.compile(r"water|hydro", re.I)

# Approximate valley/cluster boxes, used for the geographic-spread gate and as the
# grouping unit for Stage 4's cross-validation folds.
CLUSTERS = {
    "tarbela_haripur": (72.55, 33.80, 73.25, 34.25),
    "mangla_ajk": (73.30, 32.90, 74.30, 33.60),
    "chitral_golen": (71.40, 35.40, 72.30, 36.30),
    "kaghan": (73.20, 34.40, 74.00, 35.10),
    "warsak_peshawar": (71.10, 33.90, 71.80, 34.40),
    "hunza": (74.20, 36.00, 75.10, 36.70),
    "dir_panjkora": (71.60, 34.60, 72.20, 35.60),
    "swat_daral": (72.20, 34.60, 72.95, 35.60),
    "astore": (74.50, 34.90, 75.30, 35.60),
    "neelum": (73.60, 34.20, 74.60, 34.90),
    "gilgit": (73.90, 35.60, 74.60, 36.10),
    "skardu": (75.20, 35.00, 76.00, 35.60),
}

# Geographic holdout: the Swat + Panjkora system. One hydrological basin, 130-180 km from
# both large clusters, and -- decisively -- already covered by composites, unlike Kaghan /
# Hunza / Astore which have zero composited cells and so cannot be scored at all.
HOLDOUT_BBOX = (71.55, 34.55, 72.95, 35.65)

CONFIG_HASH = hashlib.sha1(
    json.dumps({"filters": FILTERS, "export": EXPORT_CONFIG, "v": 1}, sort_keys=True).encode()
).hexdigest()[:8]


# --------------------------------------------------------------------------- PBF


def _run(cmd: list[str]) -> None:
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def pbf_timestamp(pbf: Path) -> dt.datetime | None:
    """Geofabrik snapshot instant from the PBF header, or None if unreadable."""
    try:
        out = subprocess.run(
            ["osmium", "fileinfo", "-g", "header.option.timestamp", str(pbf)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return dt.datetime.fromisoformat(out.replace("Z", "+00:00"))
    except Exception as e:  # noqa: BLE001
        log.warning("could not read timestamp from %s: %s", pbf, e)
        return None


def fetch_pbf(refresh: bool, max_age_days: int, allow_stale: bool) -> tuple[Path, dt.datetime | None, int]:
    """Return (pbf, snapshot_ts, age_days), downloading to a DATED filename if asked.

    Never touches `pakistan-latest.osm.pbf`: that file is the substation lineage's input,
    and `osm._download` returns early whenever it exists, so it can never self-refresh.
    """
    dated = OSM_DIR / f"pakistan-{dt.date.today():%Y%m%d}.osm.pbf"
    if refresh and not dated.exists():
        _run(["wget", "-c", "-q", "--show-progress", GEOFABRIK_PK, "-O", str(dated)])
    pbf = dated if dated.exists() else OSM_DIR / "pakistan-latest.osm.pbf"
    if not pbf.exists():
        raise SystemExit(f"no PBF at {pbf}; pass --refresh-pbf")
    ts = pbf_timestamp(pbf)
    age = (dt.datetime.now(dt.timezone.utc) - ts).days if ts else 10_000
    log.info("PBF %s  snapshot=%s  age=%d d", pbf.name, ts, age)
    if age > max_age_days and not allow_stale:
        raise SystemExit(
            f"{pbf.name} snapshot {ts} is {age} d old (limit {max_age_days}). "
            f"Pass --refresh-pbf to download today's, or --allow-stale to proceed."
        )
    return pbf, ts, age


def extract(pbf: Path, bbox: tuple[float, float, float, float], force: bool) -> gpd.GeoDataFrame:
    """osmium tags-filter -> extract -> export -> GeoDataFrame, with all tags intact."""
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{pbf.stem}_hydro_{CONFIG_HASH}"
    filt = OSM_DIR / f"{stem}.osm.pbf"
    clip = OSM_DIR / f"{stem}_clip.osm.pbf"
    seq = OSM_DIR / f"{stem}.geojsonseq"
    if force:
        for p in (filt, clip, seq):
            p.unlink(missing_ok=True)

    # The config hash is in the filename precisely so that changing FILTERS or
    # EXPORT_CONFIG forces a cache miss. osm.py's `if not power.exists()` guard silently
    # reuses a stale intermediate when its tag list changes; this cannot.
    if not filt.exists():
        _run(["osmium", "tags-filter", "-o", str(filt), "--overwrite", str(pbf), *FILTERS])
    if not clip.exists():
        _run(["osmium", "extract", "-s", "smart", "--overwrite",
              "-b", ",".join(str(x) for x in bbox), "-o", str(clip), str(filt)])
    if not seq.exists():
        cfg = OSM_DIR / "hydro_export_config.json"
        cfg.write_text(json.dumps(EXPORT_CONFIG))
        _run(["osmium", "export", str(clip), "-f", "geojsonseq", "--overwrite",
              "-c", str(cfg), "--add-unique-id=type_id", "-o", str(seq)])

    # Parsed by hand rather than via gpd.read_file: GDAL's GeoJSONSeq driver infers a
    # schema from a scan, and a tag carried by only a handful of late features --
    # plant:output:electricity is exactly that -- can silently vanish as a column.
    rows = []
    with seq.open() as f:
        for line in f:
            line = line.strip().lstrip("\x1e")
            if not line:
                continue
            try:
                feat = json.loads(line)
            except json.JSONDecodeError:
                continue
            props = dict(feat.get("properties", {}))
            props["osm_uid"] = feat.get("id")
            props["osm_type"] = props.pop("@type", None)
            props["osm_id"] = props.pop("@id", None)
            try:
                props["geometry"] = shape(feat["geometry"])
            except Exception:  # noqa: BLE001 — skip broken geometries
                continue
            rows.append(props)
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    log.info("exported %d features from %s", len(gdf), seq.name)

    missing = REQUIRED_COLS - set(gdf.columns)
    if missing:
        raise SystemExit(
            f"osmium export dropped {sorted(missing)} -- stale intermediate? "
            f"delete {filt.name}/{seq.name} or pass --force"
        )
    return gdf


# ------------------------------------------------------------------- classification


def _col(gdf: gpd.GeoDataFrame, key: str) -> pd.Series:
    if key not in gdf.columns:
        return pd.Series("", index=gdf.index, dtype=object)
    return gdf[key].fillna("").astype(str)


def classify(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Tag-based class assignment. Never geometry-type based.

    `osmium tags-filter` drags in every node referenced by a matched way, so the export
    contains untagged members that match nothing -- post-filtering on tags is mandatory,
    not optional. Matching is by substring so `plant:source=hydro;solar` is caught.
    """
    power = _col(gdf, "power")
    is_plant = (power == "plant") & (
        _col(gdf, "plant:source").str.contains(HYDRO_SRC)
        | _col(gdf, "plant:method").str.contains(WATER_METHOD)
    )
    is_gen = (power == "generator") & (
        _col(gdf, "generator:source").str.contains(HYDRO_SRC)
        | _col(gdf, "generator:method").str.contains(WATER_METHOD)
    )
    waterway = _col(gdf, "waterway")
    is_conduit = (
        waterway.isin(["penstock", "pressurised"])
        | _col(gdf, "usage").str.contains(CONDUIT_USAGE)
        | ((_col(gdf, "man_made") == "pipeline") & _col(gdf, "substance").str.contains(WATER_SUBSTANCE))
    )
    is_headrace = waterway.isin(["canal", "ditch", "drain"]) & ~is_conduit

    cls = pd.Series("", index=gdf.index, dtype=object)
    cls[is_headrace] = "headrace_cand"
    cls[is_conduit] = "conduit"
    cls[is_gen] = "generator"
    cls[is_plant] = "plant"
    gdf = gdf.assign(hydro_class=cls)

    kept = gdf[gdf.hydro_class != ""].copy()
    log.info("classified: %s (dropped %d unclassified)",
             kept.hydro_class.value_counts().to_dict(), len(gdf) - len(kept))
    log.info("geometry types by class:\n%s",
             pd.crosstab(kept.hydro_class, kept.geometry.geom_type).to_string())
    return kept


_UNIT = {"": 1e-6, "k": 1e-3, "m": 1.0, "g": 1e3}
_POWER_RE = re.compile(r"([\d.]+)\s*([kMGmg]?)\s*W", re.I)


def parse_power_mw(v) -> float:
    """'4.8 MW' -> 4.8 | '84000 kW' -> 84.0 | '1.5 MW;2 MW' -> 3.5 | 'yes' -> nan.

    Values within one tag are summed: OSM lists per-unit outputs separated by ';', so a
    plant tagged '2 MW;2 MW' is a 4 MW plant. Lowercase 'm' is read as mega, not milli --
    SI says milliwatts, but no mapper has ever meant that.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return float("nan")
    total = 0.0
    for num, unit in _POWER_RE.findall(str(v)):
        try:
            total += float(num) * _UNIT[unit.lower()]
        except (ValueError, KeyError):
            continue
    return total if total > 0 else float("nan")


# ---------------------------------------------------------------------- assembly


def _utm_for(lon: float, lat: float) -> str:
    return f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) / 6) + 1}"


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def assemble_sites(
    feats: gpd.GeoDataFrame,
    *,
    buffer_m: float = 75.0,
    link_m: float = 150.0,
    attach_m: float = 150.0,
    headrace_attach_m: float = 250.0,
    max_conduit_len_m: float = 2000.0,
    with_headrace: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """One dissolved footprint per physical installation. Returns (sites, components)."""
    anchors = feats[feats.hydro_class.isin(["plant", "generator"])].copy()
    conduits = feats[feats.hydro_class == "conduit"].copy()
    headraces = feats[feats.hydro_class == "headrace_cand"].copy() if with_headrace \
        else feats.iloc[0:0].copy()
    if anchors.empty:
        raise SystemExit("no hydro plant/generator anchors found -- check the tag filters")

    # Metric work in the local UTM, not EPSG:6933: that projection is equal-*area*, and at
    # lat 35 its E-W/N-S scale factors are 1.057/0.946 -- a 12% anisotropy, harmless for a
    # site buffer and not harmless for a distance threshold.
    c = anchors.geometry.union_all().centroid
    utm = _utm_for(c.x, c.y)
    anchors = anchors.to_crs(utm)
    anchors["geometry"] = anchors.geometry.buffer(0)
    conduits = conduits.to_crs(utm) if len(conduits) else conduits
    headraces = headraces.to_crs(utm) if len(headraces) else headraces

    # Union-find over an STRtree self-query rather than a geometry dissolve: the dissolve
    # would throw away the object -> site mapping that n_components and src_ids need.
    # Half of link_m each, so two anchors merge when they are within link_m of each other.
    bufs = list(anchors.geometry.buffer(link_m / 2.0))
    uf = _UnionFind(len(anchors))
    if bufs:
        tree = STRtree(bufs)
        qi, qj = tree.query(bufs, predicate="intersects")
        for a, b in zip(qi, qj):
            uf.union(int(a), int(b))
    roots = [uf.find(i) for i in range(len(anchors))]
    anchors = anchors.assign(_root=roots)

    members: dict[int, list[dict]] = {r: [] for r in set(roots)}
    for (_, row), root in zip(anchors.iterrows(), roots):
        members[root].append({"row": row, "cls": row.hydro_class, "geom": row.geometry,
                              "len_full": 0.0, "clipped": False})

    anchor_geoms = list(anchors.geometry)
    anchor_tree = STRtree(anchor_geoms) if anchor_geoms else None
    stats = {"unattached": 0, "ambiguous": 0, "clipped": 0}

    def _attach(lines: gpd.GeoDataFrame, cls: str, within_m: float) -> None:
        if anchor_tree is None or lines.empty:
            return
        for _, row in lines.iterrows():
            line = row.geometry
            if line is None or line.is_empty:
                continue
            hits = anchor_tree.query(line.buffer(within_m), predicate="intersects")
            if len(hits) == 0:
                stats["unattached"] += 1
                continue
            if len({roots[int(h)] for h in hits}) > 1:
                stats["ambiguous"] += 1
            # Nearest anchor ONLY. Attaching a conduit to every anchor it reaches would
            # chain cascade stages into one blob -- the longest matched pipe here is
            # 29 km, easily long enough to swallow a whole valley.
            nearest = min(hits, key=lambda i: line.distance(anchor_geoms[int(i)]))
            root = roots[int(nearest)]
            full_len = float(line.length)
            clipped = False
            if full_len > 2 * max_conduit_len_m:
                # Clip by arc length around the anchor. p90 of matched pipe length is
                # 702 m, so 2 km keeps every real penstock and trims only the long
                # headrace tunnels, whose 75 m buffer would otherwise dominate the area.
                try:
                    apt, _ = nearest_points(anchor_geoms[int(nearest)], line)
                    d = line.project(apt)
                    seg = substring(line, max(0.0, d - max_conduit_len_m),
                                    min(line.length, d + max_conduit_len_m))
                    if seg is not None and not seg.is_empty:
                        line, clipped = seg, True
                        stats["clipped"] += 1
                except Exception:  # noqa: BLE001 — keep the unclipped line on failure
                    pass
            members[root].append({"row": row, "cls": cls, "geom": line,
                                  "len_full": full_len, "clipped": clipped})

    _attach(conduits, "conduit", attach_m)
    if with_headrace:
        have_conduit = {r for r, ms in members.items() if any(m["cls"] == "conduit" for m in ms)}
        cand = headraces[~headraces.index.isin([])]
        _attach(cand[~cand.index.isin([])], "headrace_cand", headrace_attach_m)
        # headraces only supplement sites with no real conduit
        for r in have_conduit:
            members[r] = [m for m in members[r] if m["cls"] != "headrace_cand"]

    log.info("conduit attachment: %d unattached, %d ambiguous (nearest wins), %d clipped",
             stats["unattached"], stats["ambiguous"], stats["clipped"])

    site_rows, comp_rows = [], []
    for root, ms in members.items():
        geom = unary_union([m["geom"].buffer(buffer_m) for m in ms]).buffer(0)
        site = gpd.GeoSeries([geom], crs=utm).to_crs("EPSG:4326").iloc[0]
        core_geoms = [m["geom"] for m in ms if m["cls"] == "plant"]
        core_4326 = (gpd.GeoSeries(core_geoms, crs=utm).to_crs("EPSG:4326")
                     if core_geoms else gpd.GeoSeries([], crs="EPSG:4326"))
        core_area = float(sum(geodesic_area_m2(g) for g in core_4326))
        anchor_geoms_4326 = gpd.GeoSeries(
            [m["geom"] for m in ms if m["cls"] in ("plant", "generator")], crs=utm
        ).to_crs("EPSG:4326")
        rep = unary_union(list(anchor_geoms_4326)).representative_point()

        plant_mw = max([parse_power_mw(m["row"].get("plant:output:electricity"))
                        for m in ms if m["cls"] == "plant"] or [float("nan")])
        gen_mw = np.nansum([parse_power_mw(m["row"].get("generator:output:electricity"))
                            for m in ms if m["cls"] == "generator"] or [float("nan")])
        # max, not sum: the plant tag is the whole-plant figure while generator tags are
        # per-unit, so adding them double-counts a plant that tags both.
        cands = [x for x in (plant_mw, float(gen_mw)) if x and not np.isnan(x)]
        capacity = max(cands) if cands else float("nan")

        names = [str(m["row"].get("name") or m["row"].get("name:en") or "")
                 for m in ms if m["cls"] == "plant"] + \
                [str(m["row"].get("name") or "") for m in ms if m["cls"] == "generator"]
        name = next((n for n in names if n), "")

        conduit_ms = [m for m in ms if m["cls"] in ("conduit", "headrace_cand")]
        tunnel_len = sum(float(m["geom"].length) for m in conduit_ms
                         if str(m["row"].get("tunnel") or "") not in ("", "no")
                         or str(m["row"].get("location") or "") == "underground")
        conduit_len = sum(float(m["geom"].length) for m in conduit_ms)

        site_rows.append({
            "geometry": site,
            "site_area_m2": geodesic_area_m2(site),
            "core_area_m2": core_area,
            "capacity_mw": capacity,
            "name": name,
            "has_penstock": any(m["cls"] == "conduit" for m in ms),
            "has_headrace": any(m["cls"] == "headrace_cand" for m in ms),
            "conduit_len_m": conduit_len,
            "conduit_len_m_full": sum(m["len_full"] for m in conduit_ms),
            "conduit_tunnel_frac": (tunnel_len / conduit_len) if conduit_len > 0 else 0.0,
            "n_components": len(ms),
            "n_plant": sum(m["cls"] == "plant" for m in ms),
            "n_generator": sum(m["cls"] == "generator" for m in ms),
            "n_conduit": sum(m["cls"] == "conduit" for m in ms),
            "src_ids": ";".join(sorted(str(m["row"].get("osm_uid")) for m in ms)),
            "lon": rep.x, "lat": rep.y,
            "_root": root,
        })
        for m in ms:
            g4326 = gpd.GeoSeries([m["geom"]], crs=utm).to_crs("EPSG:4326").iloc[0]
            comp_rows.append({
                "_root": root, "cls": m["cls"],
                "osm_uid": m["row"].get("osm_uid"),
                "osm_type": m["row"].get("osm_type"),
                "osm_id": m["row"].get("osm_id"),
                "area_m2": geodesic_area_m2(g4326),
                "length_m": float(m["geom"].length),
                "length_m_full": m["len_full"],
                "clipped": m["clipped"],
                "geometry": g4326,
            })

    sites = gpd.GeoDataFrame(site_rows, crs="EPSG:4326")
    sites = sites.sort_values(["lat", "lon"], ascending=[False, True]).reset_index(drop=True)
    sites["site_id"] = [f"PKH-{i:04d}" for i in range(len(sites))]
    root_to_id = dict(zip(sites._root, sites.site_id))
    comps = gpd.GeoDataFrame(comp_rows, crs="EPSG:4326")
    comps["site_id"] = comps._root.map(root_to_id)
    return sites.drop(columns=["_root"]), comps.drop(columns=["_root"])


# ------------------------------------------------------------------- enrichment


def _bucket(a: float) -> str:
    if a <= 0:
        return "node_only"
    for lo, hi, name in [(0, 1000, "<1k"), (1000, 2000, "1k-2k"), (2000, 5000, "2k-5k"),
                         (5000, 20000, "5k-20k")]:
        if lo < a < hi:
            return name
    return ">=20k"


def enrich(sites: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add size buckets, cluster, composite cell, coverage flags and the split column."""
    sites = sites.copy()
    sites["size_bucket"] = sites.core_area_m2.map(_bucket)
    sites["buffered_bucket"] = sites.site_area_m2.map(_bucket)

    def _cluster(lon, lat):
        for name, (x0, y0, x1, y1) in CLUSTERS.items():
            if x0 <= lon <= x1 and y0 <= lat <= y1:
                return name
        return ""

    sites["cluster"] = [_cluster(r.lon, r.lat) for r in sites.itertuples()]
    sites["cell_0p5"] = [f"{np.floor(r.lon / 0.5) * 0.5:.1f}_{np.floor(r.lat / 0.5) * 0.5:.1f}"
                         for r in sites.itertuples()]
    ix = np.floor((sites.lon.values - GRID_MINX) / CELL_DEG).astype(int)
    iy = np.floor((sites.lat.values - GRID_MINY) / CELL_DEG).astype(int)
    sites["cell"] = [f"{a:04d}_{b:04d}" for a, b in zip(ix, iy)]
    sites["has_composite"] = [(COMPOSITES / c / "composite_0.tif").exists() for c in sites.cell]
    sites["has_s1"] = [(COMPOSITES / c / "composite_s1.tif").exists() for c in sites.cell]
    sites["has_tower_prior"] = [(COMPOSITES / c / "composite_tower.tif").exists() for c in sites.cell]
    sites["has_prob_v9"] = [(PROB_V9 / f"{c}.tif").exists() for c in sites.cell]
    sites["east_of_76"] = sites.lon > 76.0

    x0, y0, x1, y1 = HOLDOUT_BBOX
    sites["split"] = np.where(
        (sites.lon >= x0) & (sites.lon <= x1) & (sites.lat >= y0) & (sites.lat <= y1),
        "holdout", "study")
    return sites


# ------------------------------------------------------------------------ gates


def evaluate_gates(sites: gpd.GeoDataFrame, age_days: int, n_new: int | None,
                   stability: dict[float, int], max_age_days: int) -> tuple[dict, bool]:
    """Gate 0. Every threshold is fixed in the plan before the numbers were seen."""
    core = sites[~sites.east_of_76]
    n = len(core)
    n_big = int((core.site_area_m2 >= 20000).sum())
    frac_big = n_big / n if n else 0.0
    n_core_2k = int((core.core_area_m2 >= 2000).sum())
    cl = core[core.cluster != ""].cluster.value_counts()
    n_clusters_5 = int((cl >= 5).sum())
    largest_frac = float(cl.iloc[0] / n) if len(cl) and n else 1.0
    cov = float(((core.has_composite) & (core.has_s1)).mean()) if n else 0.0
    base = stability.get(150.0, n)
    spread = (max(stability.values()) - min(stability.values())) / base if stability and base else 0.0

    g = {
        "G0.1_freshness": (age_days <= max_age_days, f"PBF age {age_days} d <= {max_age_days}"),
        "G0.2_volume": (n >= 120, f"{n} sites in PK (need >=120)"),
        "G0.3_size": (n_big >= 100 and frac_big >= 0.60,
                      f"{n_big} sites >=20k m2 ({frac_big:.0%}); need >=100 and >=60%"),
        "G0.4_spread": (n_clusters_5 >= 8 and largest_frac <= 0.25,
                        f"{n_clusters_5} clusters with >=5 sites, largest {largest_frac:.0%}"),
        "G0.5_stability": (spread < 0.10, f"site count spread {spread:.1%} over link_m sweep"),
        "G0.6_coverage": (cov >= 0.60, f"{cov:.0%} of sites have S2+S1 composites"),
    }
    metrics = {
        "n_sites_total": len(sites), "n_sites_pk": n, "n_east_of_76": int(sites.east_of_76.sum()),
        "n_sites_new_vs_old_pbf": n_new if n_new is not None else "",
        "pbf_age_days": age_days,
        "n_site_area_ge_20k": n_big, "frac_site_area_ge_20k": round(frac_big, 4),
        "n_core_area_ge_2000": n_core_2k,
        "median_core_area_m2": round(float(core.core_area_m2.median()), 1) if n else 0,
        "median_site_area_m2": round(float(core.site_area_m2.median()), 1) if n else 0,
        "n_clusters_ge_5": n_clusters_5, "largest_cluster_frac": round(largest_frac, 4),
        "frac_with_composite_s1": round(cov, 4),
        "n_holdout": int((core.split == "holdout").sum()),
        "n_has_penstock": int(core.has_penstock.sum()),
        "link_m_stability": json.dumps({str(k): v for k, v in stability.items()}),
    }
    for k, (ok, msg) in g.items():
        metrics[k] = "PASS" if ok else "FAIL"
        metrics[k + "_detail"] = msg
    return metrics, all(ok for ok, _ in g.values())


# ------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--bbox", default=",".join(str(x) for x in NORTH_BBOX))
    ap.add_argument("--buffer-m", type=float, default=75.0)
    ap.add_argument("--link-m", type=float, default=150.0)
    ap.add_argument("--attach-m", type=float, default=150.0)
    ap.add_argument("--max-conduit-len-m", type=float, default=2000.0)
    ap.add_argument("--refresh-pbf", action="store_true")
    ap.add_argument("--max-pbf-age-days", type=int, default=14)
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--compare-pbf", type=Path, default=OSM_DIR / "pakistan-latest.osm.pbf",
                    help="second PBF to diff against, to measure how much recent mapping is new")
    ap.add_argument("--with-headrace", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bbox = tuple(float(x) for x in a.bbox.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if out.resolve() == PROD_LABELS.resolve():
        raise SystemExit(f"refusing to write into the production label dir {PROD_LABELS}")
    prod_before = {p.name: p.stat().st_mtime for p in PROD_LABELS.glob("*.parquet")}

    pbf, ts, age = fetch_pbf(a.refresh_pbf, a.max_pbf_age_days, a.allow_stale)
    feats = classify(extract(pbf, bbox, a.force))

    kw = dict(buffer_m=a.buffer_m, attach_m=a.attach_m,
              max_conduit_len_m=a.max_conduit_len_m, with_headrace=a.with_headrace)
    sites, comps = assemble_sites(feats, link_m=a.link_m, **kw)
    sites = enrich(sites)

    # G0.5: if the site count moves with the merge radius, the site definition is
    # arbitrary and every downstream n is soft. Cheap to check, so always check.
    stability = {}
    for lm in (100.0, 150.0, 250.0):
        stability[lm] = len(sites) if lm == a.link_m else len(assemble_sites(feats, link_m=lm, **kw)[0])
    log.info("link_m stability: %s", stability)

    # Buffer sensitivity table -- Stage 1/2 must be able to show the result is not an
    # artefact of the buffer choice.
    buf_table = {}
    for bm in (25.0, 75.0, 150.0):
        s = sites if bm == a.buffer_m else assemble_sites(feats, link_m=a.link_m,
                                                          **{**kw, "buffer_m": bm})[0]
        buf_table[bm] = {"n": len(s), "median_area_m2": round(float(s.site_area_m2.median()), 1),
                         "n_ge_20k": int((s.site_area_m2 >= 20000).sum())}
    log.info("buffer sensitivity: %s", json.dumps(buf_table))

    n_new = None
    if a.compare_pbf and Path(a.compare_pbf).exists() and Path(a.compare_pbf) != pbf:
        try:
            old = classify(extract(Path(a.compare_pbf), bbox, force=False))
            old_ids = set(old.osm_uid.dropna())
            new_ids = set(feats.osm_uid.dropna())
            n_new = len(new_ids - old_ids)
            log.info("vs %s: %d hydro features are new (%d -> %d)",
                     Path(a.compare_pbf).name, n_new, len(old_ids), len(new_ids))
        except SystemExit as e:
            log.warning("could not diff against %s: %s", a.compare_pbf, e)

    # Write. Nothing here may collide with a production filename.
    name = "hydro_sites_headrace" if a.with_headrace else "hydro_sites"
    sites.to_parquet(out / f"{name}.parquet")
    comps.to_parquet(out / "hydro_sites_components.parquet")
    for cls, fn in [("plant", "hydro_plant_poly"), ("generator", "hydro_gen"),
                    ("conduit", "hydro_conduit")]:
        feats[feats.hydro_class == cls].drop(columns=["hydro_class"], errors="ignore") \
            .to_parquet(out / f"{fn}.parquet")
    clash = {p.name for p in out.glob("*.parquet")} & FORBIDDEN_NAMES
    assert not clash, f"wrote a production-owned filename into {out}: {clash}"

    metrics, passed = evaluate_gates(sites, age, n_new, stability, a.max_pbf_age_days)
    metrics.update({"run_name": name, "pbf": pbf.name, "pbf_timestamp": str(ts),
                    "config_hash": CONFIG_HASH, "buffer_m": a.buffer_m, "link_m": a.link_m,
                    "buffer_sensitivity": json.dumps(buf_table),
                    "generated": dt.datetime.now().isoformat(timespec="seconds")})
    (out / "hydro_extract_manifest.json").write_text(json.dumps(metrics, indent=2, default=str))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inv = sites.drop(columns=["geometry"])
    inv.to_csv(RESULTS_DIR / "hydro_stage0_inventory.csv", index=False)
    summ = RESULTS_DIR / "hydro_stage0_summary.csv"
    row = pd.DataFrame([metrics])
    if summ.exists():
        prev = pd.read_csv(summ)
        prev = prev[~((prev.run_name == name) & (prev.config_hash == CONFIG_HASH))]
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(summ, index=False)

    prod_after = {p.name: p.stat().st_mtime for p in PROD_LABELS.glob("*.parquet")}
    assert prod_before == prod_after, "production substation labels were modified!"

    print("\n=== Stage 0: hydro site inventory ===")
    print(f"PBF {pbf.name}  snapshot {ts}  age {age} d"
          + (f"   ({n_new} hydro features new vs {Path(a.compare_pbf).name})" if n_new is not None else ""))
    print(f"sites: {len(sites)} total, {metrics['n_sites_pk']} in PK "
          f"({metrics['n_east_of_76']} flagged east of lon 76)")
    print(f"median core (powerhouse) area: {metrics['median_core_area_m2']:,.0f} m2   "
          f"-> median site-complex area: {metrics['median_site_area_m2']:,.0f} m2")
    print(f"sites >=20k m2 (the detectable bucket): {metrics['n_site_area_ge_20k']} "
          f"({metrics['frac_site_area_ge_20k']:.0%})")
    print(f"  honest count without the buffer, core >=2,000 m2: {metrics['n_core_area_ge_2000']}")
    print(f"holdout (Swat/Dir): {metrics['n_holdout']} sites   "
          f"with penstock mapped: {metrics['n_has_penstock']}")
    print("\nsize buckets (core / powerhouse only):")
    print(sites.size_bucket.value_counts().to_string())
    print("\ncluster counts:")
    print(sites[sites.cluster != ""].cluster.value_counts().to_string())
    print("\n--- gates ---")
    for k in sorted(k for k in metrics if k.startswith("G0.")):
        if k.endswith("_detail"):
            continue
        print(f"  {metrics[k]:4}  {k:18} {metrics[k + '_detail']}")
    print(f"\nGATE 0: {'PASS' if passed else 'FAIL'}")
    print(f"\nwrote {out}/{name}.parquet  and  {summ}")


if __name__ == "__main__":
    main()
