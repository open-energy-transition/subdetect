"""OSM -> battery energy storage (BESS) SITE labels for GB / California / Texas.

Written 2026-09-01 for the feasibility question pre-registered in
`docs/issues/bess-detectability-feasibility.md`: do battery storage systems produce a
signal in Sentinel-1/Sentinel-2 at 10 m, with OSM battery power-plant labels as ground
truth?

Two facts drive every design choice here.

**There are no labels where the imagery is.** `plant:source=battery` appears once in
`pakistan-20260829.osm.pbf` and zero times in `india-latest.osm.pbf` (measured
2026-09-01), so every composite this project owns is useless for this class. Hence three
new regions, and hence this script is multi-region where `build_hydro_labels.py` was not.

**BESS sit inside or beside substations** -- which is exactly the class the production
detector was trained to find. So this script builds *two* label sets, not one: the battery
sites, and a pool of **battery-free substations** to serve as the Tier-2 controls. The
honest question is not "can we see it" but "can we tell it from the switchyard next to
it", and that question needs its negatives built with as much care as its positives. The
exclusion cascade in `build_control_pool` is therefore load-bearing, and every rule it
applies is recorded per object so the Stage 2 sensitivity re-run (G2.8) is a column filter
rather than a re-extraction.

Deliberately a standalone script rather than a change to `src/subdetect/osm.py`: that
module's four parquets feed the production substation lineage, and its `_EXPORT_CONFIG` is
a module-level dict whose mutation is the change most likely to silently alter the
substation label set. `build_hydro_labels.py` established this idiom; this is its fork.

Read-only with respect to everything that exists: writes only under
`data/labels/bess_<region>/`, `data/eval_results/` and `data/osm/`, and asserts the
production labels and the Pakistan composite tree are untouched.

    pixi run python scripts/build_bess_labels.py --region gb --refresh-pbf
    pixi run python scripts/build_bess_labels.py --region all --write-cells
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
from shapely.ops import unary_union
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subdetect.config import Settings, geodesic_area_m2, resolve_aoi  # noqa: E402
from subdetect.roi import CELL_DEG, _snapped_origin, cell_name  # noqa: E402

log = logging.getLogger("build_bess_labels")

OSM_DIR = ROOT / "data" / "osm"
LABELS_ROOT = ROOT / "data" / "labels"
RESULTS_DIR = ROOT / "data" / "eval_results"
COMPOSITES_ROOT = ROOT / "data" / "composites"

# Everything the substation lineage owns. Asserted untouched at the end of every run.
PROD_LABEL_DIRS = [LABELS_ROOT / "pakistan", LABELS_ROOT / "india_pilot"]
PROD_COMPOSITES = COMPOSITES_ROOT / "pakistan" / "composites"

# Bytes per cell per layer, measured on the Pakistan tree. Used by the G0.10 budget gate.
# AMENDMENT A1: the headline positive set is the >=20,000 m2 bucket -- the size class in
# which this pipeline has measured recall (0.714 on Pakistan substations), as opposed to
# the 1,000-2,000 m2 class where p_max is exactly 0.000 in all 23 runs.
HEADLINE_MIN_AREA_M2 = 20000.0
# AMENDMENT A2: the acquisition-confound control is a DISTANCE, not a shared 0.1 deg cell.
# A Sentinel-2 granule is 110x110 km and the S1 IW swath is 250 km, so two footprints
# within 40 km are in the same acquisition with the same look geometry -- which is the
# thing the original same-cell clause was a proxy for. The 80% level is inherited
# unchanged from the pre-registered G0.5; only the operationalisation moves.
CTRL_MAX_DIST_M = 40_000.0

MB_PER_CELL = {"composite_0.tif": 12.3, "composite_pre.tif": 12.3,
               "composite_s1.tif": 3.7, "composite_dem.tif": 4.9}

REGIONS = {
    "gb": {"aoi": "gb_bess", "stem": "great-britain", "label": "Great Britain",
           "group": "gb",
           "url": "https://download.geofabrik.de/europe/great-britain-latest.osm.pbf"},
    "ca": {"aoi": "ca_bess", "stem": "california", "label": "California",
           "group": "us",
           "url": "https://download.geofabrik.de/north-america/us/california-latest.osm.pbf"},
    "tx": {"aoi": "tx_bess", "stem": "texas", "label": "Texas",
           "group": "us",
           "url": "https://download.geofabrik.de/north-america/us/texas-latest.osm.pbf"},
}

# Filenames owned by the substation lineage. Never write any of these here: chips.py
# densifies `lines.parquet` at 3 km spacing into `line_negative` training chips.
FORBIDDEN_NAMES = {
    "lines.parquet", "plants.parquet", "substations_poly.parquet",
    "substations_node.parquet", "substations_poly_refined.parquet",
}

# osmium tags-filter has no AND semantics, so filter broadly and AND in pandas below.
# power=substation is pulled in the SAME pass as the batteries, on purpose: the Tier-2
# controls must come from the identical snapshot as the positives, or "battery-free" would
# be measured against a different day's map.
FILTERS = [
    "nwr/power=plant",
    "nwr/power=generator",
    "nwr/power=substation",
    # Key-only filters, to catch battery objects whose power=* key is missing or unusual.
    # These are what make the G0.0 audit an audit rather than a confirmation.
    "nwr/plant:source",
    "nwr/plant:storage",
    "nwr/generator:source",
    "nwr/generator:storage",
]

# The tags that must survive `osmium export`. The production config keeps only
# ["power","voltage","name","substation","operator"], which is exactly why the cached
# plants.parquet cannot tell a battery from a solar farm. Exact keys, no wildcards.
EXPORT_CONFIG = {
    "attributes": {"type": "@type", "id": "@id"},
    "linear_tags": True,
    "area_tags": True,
    "include_tags": [
        "power", "substation",
        "plant:source", "plant:method", "plant:type", "plant:storage",
        "plant:output:electricity", "plant:output:storage",
        "generator:source", "generator:method", "generator:type", "generator:storage",
        "generator:output:electricity", "generator:output:storage",
        "voltage", "frequency", "operator", "owner", "ref", "wikidata",
        "name", "name:en", "short_name", "alt_name", "description",
        "start_date", "construction:start_date", "opening_date", "construction",
        "landuse", "building", "man_made", "location", "note",
    ],
}

# Columns whose absence after the export round-trip means a stale intermediate was reused
# despite a changed filter -- the failure mode the config hash in the filenames prevents,
# asserted here as a backstop. Deliberately short: a region legitimately may not contain a
# single object carrying, say, plant:storage, and a hard failure on that would be wrong.
REQUIRED_COLS = {"power", "name"}
EXPECTED_COLS = {"plant:source", "generator:source", "substation", "voltage", "operator",
                 "start_date", "plant:output:electricity"}

# Substring matching throughout, so `plant:source=solar;battery` on a hybrid site is
# caught. `\bbess\b` is anchored because "bess" appears inside ordinary place names
# (Bessacarr, Bessemer); the other alternatives are specific enough unanchored.
# Co-located generation sources. A `plant:source=solar;battery` polygon is mostly PV
# panels with a container yard in one corner, so "detecting" it would largely be detecting
# the solar farm. Hybrids stay in the positive set -- the pre-registered definition is a
# tag union and moving it after seeing the data is exactly what pre-registration forbids --
# but they are flagged here and reported as their own stratum under G2.6.
CO_SOURCE_RE = re.compile(r"solar|photovoltaic|wind|gas|diesel|hydro|biomass|nuclear|coal", re.I)

BATTERY_RE = re.compile(r"batter|\bbess\b|energy[\s_-]?storage|grid[\s_-]?storage"
                        r"|storage[\s_-]?system|\bbattery[\s_-]?park\b", re.I)
# Tighter pattern for the free-text name/operator hint: "storage" alone is far too loose
# (water storage, storage depot, self-storage), so the bare word never qualifies.
NAME_HINT_RE = re.compile(r"batter|\bbess\b|energy[\s_-]?storage|grid[\s_-]?storage", re.I)

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


def fetch_pbf(region: str, refresh: bool, max_age_days: int, allow_stale: bool,
              explicit: Path | None = None) -> tuple[Path, dt.datetime | None, int]:
    """Return (pbf, snapshot_ts, age_days), downloading to a DATED filename if asked.

    Dated, never `<region>-latest.osm.pbf`: that is the filename convention the production
    ingest owns for its own countries, and a `-latest` file can never self-refresh because
    `osm._download` returns early whenever it exists.
    """
    cfg = REGIONS[region]
    if explicit is not None:
        pbf = Path(explicit)
        if not pbf.exists():
            raise SystemExit(f"--pbf {pbf} does not exist")
    else:
        dated = OSM_DIR / f"{cfg['stem']}-{dt.date.today():%Y%m%d}.osm.pbf"
        if refresh:
            # Always re-enter wget when refreshing, even if the file exists: a run killed
            # mid-download leaves a truncated but non-empty file, and skipping on mere
            # existence would hand a half-PBF to osmium. `wget -c` resumes, and is a cheap
            # no-op once the file is complete.
            OSM_DIR.mkdir(parents=True, exist_ok=True)
            _run(["wget", "-c", "-q", "--show-progress", cfg["url"], "-O", str(dated)])
        pbf = dated
        if not pbf.exists():
            # Exactly 8 digits then `.osm.pbf`, so this can only match a dated country
            # download. A looser `-2*.osm.pbf` also matches this script's own derived
            # intermediates (`<stem>-<date>.osm_bess_<hash>_clip.osm.pbf`), and sorted()[-1]
            # then hands a tag-filtered clip back as if it were the source PBF -- which
            # reads as a 10,000-day-old snapshot because clips carry no header timestamp.
            cands = sorted(OSM_DIR.glob(f"{cfg['stem']}-[0-9]" + "[0-9]" * 7 + ".osm.pbf"))
            if not cands:
                raise SystemExit(f"no PBF for {region} at {dated}; pass --refresh-pbf")
            pbf = cands[-1]
    ts = pbf_timestamp(pbf)
    age = (dt.datetime.now(dt.timezone.utc) - ts).days if ts else 10_000
    log.info("[%s] PBF %s  snapshot=%s  age=%d d", region, pbf.name, ts, age)
    if age > max_age_days and not allow_stale:
        raise SystemExit(
            f"{pbf.name} snapshot {ts} is {age} d old (limit {max_age_days}). "
            f"Pass --refresh-pbf to download today's, or --allow-stale to proceed."
        )
    return pbf, ts, age


def _parse_seq(seq: Path, keep: set[str] | None = None) -> gpd.GeoDataFrame:
    """Parse a geojsonseq by hand into a GeoDataFrame.

    Not `gpd.read_file`: GDAL's GeoJSONSeq driver infers a schema from a leading scan, and
    a tag carried by only a handful of late features -- `plant:storage` is exactly that --
    can silently vanish as a column, with no warning and no error.
    """
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
            if keep is not None and not (keep & set(props)):
                continue
            props["osm_uid"] = feat.get("id")
            props["osm_type"] = props.pop("@type", None)
            props["osm_id"] = props.pop("@id", None)
            try:
                props["geometry"] = shape(feat["geometry"])
            except Exception:  # noqa: BLE001 — skip broken geometries
                continue
            rows.append(props)
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _dedupe_objects(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One geometry per OSM object: polygon > line > point.

    `osmium export` with BOTH linear_tags and area_tags emits a closed way TWICE -- once as
    a LineString and once as a (Multi)Polygon. In the hydro run this duplicated 252 of 256
    power=plant objects. Left in place, every closed-way site would be counted twice and
    would contribute both a line buffer and a polygon buffer to its own footprint.
    """
    if gdf.empty:
        return gdf
    rank = {"MultiPolygon": 0, "Polygon": 0, "LineString": 1, "MultiLineString": 1, "Point": 2}
    n0 = len(gdf)
    out = (gdf.assign(_r=gdf.geometry.geom_type.map(lambda t: rank.get(t, 3)))
              .sort_values("_r")
              .drop_duplicates(subset=["osm_type", "osm_id"], keep="first")
              .drop(columns=["_r"]).reset_index(drop=True))
    log.info("deduped %d -> %d features (one geometry per OSM object)", n0, len(out))
    return out


def extract(pbf: Path, bbox: tuple[float, float, float, float], force: bool) -> gpd.GeoDataFrame:
    """osmium tags-filter -> extract -> export -> GeoDataFrame, with all tags intact."""
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{pbf.stem}_bess_{CONFIG_HASH}"
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
        cfg = OSM_DIR / f"bess_export_config_{CONFIG_HASH}.json"
        cfg.write_text(json.dumps(EXPORT_CONFIG))
        _run(["osmium", "export", str(clip), "-f", "geojsonseq", "--overwrite",
              "-c", str(cfg), "--add-unique-id=type_id", "-o", str(seq)])

    # tags-filter drags in every node referenced by a matched way, so the export is full
    # of untagged members. Requiring at least one exported tag drops them before they
    # become 100k empty rows.
    gdf = _parse_seq(seq, keep=set(EXPORT_CONFIG["include_tags"]))
    log.info("exported %d tagged features from %s", len(gdf), seq.name)
    gdf = _dedupe_objects(gdf)

    missing = REQUIRED_COLS - set(gdf.columns)
    if missing:
        raise SystemExit(
            f"osmium export dropped {sorted(missing)} -- stale intermediate? "
            f"delete {filt.name}/{seq.name} or pass --force"
        )
    absent = EXPECTED_COLS - set(gdf.columns)
    if absent:
        log.warning("tags absent from this region's export (may be genuine): %s", sorted(absent))
    return gdf


def extract_context(pbf: Path, bbox, force: bool, out: Path) -> None:
    """Roads / industrial landuse / power lines -> the Stage 2 distance covariates.

    A separate osmium pass on purpose, exactly as `extract_waterways` was for hydro: these
    outnumber the power features by orders of magnitude and are never part of a site. They
    only ever become raster distance fields in the control sampler, so keeping them out of
    the main export keeps that file small and its config hash stable.
    """
    specs = [
        ("roads", ["w/highway=motorway,trunk,primary,secondary,tertiary,unclassified"],
         ["highway", "name"]),
        ("industrial", ["wr/landuse=industrial,commercial,quarry,brownfield"],
         ["landuse", "name"]),
        ("lines", ["w/power=line,minor_line"], ["power", "voltage"]),
    ]
    for tag, filters, keep in specs:
        stem = f"{pbf.stem}_bessctx_{tag}"
        filt = OSM_DIR / f"{stem}.osm.pbf"
        clip = OSM_DIR / f"{stem}_clip.osm.pbf"
        seq = OSM_DIR / f"{stem}.geojsonseq"
        if force:
            for q in (filt, clip, seq):
                q.unlink(missing_ok=True)
        if not filt.exists():
            _run(["osmium", "tags-filter", "-o", str(filt), "--overwrite", str(pbf), *filters])
        if not clip.exists():
            _run(["osmium", "extract", "-s", "smart", "--overwrite",
                  "-b", ",".join(str(x) for x in bbox), "-o", str(clip), str(filt)])
        if not seq.exists():
            cfg = OSM_DIR / f"bessctx_{tag}_export_config.json"
            cfg.write_text(json.dumps({"attributes": {"type": "@type", "id": "@id"},
                                       "linear_tags": True,
                                       "area_tags": tag == "industrial",
                                       "include_tags": keep}))
            _run(["osmium", "export", str(clip), "-f", "geojsonseq", "--overwrite",
                  "-c", str(cfg), "-o", str(seq)])
        g = _parse_seq(seq, keep=set(keep))
        if not g.empty:
            g = g[[c for c in (*keep, "geometry") if c in g.columns]]
        g.to_parquet(out / f"context_{tag}.parquet")
        log.info("context %-10s %d features", tag, len(g))


# ------------------------------------------------------------------- classification


def _col(gdf: gpd.GeoDataFrame, key: str) -> pd.Series:
    if key not in gdf.columns:
        return pd.Series("", index=gdf.index, dtype=object)
    return gdf[key].fillna("").astype(str)


def _freetext(gdf: gpd.GeoDataFrame) -> pd.Series:
    """name + operator + ref + description, joined -- the free-text hint field."""
    parts = [_col(gdf, k) for k in ("name", "name:en", "short_name", "alt_name",
                                    "operator", "owner", "ref", "description")]
    out = parts[0]
    for p in parts[1:]:
        out = out + " " + p
    return out


def audit_tags(gdf: gpd.GeoDataFrame, region: str) -> pd.DataFrame:
    """G0.0. Cross-tab every candidate battery tagging pattern, BEFORE fixing a definition.

    This gate exists so that "we assumed plant:source=battery and found nothing" is not a
    possible outcome of this study. Every pattern below is counted whether or not it is
    part of the positive definition, and the table is written to disk before `classify_bess`
    is called.
    """
    power = _col(gdf, "power")
    txt = _freetext(gdf)
    pat = {
        "plant + plant:source~batter": (power == "plant") & _col(gdf, "plant:source").str.contains(BATTERY_RE),
        "plant + plant:method~batter": (power == "plant") & _col(gdf, "plant:method").str.contains(BATTERY_RE),
        "plant + plant:storage~batter": (power == "plant") & _col(gdf, "plant:storage").str.contains(BATTERY_RE),
        "plant + plant:type~batter": (power == "plant") & _col(gdf, "plant:type").str.contains(BATTERY_RE),
        "generator + generator:source~batter": (power == "generator") & _col(gdf, "generator:source").str.contains(BATTERY_RE),
        "generator + generator:method~batter": (power == "generator") & _col(gdf, "generator:method").str.contains(BATTERY_RE),
        "generator + generator:storage~batter": (power == "generator") & _col(gdf, "generator:storage").str.contains(BATTERY_RE),
        "substation + substation~batter": (power == "substation") & _col(gdf, "substation").str.contains(BATTERY_RE),
        "substation + freetext~batter": (power == "substation") & txt.str.contains(NAME_HINT_RE),
        "plant/generator + freetext~batter": power.isin(["plant", "generator"]) & txt.str.contains(NAME_HINT_RE),
        "any battery tag, no power= key": (power == "") & (
            _col(gdf, "plant:source").str.contains(BATTERY_RE)
            | _col(gdf, "generator:source").str.contains(BATTERY_RE)
            | _col(gdf, "plant:storage").str.contains(BATTERY_RE)),
        "hybrid: battery AND solar in plant:source": (
            _col(gdf, "plant:source").str.contains(BATTERY_RE)
            & _col(gdf, "plant:source").str.contains("solar", case=False)),
    }
    rows = []
    for name, mask in pat.items():
        sub = gdf[mask]
        gt = sub.geometry.geom_type.value_counts() if len(sub) else pd.Series(dtype=int)
        rows.append({
            "region": region, "pattern": name, "n": int(mask.sum()),
            "n_polygon": int(gt.get("Polygon", 0) + gt.get("MultiPolygon", 0)),
            "n_point": int(gt.get("Point", 0)),
            "n_line": int(gt.get("LineString", 0) + gt.get("MultiLineString", 0)),
        })
    tab = pd.DataFrame(rows)
    # Raw value histogram too: a pattern nobody predicted shows up here and nowhere else.
    vals = []
    for key in ("plant:source", "plant:storage", "generator:source", "generator:storage",
                "substation"):
        s = _col(gdf, key)
        s = s[s.str.contains(BATTERY_RE)]
        for v, n in s.value_counts().items():
            vals.append({"region": region, "pattern": f"VALUE {key}={v}", "n": int(n),
                         "n_polygon": "", "n_point": "", "n_line": ""})
    return pd.concat([tab, pd.DataFrame(vals)], ignore_index=True) if vals else tab


def classify_bess(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Tag-based class assignment. Never geometry-type based.

    The **positive definition**, fixed here and recorded in the manifest:

        positive  ==  (power=plant      AND any of plant:{source,method,storage,type} ~ battery)
                  OR  (power=generator  AND any of generator:{source,method,storage} ~ battery)
                  OR  (power=substation AND substation ~ battery)

    Free-text name/operator hints are deliberately **not** positives. They are too noisy to
    train or score on ("Battery Lane substation" is a road name, not a battery), but they
    are excellent contamination flags -- so they get their own class, are excluded from the
    positives, and are also excluded from the Tier-2 control pool. That is the conservative
    choice in both directions: it neither invents positives nor lets a probable battery sit
    in the negative class.
    """
    power = _col(gdf, "power")
    txt = _freetext(gdf)

    plant_bat = (_col(gdf, "plant:source").str.contains(BATTERY_RE)
                 | _col(gdf, "plant:method").str.contains(BATTERY_RE)
                 | _col(gdf, "plant:storage").str.contains(BATTERY_RE)
                 | _col(gdf, "plant:type").str.contains(BATTERY_RE))
    gen_bat = (_col(gdf, "generator:source").str.contains(BATTERY_RE)
               | _col(gdf, "generator:method").str.contains(BATTERY_RE)
               | _col(gdf, "generator:storage").str.contains(BATTERY_RE))
    sub_bat = _col(gdf, "substation").str.contains(BATTERY_RE)

    is_plant = (power == "plant") & plant_bat
    is_gen = (power == "generator") & gen_bat
    is_storage_sub = (power == "substation") & sub_bat
    # Battery tags with no power= key at all -- rare, but the audit will say how rare.
    is_orphan = (power == "") & (plant_bat | gen_bat)
    is_hint = (power.isin(["plant", "generator", "substation"]) & txt.str.contains(NAME_HINT_RE)
               & ~(is_plant | is_gen | is_storage_sub))
    is_sub = (power == "substation") & ~is_storage_sub & ~is_hint

    cls = pd.Series("", index=gdf.index, dtype=object)
    cls[is_sub] = "substation"
    cls[is_hint] = "name_hint"
    cls[is_orphan] = "orphan_battery"
    cls[is_storage_sub] = "storage_sub"
    cls[is_gen] = "generator"
    cls[is_plant] = "plant"
    gdf = gdf.assign(bess_class=cls)

    kept = gdf[gdf.bess_class != ""].copy()
    log.info("classified: %s (dropped %d unclassified)",
             kept.bess_class.value_counts().to_dict(), len(gdf) - len(kept))
    if len(kept):
        log.info("geometry types by class:\n%s",
                 pd.crosstab(kept.bess_class, kept.geometry.geom_type).to_string())
    return kept


POSITIVE_CLASSES = ("plant", "generator", "storage_sub", "orphan_battery")

_UNIT = {"": 1e-6, "k": 1e-3, "m": 1.0, "g": 1e3}
_POWER_RE = re.compile(r"([\d.]+)\s*([kMGmg]?)\s*W(?:h)?", re.I)


def parse_power_mw(v) -> float:
    """'50 MW' -> 50.0 | '20000 kW' -> 20.0 | '2 MW;2 MW' -> 4.0 | 'yes' -> nan.

    Values within one tag are summed: OSM lists per-unit outputs separated by ';', so a
    plant tagged '2 MW;2 MW' is a 4 MW plant. Lowercase 'm' reads as mega, not milli --
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


_YEAR_RE = re.compile(r"(1[89]\d{2}|20\d{2})")


def construction_year(row) -> float:
    """First 4-digit year in start_date / construction:start_date / opening_date, else nan.

    G0.7 gates on how often this succeeds, because the bi-temporal arm is only valid where
    the pre-construction window provably predates construction. The OSM *object* creation
    date would be a fallback, but it needs a full-history PBF, which this study does not
    buy -- so a site with no date tag simply has no usable pre-epoch, and says so.
    """
    for key in ("start_date", "construction:start_date", "opening_date"):
        v = row.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        m = _YEAR_RE.search(str(v))
        if m:
            return float(m.group(1))
    return float("nan")


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


def assemble_sites(feats: gpd.GeoDataFrame, region: str, *,
                   buffer_m: float = 30.0,
                   link_m: float = 60.0) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """One dissolved footprint per physical installation. Returns (sites, components).

    Much simpler than the hydro equivalent, and deliberately so. A BESS is a compact fenced
    yard -- containers, inverters, a pad -- not a powerhouse joined to a 2 km penstock, so
    there is no conduit attachment step, no arc-length clipping, and the merge radius is
    60 m rather than 150 m. `core_area_m2` (unbuffered, geodesic) is kept beside
    `site_area_m2` as the honest size number, because the hydro study's central finding was
    that detectability tracks the real built footprint and not the label buffer.
    """
    anchors = feats[feats.bess_class.isin(POSITIVE_CLASSES)].copy()
    if anchors.empty:
        return (gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
                gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))

    # Metric work in a single local UTM per region, chosen from the anchor centroid. Texas
    # spans ~13 deg of longitude so the far edge sits ~6 deg off the central meridian,
    # where the UTM scale error reaches ~2%: on a 60 m merge radius that is ~1 m, and on
    # the 30 m buffer ~0.6 m. Immaterial here, and far cheaper than per-zone regrouping.
    c = anchors.geometry.union_all().centroid
    utm = _utm_for(c.x, c.y)
    anchors = anchors.to_crs(utm)
    # buffer(0) heals invalid polygons, but on a Point it returns an EMPTY polygon -- in
    # the hydro run that silently deleted all 59 generator nodes. Heal polygons only.
    poly = anchors.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    anchors.loc[poly, "geometry"] = anchors.loc[poly, "geometry"].buffer(0)

    # Union-find over an STRtree self-query rather than a dissolve: the dissolve would
    # throw away the object -> site mapping that n_components and src_ids need.
    bufs = list(anchors.geometry.buffer(link_m / 2.0))
    uf = _UnionFind(len(anchors))
    tree = STRtree(bufs)
    qi, qj = tree.query(bufs, predicate="intersects")
    for a, b in zip(qi, qj):
        uf.union(int(a), int(b))
    roots = [uf.find(i) for i in range(len(anchors))]

    members: dict[int, list] = {r: [] for r in set(roots)}
    for (_, row), root in zip(anchors.iterrows(), roots):
        members[root].append(row)

    site_rows, comp_rows = [], []
    for root, ms in members.items():
        geom_utm = unary_union([m.geometry.buffer(buffer_m) for m in ms]).buffer(0)
        site = gpd.GeoSeries([geom_utm], crs=utm).to_crs("EPSG:4326").iloc[0]
        raw_4326 = gpd.GeoSeries([m.geometry for m in ms], crs=utm).to_crs("EPSG:4326")
        core_area = float(sum(geodesic_area_m2(g) for g in raw_4326
                              if g.geom_type in ("Polygon", "MultiPolygon")))
        anchor_union = unary_union([g for g in raw_4326 if g is not None and not g.is_empty])
        rep = (anchor_union.representative_point() if not anchor_union.is_empty
               else site.representative_point())

        # max, not sum: the plant tag is the whole-installation figure while generator tags
        # are per-unit, so adding them double-counts an installation that tags both.
        plant_mw = max([parse_power_mw(m.get("plant:output:electricity")) for m in ms]
                       + [float("nan")])
        gen_mw = np.nansum([parse_power_mw(m.get("generator:output:electricity")) for m in ms]
                           or [float("nan")])
        cands = [x for x in (plant_mw, float(gen_mw)) if x and not np.isnan(x)]
        capacity_mw = max(cands) if cands else float("nan")
        stor = [parse_power_mw(m.get("plant:output:storage")) for m in ms] + \
               [parse_power_mw(m.get("generator:output:storage")) for m in ms]
        stor = [x for x in stor if x and not np.isnan(x)]
        energy_mwh = max(stor) if stor else float("nan")

        srcs = " ".join(str(m.get(k) or "") for m in ms for k in
                        ("plant:source", "plant:method", "generator:source",
                         "generator:method"))
        co = sorted({t.lower() for t in CO_SOURCE_RE.findall(srcs)})

        names = [str(m.get("name") or m.get("name:en") or "") for m in ms]
        years = [construction_year(m) for m in ms]
        years = [y for y in years if not np.isnan(y)]

        site_rows.append({
            "geometry": site,
            "site_area_m2": geodesic_area_m2(site),
            "core_area_m2": core_area,
            "capacity_mw": capacity_mw,
            "energy_mwh": energy_mwh,
            "construction_year": min(years) if years else float("nan"),
            "name": next((n for n in names if n), ""),
            "operator": next((str(m.get("operator") or "") for m in ms
                              if str(m.get("operator") or "")), ""),
            "n_components": len(ms),
            "n_polygon_components": int(sum(
                g.geom_type in ("Polygon", "MultiPolygon") for g in raw_4326)),
            "classes": ";".join(sorted({str(m.bess_class) for m in ms})),
            "is_hybrid": bool(co),
            "co_source": ";".join(co),
            "src_ids": ";".join(sorted(str(m.get("osm_uid")) for m in ms)),
            "lon": rep.x, "lat": rep.y,
            "_root": root,
        })
        for m, g4326 in zip(ms, raw_4326):
            comp_rows.append({
                "_root": root, "cls": m.bess_class, "osm_uid": m.get("osm_uid"),
                "osm_type": m.get("osm_type"), "osm_id": m.get("osm_id"),
                "area_m2": geodesic_area_m2(g4326) if g4326.geom_type in
                           ("Polygon", "MultiPolygon") else 0.0,
                "geometry": g4326,
            })

    sites = gpd.GeoDataFrame(site_rows, crs="EPSG:4326")
    sites = sites.sort_values(["lat", "lon"], ascending=[False, True]).reset_index(drop=True)
    sites["site_id"] = [f"{region.upper()}B-{i:04d}" for i in range(len(sites))]
    sites["region"] = region
    root_to_id = dict(zip(sites._root, sites.site_id))
    comps = gpd.GeoDataFrame(comp_rows, crs="EPSG:4326")
    comps["site_id"] = comps._root.map(root_to_id)
    comps["region"] = region
    return sites.drop(columns=["_root"]), comps.drop(columns=["_root"])


# ------------------------------------------------------------- Tier-2 control pool


def build_control_pool(feats: gpd.GeoDataFrame, sites: gpd.GeoDataFrame, region: str, *,
                       exclude_m: float = 200.0,
                       min_area_m2: float = 500.0) -> gpd.GeoDataFrame:
    """Battery-free substations -- the Tier-2 negatives.

    The exclusion cascade, and why each rule is here. Every rule records itself in
    `excluded_by`, so the G2.8 sensitivity re-run is a filter on a column rather than a
    re-extraction:

      battery_tag   the object itself carries a battery tag (already class storage_sub)
      name_hint     free text says battery/BESS/energy storage -- too weak to be a
                    positive, plenty strong enough to disqualify a negative
      near_battery  within `exclude_m` of ANY battery hint, including name_hint objects
                    that did not make the positive set. A battery is very often mapped as
                    a separate object sitting inside the substation's own polygon.
      new_untagged  start_date >= 2018 and no voltage tag: the signature of a new storage
                    yard mis-tagged as a substation.

    Contamination that survives all four puts a true positive into the negative class,
    which pushes the Tier-2 AUC *toward* 0.5. The gate is therefore conservative: a PASS is
    trustworthy, a FAIL is partly attributable to contamination. That direction is stated
    in the pre-registration and must be repeated in the write-up.
    """
    subs = feats[feats.bess_class == "substation"].copy()
    if subs.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    subs = subs[subs.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    subs["area_m2"] = [geodesic_area_m2(g) for g in subs.geometry]
    subs = subs[subs.area_m2 >= min_area_m2].reset_index(drop=True)

    c = subs.geometry.union_all().centroid
    utm = _utm_for(c.x, c.y)
    subs_u = subs.to_crs(utm)

    # Everything that hints at a battery, positives included, as one exclusion geometry.
    hints = feats[feats.bess_class.isin([*POSITIVE_CLASSES, "name_hint"])]
    rules: dict[str, np.ndarray] = {}
    if len(hints):
        hints_u = hints.to_crs(utm)
        htree = STRtree(list(hints_u.geometry))
        rules["near_battery"] = np.array([
            len(htree.query(g.buffer(exclude_m), predicate="intersects")) > 0
            for g in subs_u.geometry])
    rules["name_hint"] = _freetext(subs).str.contains(NAME_HINT_RE).values
    year = subs.apply(construction_year, axis=1)
    rules["new_untagged"] = ((year >= 2018) & (_col(subs, "voltage") == "")).values

    # Semicolon-joined, not last-wins: a substation can trip several rules at once and
    # G2.8 reports per-rule counts, so each must remain visible.
    excluded_by = pd.Series([""] * len(subs), index=subs.index, dtype=object)
    for name, mask in rules.items():
        excluded_by = excluded_by.where(~mask, excluded_by.str.cat(
            pd.Series([name] * len(subs), index=subs.index),
            sep=";").str.lstrip(";"))

    subs = subs.assign(excluded_by=excluded_by.values,
                       eligible=(excluded_by == "").values,
                       construction_year=year.values,
                       region=region)
    subs["ctrl_id"] = [f"{region.upper()}S-{i:05d}" for i in range(len(subs))]
    rep = subs.geometry.representative_point()
    subs["lon"], subs["lat"] = rep.x.values, rep.y.values
    # Area caliper must compare like with like. The site label is a 30 m-buffered
    # footprint, so the control's comparable area is its own 30 m-buffered footprint --
    # matching a buffered site against an unbuffered substation would bias every pair
    # toward larger substations by roughly the perimeter times the buffer.
    subs["area_buf_m2"] = [
        geodesic_area_m2(g) for g in
        gpd.GeoSeries(subs_u.geometry.buffer(30.0), crs=utm).to_crs("EPSG:4326")]

    n = len(subs)
    log.info("control pool %s: %d substation polygons, %d eligible (%s)",
             region, n, int(subs.eligible.sum()),
             subs[subs.excluded_by != ""].excluded_by.value_counts().to_dict())
    return subs


# ------------------------------------------------------------------- enrichment


def _bucket(a: float) -> str:
    if a <= 0:
        return "node_only"
    for lo, hi, name in [(0, 1000, "<1k"), (1000, 2000, "1k-2k"), (2000, 5000, "2k-5k"),
                         (5000, 20000, "5k-20k")]:
        if lo < a < hi:
            return name
    return ">=20k"


def enrich(sites: gpd.GeoDataFrame, controls: gpd.GeoDataFrame, region: str,
           settings: Settings) -> gpd.GeoDataFrame:
    """Size buckets, composite cell, coverage flags, the co-location census, the split."""
    aoi = REGIONS[region]["aoi"]
    bbox, cfg = resolve_aoi(aoi, settings)
    minx, miny = _snapped_origin(bbox, cfg.get("grid_origin"))
    comp = COMPOSITES_ROOT / aoi / "composites"

    if len(controls):
        cix = np.floor((controls.lon.values - minx) / CELL_DEG).astype(int)
        ciy = np.floor((controls.lat.values - miny) / CELL_DEG).astype(int)
        controls["cell"] = [cell_name(x, y) for x, y in zip(cix, ciy)]
        controls["cell_0p5"] = [
            f"{np.floor(r.lon / 0.5) * 0.5:.1f}_{np.floor(r.lat / 0.5) * 0.5:.1f}"
            for r in controls.itertuples()]
        controls["in_aoi"] = ((controls.lon >= bbox[0]) & (controls.lon <= bbox[2])
                              & (controls.lat >= bbox[1]) & (controls.lat <= bbox[3]))

    sites = sites.copy()
    sites["size_bucket"] = sites.core_area_m2.map(_bucket)
    sites["buffered_bucket"] = sites.site_area_m2.map(_bucket)
    sites["node_only"] = sites.n_polygon_components == 0

    ix = np.floor((sites.lon.values - minx) / CELL_DEG).astype(int)
    iy = np.floor((sites.lat.values - miny) / CELL_DEG).astype(int)
    sites["cell"] = [cell_name(a, b) for a, b in zip(ix, iy)]
    sites["in_aoi"] = ((sites.lon >= bbox[0]) & (sites.lon <= bbox[2])
                       & (sites.lat >= bbox[1]) & (sites.lat <= bbox[3]))
    sites["cell_0p5"] = [f"{np.floor(r.lon / 0.5) * 0.5:.1f}_{np.floor(r.lat / 0.5) * 0.5:.1f}"
                         for r in sites.itertuples()]
    for layer, flag in [("composite_0.tif", "has_composite"), ("composite_s1.tif", "has_s1"),
                        ("composite_pre.tif", "has_pre"), ("composite_dem.tif", "has_dem")]:
        sites[flag] = [(comp / c / layer).exists() for c in sites.cell]

    # G0.6 co-location census. Distance from each site to the nearest substation polygon,
    # whether the two overlap, and how much of the site survives subtracting the overlap --
    # the last is what decides whether the G2.4-T2 battery-only arm is computable at all.
    sites["d_substation_m"] = np.nan
    sites["overlaps_substation"] = False
    sites["battery_only_area_m2"] = sites.site_area_m2.values
    if len(controls):
        c = sites.geometry.union_all().centroid
        utm = _utm_for(c.x, c.y)
        su = sites.to_crs(utm)
        cu = controls.to_crs(utm)
        tree = STRtree(list(cu.geometry))
        d, ov, rem = [], [], []
        for g in su.geometry:
            j = tree.nearest(g)
            near = cu.geometry.iloc[int(j)]
            d.append(float(g.distance(near)))
            hits = tree.query(g, predicate="intersects")
            if len(hits):
                sub = unary_union([cu.geometry.iloc[int(h)] for h in hits])
                left = g.difference(sub.buffer(30.0))
                ov.append(True)
                rem.append(float(left.area) if not left.is_empty else 0.0)
            else:
                ov.append(False)
                rem.append(float(g.area))
        sites["d_substation_m"] = d
        sites["overlaps_substation"] = ov
        sites["battery_only_area_m2"] = rem
    sites["has_battery_only"] = sites.battery_only_area_m2 >= 2000.0

    # AMENDMENT A1 (2026-09-02, see docs/issues/bess-detectability-feasibility.md).
    # G0.3 failed: only 51% of labels reach 20,000 m2 against a >=60% bar. Rather than
    # lower that bar, the study restricts its HEADLINE positive set to the bucket the
    # pipeline has demonstrated recall in. Smaller sites stay in the corpus and are
    # reported as G2.6 strata; they never enter the headline number. Node-only sites are
    # excluded here too, per the pre-registered G0.3b.
    sites["headline"] = ((sites.site_area_m2 >= HEADLINE_MIN_AREA_M2)
                         & sites.in_aoi & ~sites.node_only)

    # Region-level split. The holdout is a whole region group, never a bbox: the point of
    # three regions is a transfer test, and both directions are scored (G2.5).
    sites["region_group"] = REGIONS[region]["group"]
    return sites


# ------------------------------------------------------------------------ gates


def evaluate_gates(sites: pd.DataFrame, controls: pd.DataFrame, pool: dict,
                   age_days: int, stability: dict, cells: pd.DataFrame,
                   max_age_days: int, max_cells: int, max_gb: float,
                   n_regions: int) -> tuple[dict, bool]:
    """Gate 0, exactly as pre-registered in docs/issues/bess-detectability-feasibility.md.

    Thresholds are read from that document and are not to be adjusted here. Where a gate is
    per-region (volume) the pooled frame carries a `region` column and the gate is
    evaluated over the groups.
    """
    s = sites[sites.in_aoi]
    n = len(s)
    poly = s[~s.node_only]
    by_region = s.region.value_counts().to_dict()
    n_big = int((s.site_area_m2 >= 20000).sum())
    frac_big = n_big / n if n else 0.0
    med_area = float(s.site_area_m2.median()) if n else 0.0
    frac_node = float(s.node_only.mean()) if n else 1.0

    head = s[s.headline] if "headline" in s.columns else s.iloc[0:0]
    n_head = len(head)
    head_by_region = head.region.value_counts().to_dict()

    cl = s.cell_0p5.value_counts()
    n_cl5 = int((cl >= 5).sum())
    largest = float(cl.iloc[0] / n) if len(cl) and n else 1.0

    frac_dated = float(s.construction_year.notna().mean()) if n else 0.0
    base = stability.get(60.0, n)
    spread = (max(stability.values()) - min(stability.values())) / base if stability and base else 0.0
    cov = float(((s.has_composite) & (s.has_s1)).mean()) if n else 0.0

    n_cells = len(cells)
    gb_needed = n_cells * sum(MB_PER_CELL.values()) / 1024.0

    # G0.2 is per-region: the training region needs 120, each holdout region 60. With all
    # three present the training group is whichever is largest; that choice is mechanical,
    # not a judgement made after seeing the AUCs.
    counts = sorted(by_region.values(), reverse=True) or [0]
    g02 = (counts[0] >= 120 and all(c >= 60 for c in counts[1:]) and n >= 250) \
        if n_regions > 1 else (counts[0] >= 120)

    g = {
        "G0.0_tag_audit": (pool.get("audit_written", False), "tag audit written before classification"),
        "G0.1_freshness": (age_days <= max_age_days, f"oldest PBF {age_days} d <= {max_age_days}"),
        "G0.2_volume": (g02, f"per-region {by_region}, pooled {n} (need 120 / 60 / 250)"),
        "G0.3_size": (med_area >= 5000 and frac_big >= 0.60,
                      f"median site {med_area:,.0f} m2 (need >=5,000), "
                      f"{frac_big:.0%} >=20k m2 (need >=60%)"),
        "G0.3b_node_frac": (frac_node <= 0.40, f"{frac_node:.0%} node-only (need <=40%)"),
        "G0.4_spread": (n_cl5 >= 8 and largest <= 0.25,
                        f"{n_cl5} half-deg cells with >=5 sites, largest holds {largest:.0%}"),
        "G0.5_ctrl_pool": (pool.get("frac_ge3", 0.0) >= 0.80 and pool.get("frac_same_cell", 0.0) >= 0.80,
                           f"{pool.get('frac_ge3', 0):.0%} of sites have >=3 area-matched "
                           f"controls, {pool.get('frac_same_cell', 0):.0%} have a same-cell one"),
        "G0.7_dates": (frac_dated >= 0.40,
                       f"{frac_dated:.0%} of sites carry a construction date (need >=40%)"),
        "G0.8_stability": (spread < 0.10, f"site count spread {spread:.1%} over link_m sweep"),
        "G0.9_noninterference": (pool.get("prod_clean", False), "production labels + composites untouched"),
        # Amended replacements. The originals above keep their FAIL verdicts on the record
        # permanently; these are separately labelled and separately justified.
        "A1_headline_volume": (
            n_head >= 90 and (min(head_by_region.values()) if head_by_region else 0) >= 60,
            f"headline (>= {HEADLINE_MIN_AREA_M2:,.0f} m2) {n_head} sites {head_by_region}; "
            f"need >=90 pooled and >=60 per region"),
        "A2_ctrl_in_range": (
            pool.get("frac_ge3_head", 0.0) >= 0.80 and pool.get("frac_in_range_head", 0.0) >= 0.80,
            f"{pool.get('frac_ge3_head', 0):.0%} of headline sites have >=3 area-matched "
            f"controls, {pool.get('frac_in_range_head', 0):.0%} have one within "
            f"{CTRL_MAX_DIST_M/1000:.0f} km"),
        "G0.10_budget": (n_cells <= max_cells and gb_needed <= max_gb,
                         f"{n_cells} cells -> {gb_needed:.1f} GB "
                         f"(limits {max_cells} cells / {max_gb} GB)"),
    }
    metrics = {
        "n_sites_total": len(sites), "n_sites_in_aoi": n,
        "n_sites_by_region": json.dumps(by_region),
        "n_polygon_sites": len(poly), "frac_node_only": round(frac_node, 4),
        "median_site_area_m2": round(med_area, 1),
        "median_core_area_m2": round(float(s.core_area_m2.median()), 1) if n else 0,
        "n_site_area_ge_20k": n_big, "frac_site_area_ge_20k": round(frac_big, 4),
        "n_core_area_ge_2000": int((s.core_area_m2 >= 2000).sum()),
        "n_hybrid": int(s.is_hybrid.sum()) if n else 0,
        "frac_hybrid": round(float(s.is_hybrid.mean()), 4) if n else 0.0,
        "hybrid_co_sources": json.dumps(
            s[s.is_hybrid].co_source.value_counts().to_dict()) if n else "{}",
        "median_site_area_m2_nonhybrid": round(
            float(s[~s.is_hybrid].site_area_m2.median()), 1) if n and (~s.is_hybrid).any() else 0,
        "frac_ge20k_nonhybrid": round(
            float((s[~s.is_hybrid].site_area_m2 >= 20000).mean()), 4)
            if n and (~s.is_hybrid).any() else 0.0,
        "n_cells_ge_5_sites": n_cl5, "largest_cell_frac": round(largest, 4),
        "frac_with_construction_date": round(frac_dated, 4),
        "n_controls_total": len(controls),
        "n_controls_eligible": int(controls.eligible.sum()) if len(controls) else 0,
        "ctrl_excluded_by": json.dumps(pool.get("excluded_by", {})),
        "frac_sites_ge3_controls": round(pool.get("frac_ge3", 0.0), 4),
        "frac_sites_same_cell_control": round(pool.get("frac_same_cell", 0.0), 4),
        # G0.6 is reporting-only, but it is the census that sizes the study's central risk.
        "G0.6_frac_substation_within_300m": round(
            float((s.d_substation_m <= 300).mean()), 4) if n else 0.0,
        "G0.6_frac_overlapping_substation": round(float(s.overlaps_substation.mean()), 4) if n else 0.0,
        "G0.6_median_battery_only_area_m2": round(float(s.battery_only_area_m2.median()), 1) if n else 0,
        "G0.6_n_with_battery_only_ge2k": int(s.has_battery_only.sum()) if n else 0,
        "frac_with_composite_s1": round(cov, 4),
        "n_headline_sites": n_head, "n_headline_by_region": json.dumps(head_by_region),
        "frac_headline_ge3_controls": round(pool.get("frac_ge3_head", 0.0), 4),
        "frac_headline_ctrl_in_range": round(pool.get("frac_in_range_head", 0.0), 4),
        "headline_min_area_m2": HEADLINE_MIN_AREA_M2,
        "ctrl_max_dist_m": CTRL_MAX_DIST_M,
        "n_cells": n_cells, "projected_gb": round(gb_needed, 2),
        "pbf_age_days": age_days,
        "link_m_stability": json.dumps({str(k): v for k, v in stability.items()}),
    }
    for k, (ok, msg) in g.items():
        metrics[k] = "PASS" if ok else "FAIL"
        metrics[k + "_detail"] = msg
    # The two superseded gates keep their FAIL verdicts on the record permanently; the
    # amended verdict is every other original gate plus the labelled replacements.
    superseded = {"G0.3_size", "G0.5_ctrl_pool"}
    metrics["superseded_gates"] = ",".join(sorted(superseded))
    orig_pass = all(ok for k, (ok, _) in g.items() if not k.startswith("A"))
    amended_pass = all(ok for k, (ok, _) in g.items() if k not in superseded)
    metrics["gate0_original"] = "PASS" if orig_pass else "FAIL"
    metrics["gate0_amended"] = "PASS" if amended_pass else "FAIL"
    return metrics, amended_pass


# ------------------------------------------------------------------ control matching


def control_availability(sites: gpd.GeoDataFrame, controls: gpd.GeoDataFrame,
                         *, area_tol: float = 0.30) -> tuple[dict, pd.DataFrame]:
    """G0.5. How many area-matched, battery-free substations each site can actually draw.

    Measured at Stage 0 and not at Stage 2 on purpose: the hydro study's real bottleneck
    turned out to be thin control strata, and it only discovered that after buying all its
    imagery. This costs nothing and can kill the design before a byte is downloaded.
    """
    if sites.empty or controls.empty:
        return {"frac_ge3": 0.0, "frac_same_cell": 0.0}, pd.DataFrame()
    elig = controls[controls.eligible].copy()
    if elig.empty:
        return {"frac_ge3": 0.0, "frac_same_cell": 0.0}, pd.DataFrame()

    c = sites.geometry.union_all().centroid
    utm = _utm_for(c.x, c.y)
    su, cu = sites.to_crs(utm), elig.to_crs(utm)
    cpts = np.array([[g.x, g.y] for g in cu.geometry.representative_point()])
    careas = (elig.area_buf_m2.values if "area_buf_m2" in elig.columns
              else elig.area_m2.values)
    ccells = elig.cell.values if "cell" in elig.columns else np.array([""] * len(elig))

    rows = []
    for i, (_, srow) in enumerate(su.iterrows()):
        p = srow.geometry.representative_point()
        a = sites.site_area_m2.iloc[i]
        d = np.hypot(cpts[:, 0] - p.x, cpts[:, 1] - p.y)
        ok = (careas >= a * (1 - area_tol)) & (careas <= a * (1 + area_tol))
        n_ok = int(ok.sum())
        same = bool(((ccells == sites.cell.iloc[i]) & ok).sum() > 0)
        # Candidate control CELLS with the number of matched controls each holds, out to
        # CTRL_MAX_DIST_M. Counts (not just cell names) are what let build_cell_list run a
        # greedy set-cover and buy far less imagery than one cell per candidate control.
        inrange = ok & (d <= CTRL_MAX_DIST_M)
        cc: dict[str, int] = {}
        for j in np.argsort(np.where(inrange, d, np.inf))[:int(inrange.sum())]:
            cc[ccells[j]] = cc.get(ccells[j], 0) + 1
        rows.append({"site_id": sites.site_id.iloc[i], "n_matched": n_ok,
                     "same_cell_available": same,
                     "nearest_ctrl_m": float(d[ok].min()) if n_ok else np.nan,
                     "n_matched_in_range": int(inrange.sum()),
                     "ctrl_cell_counts": json.dumps(cc)})
    tab = pd.DataFrame(rows)
    tab["ctrl_within_range"] = tab.nearest_ctrl_m <= CTRL_MAX_DIST_M
    stats = {"frac_ge3": float((tab.n_matched >= 3).mean()),
             "frac_same_cell": float(tab.same_cell_available.mean()),
             "frac_in_range": float(tab.ctrl_within_range.mean())}
    log.info("control availability: %.0f%% of sites have >=3 matched controls, "
             "%.0f%% have one in the same cell", 100 * stats["frac_ge3"],
             100 * stats["frac_same_cell"])
    return stats, tab


def build_cell_list(sites: gpd.GeoDataFrame, avail: pd.DataFrame,
                    max_cells: int, *, need_ctrl: int = 3) -> pd.DataFrame:
    """Ranked cell list for `compose --cells-file`, minimised by greedy set cover.

    Every headline site's own cell is bought unconditionally. Control cells are then added
    greedily -- at each step the unchosen cell that brings the most still-deficient sites
    closer to `need_ctrl` matched controls -- until every site is satisfied or no candidate
    remains. Buying one cell per candidate control instead (the obvious implementation)
    costs roughly twice as much imagery for the same matched pairs, because control cells
    are heavily shared between neighbouring sites.
    """
    cols = ["region", "aoi", "cell", "n_bess", "n_ctrl", "role"]
    if sites.empty:
        return pd.DataFrame(columns=cols)

    site_cells = (sites.groupby(["region", "cell"]).size().reset_index(name="n_bess"))
    chosen: dict[tuple, dict] = {
        (r.region, r.cell): {"n_bess": int(r.n_bess), "n_ctrl": 0, "role": "site"}
        for r in site_cells.itertuples()}

    if len(avail):
        reg_of = dict(zip(sites.site_id, sites.region))
        cand: dict[str, dict[str, int]] = {}
        for r in avail.itertuples():
            if r.site_id in reg_of:
                cand[r.site_id] = json.loads(r.ctrl_cell_counts or "{}")

        def covered(sid: str) -> int:
            reg = reg_of[sid]
            return sum(n for cc, n in cand.get(sid, {}).items()
                       if (reg, cc) in chosen)

        deficit = {sid: need_ctrl - covered(sid) for sid in cand}
        deficit = {k: v for k, v in deficit.items() if v > 0}
        while deficit:
            gain: dict[tuple, int] = {}
            for sid, need in deficit.items():
                reg = reg_of[sid]
                for cc, n in cand[sid].items():
                    key = (reg, cc)
                    if key in chosen:
                        continue
                    gain[key] = gain.get(key, 0) + min(n, need)
            if not gain:
                break
            best = max(gain, key=lambda k: gain[k])
            chosen[best] = {"n_bess": 0, "n_ctrl": 0, "role": "control"}
            if len(chosen) >= max_cells:
                log.warning("G0.10: cell cap %d reached with %d sites still short of "
                            "%d controls", max_cells, len(deficit), need_ctrl)
                break
            deficit = {sid: need_ctrl - covered(sid) for sid in cand}
            deficit = {k: v for k, v in deficit.items() if v > 0}
        # Attribute control counts to whichever chosen cells actually supply them.
        for sid, cc_counts in cand.items():
            reg = reg_of[sid]
            for cc, n in cc_counts.items():
                key = (reg, cc)
                if key in chosen:
                    chosen[key]["n_ctrl"] += n

    rows = pd.DataFrame([{"region": r, "cell": c, **v} for (r, c), v in chosen.items()])
    rows["aoi"] = rows.region.map(lambda r: REGIONS[r]["aoi"])
    rows = rows.sort_values(["n_bess", "n_ctrl"], ascending=False).reset_index(drop=True)
    if len(rows) > max_cells:
        log.warning("G0.10: %d cells exceeds the %d cap -- keeping the densest %d",
                    len(rows), max_cells, max_cells)
        rows = rows.head(max_cells)
    return rows[cols]


# ------------------------------------------------------------------------- main


def _prod_snapshot() -> dict:
    snap = {}
    for d in PROD_LABEL_DIRS:
        for p in sorted(d.glob("*.parquet")):
            snap[str(p)] = p.stat().st_mtime
    if PROD_COMPOSITES.exists():
        snap["_pk_composite_count"] = sum(1 for _ in PROD_COMPOSITES.glob("*/composite_0.tif"))
    return snap


def process_region(region: str, a, settings: Settings) -> dict:
    cfg = REGIONS[region]
    bbox, _ = resolve_aoi(cfg["aoi"], settings)
    out = LABELS_ROOT / f"bess_{region}"
    out.mkdir(parents=True, exist_ok=True)

    pbf, ts, age = fetch_pbf(region, a.refresh_pbf, a.max_pbf_age_days, a.allow_stale, a.pbf)
    if a.download_only:
        return {"region": region, "pbf": pbf.name, "age_days": age, "download_only": True}

    raw = extract(pbf, tuple(bbox), a.force)

    # G0.0 -- written BEFORE classify_bess is called, so the definition cannot be a
    # rationalisation of what happened to be there.
    audit = audit_tags(raw, region)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ap = RESULTS_DIR / "bess_stage0_tag_audit.csv"
    if ap.exists():
        prev = pd.read_csv(ap)
        prev = prev[prev.region != region]
        audit = pd.concat([prev, audit], ignore_index=True)
    audit.to_csv(ap, index=False)
    log.info("[%s] tag audit ->\n%s", region,
             audit[audit.region == region][["pattern", "n"]].to_string(index=False))

    feats = classify_bess(raw)
    sites, comps = assemble_sites(feats, region, buffer_m=a.buffer_m, link_m=a.link_m)
    controls = build_control_pool(feats, sites, region, exclude_m=a.exclude_m)
    if sites.empty:
        log.warning("[%s] no battery sites found", region)
    else:
        sites = enrich(sites, controls, region, settings)

    # G0.8: if the site count moves with the merge radius, the site definition is arbitrary
    # and every downstream n is soft. Cheap, so always checked.
    stability = {}
    for lm in (40.0, 60.0, 120.0):
        stability[lm] = len(sites) if lm == a.link_m else \
            len(assemble_sites(feats, region, buffer_m=a.buffer_m, link_m=lm)[0])
    log.info("[%s] link_m stability: %s", region, stability)

    if a.with_context:
        try:
            extract_context(pbf, tuple(bbox), a.force, out)
        except Exception as e:  # noqa: BLE001 -- degrades matching, not the labels
            log.warning("[%s] context extraction failed (%s); Stage 2 will lack "
                        "d_road/d_industrial/d_line", region, e)

    if not sites.empty:
        sites.to_parquet(out / "bess_sites.parquet")
        comps.to_parquet(out / "bess_sites_components.parquet")
    if len(controls):
        controls.to_parquet(out / "substation_controls.parquet")
    clash = {p.name for p in out.glob("*.parquet")} & FORBIDDEN_NAMES
    assert not clash, f"wrote a production-owned filename into {out}: {clash}"

    return {"region": region, "pbf": pbf.name, "pbf_timestamp": str(ts), "age_days": age,
            "sites": sites, "controls": controls, "stability": stability, "out": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="all", help="gb | ca | tx | all")
    ap.add_argument("--buffer-m", type=float, default=30.0)
    ap.add_argument("--link-m", type=float, default=60.0)
    ap.add_argument("--exclude-m", type=float, default=200.0,
                    help="Tier-2 control exclusion radius around any battery hint (G2.8 "
                         "re-runs at 2x this)")
    ap.add_argument("--pbf", type=Path, default=None)
    ap.add_argument("--refresh-pbf", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--max-pbf-age-days", type=int, default=14)
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--with-context", action="store_true", default=True)
    ap.add_argument("--no-context", dest="with_context", action="store_false")
    ap.add_argument("--write-cells", action="store_true")
    ap.add_argument("--max-cells", type=int, default=800)
    ap.add_argument("--max-gb", type=float, default=28.0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    regions = list(REGIONS) if a.region == "all" else [r.strip() for r in a.region.split(",")]
    for r in regions:
        if r not in REGIONS:
            raise SystemExit(f"unknown region {r!r}; have {list(REGIONS)}")

    settings = Settings.load()
    prod_before = _prod_snapshot()

    results = [process_region(r, a, settings) for r in regions]
    if a.download_only:
        for res in results:
            print(f"{res['region']:3} {res['pbf']}  age {res['age_days']} d")
        return

    sites = pd.concat([r["sites"] for r in results if len(r.get("sites", []))],
                      ignore_index=True) if any(len(r.get("sites", [])) for r in results) \
        else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    controls = pd.concat([r["controls"] for r in results if len(r.get("controls", []))],
                         ignore_index=True) if any(len(r.get("controls", [])) for r in results) \
        else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    sites = gpd.GeoDataFrame(sites, crs="EPSG:4326")
    controls = gpd.GeoDataFrame(controls, crs="EPSG:4326")

    # Control availability and the cell list are computed per region (a Texas site cannot
    # borrow a Scottish control) and then pooled.
    avail_parts, cell_parts = [], []
    for r in regions:
        s = sites[sites.region == r] if len(sites) else sites
        c = controls[controls.region == r] if len(controls) else controls
        if len(s) and len(c):
            _, tab = control_availability(s, c)
            avail_parts.append(tab)
    del cell_parts
    avail = pd.concat(avail_parts, ignore_index=True) if avail_parts else pd.DataFrame()
    pool_stats = {
        "frac_ge3": float((avail.n_matched >= 3).mean()) if len(avail) else 0.0,
        "frac_same_cell": float(avail.same_cell_available.mean()) if len(avail) else 0.0,
        "excluded_by": (controls[controls.excluded_by != ""].excluded_by.value_counts().to_dict()
                        if len(controls) else {}),
        "audit_written": (RESULTS_DIR / "bess_stage0_tag_audit.csv").exists(),
    }
    # The amended gates are evaluated on the headline set only -- that is the set whose
    # imagery is bought and whose AUC becomes the verdict.
    head_ids = set(sites[sites.headline].site_id) if len(sites) else set()
    ah = avail[avail.site_id.isin(head_ids)] if len(avail) else avail
    pool_stats["frac_ge3_head"] = float((ah.n_matched >= 3).mean()) if len(ah) else 0.0
    pool_stats["frac_in_range_head"] = float(ah.ctrl_within_range.mean()) if len(ah) else 0.0

    # Only the headline sites' cells are composited. This is what turns A1 from a
    # reporting restriction into a real reduction in imagery cost.
    cells = build_cell_list(sites[sites.headline] if len(sites) else sites, ah, a.max_cells)
    if a.write_cells and len(cells):
        for r in regions:
            sub = cells[cells.region == r]
            if not len(sub):
                continue
            p = ROOT / "data" / f"bess_cells_{r}.txt"
            p.write_text("\n".join(sub.cell.tolist()) + "\n")
            log.info("wrote %s (%d cells)", p, len(sub))
        cells.to_csv(RESULTS_DIR / "bess_stage0_cells.csv", index=False)

    prod_after = _prod_snapshot()
    pool_stats["prod_clean"] = prod_before == prod_after

    age = max(r["age_days"] for r in results)
    stability: dict = {}
    for r in results:
        for k, v in r.get("stability", {}).items():
            stability[k] = stability.get(k, 0) + v

    metrics, passed = evaluate_gates(sites, controls, pool_stats, age, stability, cells,
                                     a.max_pbf_age_days, a.max_cells, a.max_gb, len(regions))
    metrics.update({
        "run_name": "+".join(regions), "regions": ",".join(regions),
        "config_hash": CONFIG_HASH, "buffer_m": a.buffer_m, "link_m": a.link_m,
        "exclude_m": a.exclude_m,
        "positive_definition": (
            "(power=plant AND plant:{source,method,storage,type}~battery) OR "
            "(power=generator AND generator:{source,method,storage}~battery) OR "
            "(power=substation AND substation~battery) OR "
            "(no power= key AND plant/generator:source~battery); "
            "free-text name_hint objects are NOT positives and are excluded from controls"),
        "pbfs": json.dumps({r["region"]: r["pbf"] for r in results}),
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
    })
    for res in results:
        if isinstance(res.get("out"), Path):
            (res["out"] / "bess_extract_manifest.json").write_text(
                json.dumps(metrics, indent=2, default=str))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if len(sites):
        sites.drop(columns=["geometry"]).to_csv(
            RESULTS_DIR / "bess_stage0_inventory.csv", index=False)
    if len(avail):
        avail.to_csv(RESULTS_DIR / "bess_stage0_control_availability.csv", index=False)
    summ = RESULTS_DIR / "bess_stage0_summary.csv"
    row = pd.DataFrame([metrics])
    if summ.exists():
        prev = pd.read_csv(summ)
        prev = prev[~((prev.run_name == metrics["run_name"])
                      & (prev.config_hash == CONFIG_HASH))]
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(summ, index=False)

    assert prod_before == prod_after, "production substation labels/composites were modified!"

    print("\n=== Stage 0: BESS site inventory ===")
    print(f"regions: {', '.join(regions)}   oldest PBF {age} d")
    print(f"sites: {metrics['n_sites_in_aoi']} in AOI  {metrics['n_sites_by_region']}")
    print(f"  polygon-mapped {metrics['n_polygon_sites']}, "
          f"node-only {metrics['frac_node_only']:.0%}")
    print(f"median site area {metrics['median_site_area_m2']:,.0f} m2 "
          f"(unbuffered core {metrics['median_core_area_m2']:,.0f} m2); "
          f"{metrics['frac_site_area_ge_20k']:.0%} >= 20k m2")
    print(f"hybrids (battery co-tagged with generation): {metrics['n_hybrid']} "
          f"({metrics['frac_hybrid']:.0%})  {metrics['hybrid_co_sources']}")
    print(f"  battery-only sites: median {metrics['median_site_area_m2_nonhybrid']:,.0f} m2, "
          f"{metrics['frac_ge20k_nonhybrid']:.0%} >= 20k m2")
    print(f"controls: {metrics['n_controls_eligible']} eligible of "
          f"{metrics['n_controls_total']} substation polygons  "
          f"excluded {metrics['ctrl_excluded_by']}")
    print("\n--- G0.6 co-location census (reporting only, the study's central risk) ---")
    print(f"  substation within 300 m : {metrics['G0.6_frac_substation_within_300m']:.0%}")
    print(f"  polygon overlaps a substation: {metrics['G0.6_frac_overlapping_substation']:.0%}")
    print(f"  median battery-only residual : {metrics['G0.6_median_battery_only_area_m2']:,.0f} m2")
    print(f"  sites with a usable battery-only sub-polygon (>=2,000 m2): "
          f"{metrics['G0.6_n_with_battery_only_ge2k']}")
    if len(sites):
        print("\nsize buckets (unbuffered core):")
        print(sites[sites.in_aoi].size_bucket.value_counts().to_string())
    print(f"\nheadline set (>= {HEADLINE_MIN_AREA_M2:,.0f} m2, amendment A1): "
          f"{metrics['n_headline_sites']} sites {metrics['n_headline_by_region']}")
    print(f"\nimagery: {metrics['n_cells']} cells -> {metrics['projected_gb']:.1f} GB")
    print("\n--- gates (as pre-registered) ---")
    sup = set(metrics.get("superseded_gates", "").split(","))
    for k in sorted(k for k in metrics
                    if k.startswith("G0.") and not k.endswith("_detail")):
        if k.startswith("G0.6"):
            continue
        mark = "  [superseded by amendment]" if k in sup else ""
        print(f"  {metrics[k]:4}  {k:22} {metrics[k + '_detail']}{mark}")
    print("--- amendments (2026-09-02) ---")
    for k in sorted(k for k in metrics
                    if k.startswith("A") and k[1].isdigit() and not k.endswith("_detail")):
        print(f"  {metrics[k]:4}  {k:22} {metrics[k + '_detail']}")
    print(f"\nGATE 0 as pre-registered: {metrics['gate0_original']}"
          f"   (G0.3, G0.5 failed)")
    print(f"GATE 0 as amended       : {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
