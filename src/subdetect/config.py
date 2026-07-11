"""Shared configuration, band constants, and small geo helpers.

Substation detection reuses earthpv's TerraMind band mapping verbatim (the S2 local
composites are the same 10-band uint16 COGs). The Overture/label machinery of earthpv
is dropped — labels here come from OSM PBFs via osm.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pyproj import Geod

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CONFIG_PATH = REPO_ROOT / "configs" / "aoi.yaml"

# Bands present in the S2 composites (10-band uint16 COGs), in file order — identical
# to earthpv / rooftopsenti so hardlinked Pakistan cells are byte-compatible.
LOCAL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# TerraMind band name for each of our LOCAL_BANDS (same order). TerraMind's pretrained
# S2L2A patch-embed is 12-band; TerraTorch subsets it to these 10 at load.
LOCAL_TO_TERRAMIND = {
    "B02": "BLUE", "B03": "GREEN", "B04": "RED", "B05": "RED_EDGE_1",
    "B06": "RED_EDGE_2", "B07": "RED_EDGE_3", "B08": "NIR_BROAD",
    "B8A": "NIR_NARROW", "B11": "SWIR_1", "B12": "SWIR_2",
}
MODEL_BANDS = [LOCAL_TO_TERRAMIND[b] for b in LOCAL_BANDS]  # 10 TerraMind band names

# Alias used by imagery.py's Planetary-Computer path.
S2_BANDS = LOCAL_BANDS

# Sentinel-1 (PC sentinel-1-rtc): VV, VH. TerraMind's untok_sen1rtc@224 pretrain
# stats (dB) — the backbone does NOT standardize internally, so the datamodule must.
S1_BANDS = ["vv", "vh"]
S1_MEAN = [-10.93, -17.329]
S1_STD = [4.391, 4.459]
# S1 composite storage: dB -> uint16 DN = clip((dB + OFFSET) * SCALE, 1, 65535); 0 = nodata.
# dB range roughly [-35, +5] -> DN roughly [7500, 27500], comfortably inside uint16.
S1_OFFSET_DB = 50.0
S1_SCALE = 500.0

# One full year of observations (unused for substations but kept for API parity).
SEASONS = {
    "spring": ("2025-03-01", "2025-05-31"),
    "summer": ("2025-06-01", "2025-08-31"),
    "autumn": ("2025-09-01", "2025-11-30"),
    "winter": ("2025-12-01", "2026-02-28"),
}

CHIP_SIZE = 224  # pixels @ 10 m -> 2.24 km, 14x14 ViT patches
CHIP_RES = 10.0  # metres

# Substation label thresholds (defaults; overridden by configs/aoi.yaml).
MIN_SUB_AREA = 1000.0        # m2: polygons >= this -> class 1, smaller -> ignore
NODE_IGNORE_RADIUS_M = 60.0  # ignore disc radius around a substation node

_GEOD = Geod(ellps="WGS84")


def geodesic_area_m2(geom) -> float:
    """Unsigned geodesic area — CRS-free, works globally."""
    if geom is None or geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
        return 0.0
    area, _ = _GEOD.geometry_area_perimeter(geom)
    return abs(area)


@dataclass
class Settings:
    """Runtime settings loaded from configs/aoi.yaml with sane defaults."""

    aois: dict = field(default_factory=dict)
    min_sub_area_m2: float = MIN_SUB_AREA
    node_ignore_radius_m: float = NODE_IGNORE_RADIUS_M
    roi_radius_km: float = 20.0
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or CONFIG_PATH
        raw = yaml.safe_load(path.read_text()) if path.exists() else {}
        return cls(
            aois=raw.get("aois", {}),
            min_sub_area_m2=raw.get("min_sub_area_m2", MIN_SUB_AREA),
            node_ignore_radius_m=raw.get("node_ignore_radius_m", NODE_IGNORE_RADIUS_M),
            roi_radius_km=raw.get("roi_radius_km", 20.0),
            raw=raw,
        )


def resolve_aoi(aoi: str, settings: Settings) -> tuple[tuple[float, float, float, float], dict]:
    cfg = settings.aois.get(aoi)
    if cfg is None:
        raise KeyError(f"AOI '{aoi}' not in configs/aoi.yaml (have: {list(settings.aois)})")
    return tuple(cfg["bbox"]), cfg
