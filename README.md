# subdetect — power substation detection from Sentinel-2/-1

⚠️ This is a prototype that is not intended for production or collaboration purposes. If you would like to use this project, please contact the main developer. ⚠️

Detects transmission-class power substations in Sentinel-2 L2A
dry-season composites by fine-tuning the **TerraMind** geospatial foundation
model (IBM/ESA) with **TerraTorch**. Labels come from OpenStreetMap power
tags (Geofabrik PBF, no Overpass). Recall-oriented: candidates are ranked for
human validation against imagery (MapRoulette export included) — raw candidate
lists are NOT trustworthy without review.

- **Training regions:** Pakistan + NW-India pilot (Indian Punjab/Haryana/Delhi), plus mined hard negatives
- **Inference targets:** Pakistan (1,564 cells), India pilot (474 cells)
- **Imagery:** cloud-masked dry-season medians (2025-11 → 2026-03), 10 S2 L2A bands @ 10 m,
  optional co-registered Sentinel-1 RTC VV/VH, from Microsoft Planetary Computer

## Setup

```bash
pixi install          # data pipeline env
pixi install -e ml    # + PyTorch cu126 (Pascal-safe) + TerraTorch
pixi run -e ml gpu-check
```

## Pipeline

```bash
pixi run osm      --aoi pakistan          # Geofabrik PBF -> power lines/substations
pixi run roi      --aoi pakistan          # 0.1° cells within 20 km of grid infrastructure
pixi run compose  --aoi pakistan          # S2 dry-season composites (resumable); --sensor s1 for VV/VH
pixi run chips    --aoi pakistan          # training chips + burned masks; --s1 for dual-modality
pixi run -e ml train    --config configs/terramind_sub_v3b_hardneg_half.yaml
pixi run -e ml evaluate --aoi pakistan --checkpoint <ckpt>   # pixel IoU/F1 + per-installation recall
pixi run -e ml infer    --aoi pakistan --checkpoint <ckpt> --out-dir data/predictions_v3b
pixi run postprocess --aoi pakistan --pred-dir data/predictions_v3b   # polygonize + rank (grid prior)
pixi run export      --aoi pakistan --pred-dir data/predictions_v3b   # GeoJSON / MapRoulette
```

All long steps skip existing outputs, so they can be killed and re-run safely.

## Model lineage (Pakistan val: pixel IoU / ≥20k m² recall)

| model | change | notes |
|---|---|---|
| v1 | 1k m² label floor | underfit (train IoU ≤ 0.11) |
| v2 | 20k m² floor, focal Tversky | 0.30 / 60% |
| v2_india | + India chips | best deep recall; new-lead precision ~0.4% (bare-land FPs) |
| v3 / v3b | + mined hard negatives (1×/0.5×) | 8–15× fewer candidates, best top-100 triage yield (v3b) |
| v4_s2only / v4_s1fusion | fresh-init S1 fusion experiment | see `configs/terramind_sub_v4_*.yaml` |

The released Lindsay-Lab SWIN model (sibling repo `../substation-seg`) serves as a
zero-shot second opinion; agreement filtering and OSM line-endpoint topology
(AUC 0.95 vs reviewed FPs) concentrate the review lists.

## Sentinel-1 + Sentinel-2 fusion

**Why:** the dominant false-positive class is bare land — spectrally similar to a
substation's gravel yard in single-date optical imagery. Radar separates them:
transformers, gantries and busbars are corner reflectors (bright, especially in
cross-pol VH), smooth bare soil scatters forward (dark). Measured on 150 known
substations vs 150 human-reviewed bare-land FPs: **VH AUC 0.89** (substation
median −11.4 dB vs −15.8 dB; `scripts/s1_separability.py`,
`data/s1_separability_samples.csv`). Limit: large metal-roofed industrial
buildings are also SAR-bright — S1 does not resolve that (smaller) FP class.

**Data:** `compose --sensor s1` builds a dry-season Sentinel-1 RTC VV/VH
composite per cell (`composite_s1.tif`), median in *linear power* (speckle-robust)
then dB-encoded to uint16, pinned to the exact GeoBox of the cell's S2
composite — pixel-aligned by construction. `chips --s1` (and
`scripts/s1_for_hardneg.py` for the mined negatives) cut co-registered
2-band S1 chips; the chip index gains an `s1` path column. At load time the
datamodule decodes DN → dB → z-score and returns
`{"image": {"S2L2A": t, "S1RTC": t}, "mask": m}`.

**Model:** token-level mid-fusion inside TerraMind. Each modality has its own
*pretrained* patch-embed (no SAR representation learned from scratch); both token
sequences pass through the shared ViT encoder, so self-attention fuses S2 and S1
patches; tokens at the same position are merged by mean before the (unchanged)
neck + UNet decoder. `backbone_modality_drop_rate: 0.1` randomly drops a whole
modality in training, so inference degrades gracefully where S1 is missing.
Attention memory grows ~4× with the doubled sequence → batch 4 + grad-accum 4
on the 6 GB GPU. `evaluate` and `infer` auto-detect dual-modality checkpoints
from hparams and feed `composite_s1.tif` windows automatically.

**Result (three-arm ablation, fresh init, identical chips/recipe, 2026-07-11):**

| arm (Pakistan val, 19 installs) | pixel IoU | F1 | ≥20k m² recall | ≥220 kV |
|---|---|---|---|---|
| `v4_s1only` (VV/VH only) | **0.310** | **0.473** | **84%** | **100%** |
| `v4_s1fusion` (S2+S1) | 0.266 | 0.420 | 63% | 71% |
| `v4_s2only` (control) | 0.243 | 0.391 | 32% | 71% |

Radar alone is the strongest single signal — corner-reflector texture identifies
switchyards more reliably than optical spectra, and S1-only posts the best val
numbers of any model in the project (prior best: v2_india IoU 0.274 / 60%).
The naive mean-merge fusion *dilutes* rather than combines the signals; if fusion
is revisited, try concat-merge or longer training. Note the FP profiles differ:
S2 models fail on bare land, S1 models will fail on other radar-bright metal
structures (industry, rail). Caveats: 19-installation val set, single seed;
125/448 hard-negative chips lacked S1 composites and were dropped from all arms.

## Osmose-guided regional detection (`scripts/osmose_detect.py`)

End-to-end workflow to find *missing* substations anywhere, without needing the
AOI/labels machinery: OpenStreetMap's Osmose QA engine already flags transmission
lines that end nowhere ("unfinished major power line", item 7040 class 2) — a line
must terminate at a substation, so each flagged endpoint far from any mapped
substation marks a probable unmapped one. Line-endpoint topology was the strongest
FP discriminator we measured (AUC 0.95), so these leads are high-prior by construction.

```bash
pixi run -e ml python scripts/osmose_detect.py --region punjab_in --country india_punjab
# options: --bbox lon1,lat1,lon2,lat2   --sub-dist-m 700   --search-km 10
#          --threshold 0.3   --workers 4   --limit-cells N   --dry-run
```

Steps (all resumable; `--dry-run` stops after the cell plan and cost estimate):

1. **Fetch Osmose issues** for the country/state code (see osmose.openstreetmap.fr
   for codes, e.g. `pakistan`, `india_punjab`). Fetched in 2° bbox tiles — the API
   caps at ~500 issues per request.
2. **Filter endpoints**: OSM substations for the region are fetched live from
   Overpass (`power=substation`, any size incl. nodes — no local labels needed);
   endpoints within `--sub-dist-m` (default 700 m) of one are dropped as
   mapping-detail noise. Survivors → `endpoints.geojson`.
3. **Cell plan**: all 0.1° cells within `--search-km` (default 10 km) of a surviving
   endpoint. Typical state: tens to a few hundred cells (~38 MB and ~2 min each).
4. **Compose S2 + S1** dry-season composites for exactly those cells
   (same code paths as the main pipeline; S1 pinned to the S2 grid).
5. **Detect with the established best stack**: tiled Hann-blended inference where
   `P = P_S1only × (0.5 + 0.5 · P_S2only)` — the S1-only detector (best recall,
   84% ≥20k m² on val) softly gated by the optical model (best measured pixel
   config, IoU 0.345; the gate damps radar-bright industrial FPs but cannot veto).
   Checkpoints default to `stageA_v4_s1only` / `stageA_v4_s2only` best epochs.
6. **Post-process**: polygonize at `--threshold` (0.3), drop candidates below the
   20k m² area floor, rank by `confidence × exp(−endpoint_distance / 2 km)` —
   candidates that sit where an unfinished line points are ranked first.

Output under `data/osmose_regions/<region>/`: `endpoints.geojson`,
`composites/<cell>/composite_{0,s1}.tif`, `prob/<cell>.tif`, and **`leads.geojson`**
(review-ready, sorted; columns: `confidence`, `area_m2`, `endpoint_dist_m`,
`n_endpoints_in_radius`, `rank_score`). Every lead should be human-validated
against high-resolution imagery before mapping.

## Beyond the CLI: analysis scripts

- `scripts/rerank_s1.py` — sample S1 VH per candidate (metal prior), adds `rank_score_s1`
- `scripts/osmose_leads.py` — Osmose "unfinished power line" endpoints (>700 m from any mapped substation) cross-referenced with model candidates; dual-evidence leads in `data/osmose/`
- `scripts/mine_hard_negatives.py`, `scripts/s1_for_hardneg.py` — hard-negative chips (+S1)
- `scripts/compose_s1_chip_cells.py` — targeted S1 compositing for chip cells only
- `scripts/s1_separability.py`, `building_fill.py`, `optical_features.py` — FP-discriminator studies

## Hardware note

Pinned to torch cu126 wheels: the local GPU is a GTX 1060 (Pascal, sm_61), which
CUDA 13 wheels no longer support. Dual-modality training halves batch size and
doubles grad accumulation to fit 6 GB.
