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

**Result (single-variable experiment, 2026-07-11):** `v4_s1fusion` vs
`v4_s2only`, both fresh-initialized on the identical chip index. Pakistan val:
fusion IoU 0.266 / **63% installation recall (≥20k m²)** vs control IoU 0.243 /
32% — recall doubled, with the biggest gain on small/unknown-voltage
installations (60% vs 10%). Fusion (fresh init) also beats the warm-started
production v3b (IoU 0.237 / 45%). Caveats: 19-installation val set; 125/448
hard-negative chips lacked S1 composites and were dropped from both arms.

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
