---
title: Architecture
---

# Architecture

![Pipeline architecture: core CLI plus the Osmose regional side-path](assets/architecture.svg)

subdetect has two entry points into the same underlying machinery: a **core CLI**
(`src/subdetect/cli.py`) for the standard train/evaluate/infer-on-a-known-AOI
workflow, and a **standalone script** (`scripts/osmose_detect.py`) that runs the
established best model over an ad-hoc search area seeded by OpenStreetMap's own
QA engine, with no pre-existing labels required. Both are described in full below.

## The core pipeline, stage by stage

Each stage is a `subdetect` CLI command (`pixi run -e ml subdetect <stage> ...`);
see the [README quickstart](../README.md#quickstart) for copy-pasteable commands.

1. **`osm`** — download a Geofabrik `.osm.pbf` extract for an AOI and pull out
   `power=substation` (polygons and bare nodes), `power=plant`, and `power=line`
   features into GeoParquet under `data/labels/<aoi>/`. This is the one-time label
   ingestion step; everything downstream reads from these files, not from OSM
   directly. See `src/subdetect/osm.py`.
2. **`roi`** — decide *which* 0.1° grid cells are worth compositing: every cell
   within `--radius-km` of a power line or substation, plus any cell containing a
   training-positive substation polygon (so positives are never excluded even if
   geometrically distant from the general corridor buffer). Writes a diagnostic
   PNG (`data/roi/<aoi>_roi.png`) so you can sanity-check coverage before spending
   network time compositing it. See `src/subdetect/roi.py`.
3. **`compose`** — for each ROI cell, query Sentinel-2 (`--sensor s2`, the
   default) or Sentinel-1 (`--sensor s1`) STAC catalogs (Planetary Computer,
   falling back to Earth Search on outage) and write a cloud-masked dry-season
   median composite. S1 is always pinned to its matching cell's S2 grid so the
   two modalities are pixel-co-registered by construction. Resumable — cells with
   an existing composite file are skipped. See `src/subdetect/imagery.py`,
   `src/subdetect/compose.py`.
4. **`chips`** — cut fixed-size (224×224 px) training crops: one per labeled
   substation polygon (jittered off-center to avoid teaching the model a
   center-bias), some along power lines far from any substation (hard negatives
   matching the inference domain), and some uniform-random background. Burns a
   per-chip mask (1 = substation ≥ the area floor, -1 = ignore for sub-floor/node/
   plant features, 0 = background). `--min-area-m2` overrides the training area
   floor for this call only (0 = every substation polygon trains as class 1 — see
   [Model lineage](model-lineage.md) for why this matters). See
   `src/subdetect/chips.py`.
5. **`train`** — fine-tune TerraMind via TerraTorch against a config
   (`configs/terramind_sub_*.yaml` — one per model-lineage generation). `--smoke`
   runs 50 steps only, for catching config/shape errors before committing GPU time.
   See `src/subdetect/train.py`.
6. **`evaluate`** — pixel IoU/F1 plus **per-installation recall**, bucketed by
   substation area and OSM voltage class — the metric that actually matters for
   "did we find the substation," not just "did we trace its exact outline."
   `--min-area-m2` overrides which installations count as ground truth here too
   (the settings floor otherwise silently excludes sub-floor installations from
   every bucket, not just training). See `src/subdetect/evaluate.py`.
7. **`infer`** — tiled, Hann-window-blended inference over an AOI's composited
   cells, writing one probability GeoTIFF per cell. See `src/subdetect/infer.py`.
8. **`postprocess`** — threshold/polygonize the probability rasters, add a
   grid-proximity prior (distance to nearest power line) and a known/new flag
   (does this candidate coincide with an already-mapped substation?). See
   `src/subdetect/postprocess.py`.
9. **`export`** — write final candidates as GeoParquet/GeoJSON, and package new
   (unmapped) candidates as a MapRoulette challenge for human review. See
   `src/subdetect/export.py`.

## The earthpv relationship

subdetect is forked from a sibling project, **earthpv** (rooftop solar PV
detection), and depends on it in a way that isn't obvious from using the CLI
alone:

- **Shared band/grid conventions.** `LOCAL_BANDS`/`LOCAL_TO_TERRAMIND` in
  `src/subdetect/config.py` are copied verbatim from earthpv, so composites are
  byte-compatible between the two projects.
- **Shared grid lattice.** Pakistan's `grid_origin` in `configs/aoi.yaml` is
  copied verbatim from earthpv's own `configs/aoi.yaml`, specifically so
  0.1° cell names (`ix_iy`) refer to the *same physical cell* in both projects.
- **Hardlinked imagery.** `scripts/link_pakistan_composites.py` hardlinks
  earthpv's already-composited Pakistan Sentinel-2 cells straight into
  subdetect's `data/composites/pakistan/` tree — zero bytes re-downloaded, zero
  STAC queries repeated, because the grid alignment above guarantees they're the
  literal same cells.

Practical upshot: if you're setting up subdetect fresh without access to
earthpv's already-composited data, Pakistan `compose` runs will need to query S2
STAC catalogs for cells earthpv already has cached — expect it to take
meaningfully longer than described anywhere that assumes the hardlink shortcut.

## Osmose-guided regional lead generation

The bottom swimlane in the diagram above is `scripts/osmose_detect.py` — a
different way of pointing the same trained model at the world. Instead of
requiring a curated AOI with existing labels, it lets OpenStreetMap's own data
quality tooling nominate search areas: see [Osmose-guided detection](osmose-detect.md)
for the full walkthrough, output schema, and real Yunnan/Sindh results.

## Analysis scripts (`scripts/`)

Beyond the two entry points above, `scripts/` holds standalone experiments and
diagnostics — not wired into the CLI, run directly with `pixi run python
scripts/<name>.py`:

| script | purpose |
|---|---|
| `report_cells.py` | Pre-flight AOI stats (substation/label/ROI counts) to tune ROI radius and val split before a network-bound `compose` run. |
| `link_pakistan_composites.py` | Hardlinks earthpv's already-composited Pakistan cells (see above). |
| `compose_s1_chip_cells.py` | Targeted S1 compositing scoped to only the cells containing training chips, prioritized (val → train/hardneg → india_pilot). |
| `compose_contrast_season.py` | Builds a second-season S2 composite to test two-season stacking for substations. |
| `merge_chip_index.py` | Merges per-AOI chip indexes into `data/chips/combined/index.parquet`, with an optional oversampling repeat factor per AOI. |
| `mine_hard_negatives.py` | Mines hard-negative chips from manually-reviewed false-positive candidates. |
| `mine_hard_negatives_yunnan.py` | Mines hard negatives from the *bottom half* of an unreviewed Osmose pilot's `rank_score` — careful to avoid burning genuine unmapped substations as negatives. |
| `mine_hard_negatives_remote.py` | Mines hard negatives from terrain far from any known building (VIDA Open Buildings via DuckDB httpfs) — inference-independent, no prior detection run needed. |
| `s1_for_hardneg.py` | Backfills co-registered S1 chips for the hard-negative set. |
| `s1_separability.py` | Hypothesis test: does S1 VH separate real substations from bare-land false positives? (Feeds the VH AUC 0.89 finding.) |
| `building_fill.py` | Per-candidate OSM building-footprint/industrial-landuse overlap fraction (substations are open-air; factories are buildings). |
| `optical_features.py` | Per-candidate roof-vs-gravel test from local S2 composites (brightness/NDVI/texture). |
| `eval_decision_fusion.py` | Compares S1-only + S2-only decision-level fusion strategies (geometric mean, arithmetic mean, max, S1-gated) without retraining. |
| `eval_polygonize_v2.py` | Compares polygonize/scoring variants against free ground truth (Overpass or local labels); AUC/precision@k. |
| `eval_small_subs.py` | Checks whether a model detects substations below its training area floor. |
| `corridor_recall.py` | Deployment-realistic recall: does any exported candidate land within a match radius of a real substation, split by held-out val bbox vs. whole inferred area. |
| `rerank_s1.py` | Re-ranks exported candidates using sampled S1 VH backscatter as a "metal prior." |
| `osmose_leads.py` | Cross-references Osmose endpoints against an AOI's existing local-label candidate set (rather than live Overpass) for dual-evidence leads. |
| `field_eval.py` | Persisted, chip-index-rebuild-proof scoring (AUC/P@20/P@50/area-bucket recall/FP proxy) against raw OSM ground truth — appends one row per run to `data/eval_results/field_eval.csv`. See [Model lineage](model-lineage.md#fixing-the-evaluation-itself). |
| `build_chips_from_substation_ds.py` | Converts the external TorchGeo `Substation` dataset (26,522 global S2 chips) into this project's chip format. See [Expanding the training data](expanding-training-data.md). |
| `merge_chip_index_v9_s2only.py` | Merges the converted TorchGeo chips with `chips_v5/combined` into `data/chips_v9/combined/index.parquet`. |
| `build_global_hardneg_chips.py` | Mines hard-negative chips from well-mapped global cities/water bodies, Overpass-verified clean of any real substation. See [Expanding the training data](expanding-training-data.md#2-mining-global-hard-negatives-from-well-mapped-regions). |
| `make_docs_assets.py` | Regenerates every figure in this documentation from real project data — see its own docstring for usage. |
