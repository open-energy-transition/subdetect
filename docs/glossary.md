---
title: Glossary
---

# Glossary

Terms used throughout this documentation, in the order you're likely to meet them.

## Data sources

**OSM** — OpenStreetMap, the crowd-sourced map database that supplies both the
training labels (mapped substations, power lines) and the search signal (Osmose,
below) this project uses.

**Geofabrik** — a third-party service that republishes OSM data as downloadable
regional `.osm.pbf` extracts. `subdetect osm --aoi <aoi>` downloads one of these and
extracts `power=substation`/`power=plant`/`power=line` features to GeoParquet.

**Overpass (API)** — a live query API over OSM data, used instead of a Geofabrik
extract when the project needs an up-to-the-minute answer for an arbitrary bounding
box (e.g. "which substations exist near this Osmose endpoint right now?") rather
than a static regional download.

**Osmose** — an OSM data-quality-assurance engine ([osmose.openstreetmap.fr](https://osmose.openstreetmap.fr))
that flags likely mapping errors. This project uses one specific issue type: **item
7040, class 2**, "unfinished major power line" — a transmission line digitized in
OSM that dead-ends without reaching a substation. Since power lines don't really
end in a field, a flagged endpoint far from any *mapped* substation is a strong
prior that a *real, unmapped* substation exists nearby. This is the seed for the
whole `scripts/osmose_detect.py` workflow — see [Osmose-guided detection](osmose-detect.md).

**MapRoulette** — a crowdsourced task platform for fixing OSM data. The `export`
CLI stage can package detected candidates as a MapRoulette challenge, so a human
mapper reviews each candidate against imagery before adding it to OSM.

## Imagery

**Sentinel-2 (S2) / S2 L2A** — ESA's optical satellite, 10 m/pixel at visible/NIR
bands. "L2A" means atmospherically corrected (surface reflectance, not raw
top-of-atmosphere). This project uses 10 bands (`LOCAL_BANDS` in `config.py`:
B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12) at native 10 m.

**dry-season composite** — rather than a single S2 scene (which may be cloudy),
the pipeline takes the per-pixel median of every available scene in a fixed
seasonal date window (`s2_window` in `configs/aoi.yaml`), after masking clouds via
the scene classification layer. This gives one clean, mostly cloud-free image per
cell even in regions with persistent cloud cover.

**Sentinel-1 (S1) / RTC / VV / VH** — ESA's radar (SAR) satellite. "RTC"
(radiometrically terrain-corrected) means the backscatter values are already
corrected for terrain-slope effects. VV and VH are the two polarizations
(transmit-vertical/receive-vertical vs. transmit-vertical/receive-horizontal) —
different polarizations react differently to different surface types, which is
exactly what makes S1 useful here (see **corner reflector** below).

**dB** — decibels; S1 backscatter power is stored/interpreted on a log (dB) scale.
The composite files hold this as an encoded uint16 (`S1_SCALE=500`,
`S1_OFFSET_DB=50` in `config.py`: `dB = DN / 500 - 50`), not raw dB, purely to fit
losslessly into a compact GeoTIFF.

**corner reflector** — a geometric arrangement of two or three roughly
perpendicular metal surfaces (exactly what substation gantries, transformers, and
busbars look like) that reflects radar very efficiently straight back to the
sensor, making it appear unusually bright in SAR imagery — especially in the
cross-polarized VH channel. This is *why* radar is useful here: bare/exposed
natural land (the model's main false-positive class in optical-only imagery)
looks nothing like a corner reflector, so radar separates them. Measured in this
project: VH backscatter AUC 0.89 between real substations and human-reviewed
false positives (`scripts/s1_separability.py`).

**STAC** — SpatioTemporal Asset Catalog, the API standard both Microsoft Planetary
Computer and AWS Earth Search expose their Sentinel-1/2 archives through. The
pipeline queries these live rather than storing a local imagery archive.

## Geometry & grid

**AOI** — Area Of Interest: a named region with a bounding box and label source,
defined in `configs/aoi.yaml` (e.g. `pakistan`, `india_pilot`). Distinct from a
Osmose **region** (see below), which is an ad-hoc search area with no pre-existing
labels.

**ROI** — Region Of Interest: *within* an AOI, the actual set of grid cells worth
compositing/running inference on — every 0.1° cell within a radius of a power
line or substation, rather than the whole AOI bounding box. Keeps compute bounded
without hurting recall (substations are never far from the grid).

**grid_origin / cell** — the pipeline tiles the world into a fixed 0.1°×0.1°
lattice, snapped to a shared origin point (`grid_origin` in `aoi.yaml`) so that
cell names (`f"{ix:04d}_{iy:04d}"`) are identical across AOIs. This is what makes
Pakistan cell reuse from the sibling `earthpv` project possible byte-for-byte —
see [Architecture](architecture.md#the-earthpv-relationship).

**chip** — a small, fixed-size (224×224 px, 2.24 km) training crop cut from a
cell's composite, centered (with jitter) on a labeled substation, a point along a
power line, or a random background location. Chips are what the model actually
trains on, not full cells.

## Model & training

**TerraMind** — IBM/ESA's pretrained multimodal geospatial foundation model. It
has separate pretrained patch-embeddings per modality (S2, S1, etc.) feeding a
shared vision-transformer encoder, so it already "knows" useful visual structure
before this project's fine-tuning ever starts.

**TerraTorch** — the training framework (built on PyTorch Lightning) used to
fine-tune TerraMind for this project's segmentation task via config-driven
`SemanticSegmentationTask`s (`configs/terramind_sub_*.yaml`).

**hard-negative mining** — after running the current model over new imagery,
manually or heuristically identifying its confident *false* positives and adding
them as explicit background training examples, so the model is directly taught
its own specific mistakes (typically bare/exposed land it currently confuses for
a substation's gravel yard).

**focal Tversky loss** — a loss function that lets you independently weight false
positives vs. false negatives (via `alpha`/`beta`) and control how much the
hardest examples dominate training (via `gamma`) — used here instead of plain
cross-entropy because substation pixels are a tiny minority of every chip.

**IoU (Intersection over Union) / F1** — standard pixel-level segmentation
metrics: how well the predicted substation mask overlaps the true mask. Reported
per-pixel across a validation set.

**recall (per-installation)** — a different, arguably more useful metric this
project also tracks: out of N real substations, how many are found by *at least
one* correctly predicted pixel anywhere inside their footprint? A model can have
mediocre pixel IoU but excellent per-installation recall (it finds every
substation, just doesn't trace its exact outline perfectly) — see
`evaluate.py`'s `AREA_BUCKETS`/`VOLT_BUCKETS` recall tables.

**modality drop(out)** — during training, randomly hiding one whole input
modality (e.g. showing only S2, no S1) for some fraction of examples, so a
dual-modality model still works reasonably at inference time if one modality is
missing or degraded for a given cell.

## Post-processing & ranking

**hysteresis (polygonization)** — instead of a single confidence threshold,
seed candidate regions only where probability exceeds a high threshold (`--hi`,
default 0.4), then grow each seed through any adjacent pixels above a lower
threshold (`--lo`, default 0.2). This merges a real substation's fragmented
high-confidence core with its lower-confidence edge into one clean polygon,
without letting isolated single-pixel noise seed a spurious detection on its own.

**confidence vs. rank_score** — `confidence` is a property of one detected
polygon alone (its peak predicted probability). `rank_score` additionally folds
in *where* that polygon sits — how close to an Osmose endpoint, how close to a
known power line — so the review order reflects the project's actual goal
(explaining unfinished lines), not just raw model confidence. Sort by
`confidence` instead if you want a general "any unmapped substation" review order.

**below_floor** — a candidate whose polygon area falls under the training area
floor (`min_sub_area_m2`, historically 20,000 m²). Flagged and sunk to the bottom
of the review order rather than dropped outright — small substations are real and
sometimes detected, just lower-confidence on average.

**known / new** — during AOI-based (non-Osmose) postprocessing, whether a
detected candidate spatially coincides with an already-OSM-mapped substation
(`known`) or not (`new` — the interesting ones, worth reviewing for mapping).
