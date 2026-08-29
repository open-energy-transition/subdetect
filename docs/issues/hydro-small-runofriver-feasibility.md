# Small run-of-river hydro in northern Pakistan: is it detectable at 10 m?

> Status: gates pre-registered 2026-08-29, **before** any Stage 1/2/4 number was
> computed. Results are appended below as each stage completes. Same structure as
> `s1-tower-chain-corridor-evidence.md`, but pre-registered rather than retrofitted —
> at n≈159 sites the temptation to move a threshold after seeing the number is the
> single largest risk to this study.

## Problem

The user has mapped many small hydropower sites across Gilgit-Baltistan, KPK, Chitral,
Swat and AJK — the ones where a penstock runs down a slope to a powerhouse. Can the
substation pipeline be pointed at them?

(Terminology: the request said "pumped hydro". What is actually mapped is **run-of-river
micro/mini hydro** — intake weir, headrace, forebay, penstock, powerhouse, tailrace. No
pumped-storage class exists in the northern PK OSM data. The imaging signature is the
same; this study targets what is mapped.)

## Why the naive version cannot work

| fact | number | source |
|---|---|---|
| hydro plant polygons, north PK | 159 (+65 generator nodes) | `pakistan-latest.osm.pbf`, 2026-07-06 |
| **median plant footprint** | **535 m² (~5 S2 pixels)** | geodesic area, n=159 |
| plants ≥ 20,000 m² (the training floor) | 9 | " |
| plants ≥ 2,000 m² | 37 | " |
| **pipeline recall, 1,000–2,000 m² bucket** | **0.000 in all 23 runs, both eval versions** | `data/eval_results/field_eval.csv` |
| model response there | `p_max` **exactly 0.000** (`lost:silent`) | `missed_diagnosis_sindh_test_v12mean_allbuckets.csv` |
| `waterway=penstock` in Pakistan | **0 objects nationwide** | PBF query |
| real tag `usage=penstock`, north PK | 42 lines, median **87 m**, mostly `tunnel=yes` | " |
| hydro features with any penstock mapped | 34% | " |

The powerhouse is *below* a size class where this model produces literally no response,
and the penstock is sub-pixel in width and usually buried.

## Physical signal, and what is actually being labelled

The detectable object is not the powerhouse and not the pipe. It is the **site complex**:
intake weir, headrace, forebay, penstock corridor (a cleared scar down a slope), powerhouse
pad, tailrace, access track, spoil. That is a several-hundred-metre-scale scene inside a
2.24 km chip.

So the positive is the dissolved 75 m buffer of plant ∪ generator ∪ conduit for one
installation — ~30,000 m², landing in the ≥20,000 m² bucket where measured recall is 0.714.
`core_area_m2` preserves the honest unbuffered number alongside it. This detects *"a hydro
site is here"*, which is what a ranked-lead product needs; it does not claim to segment a
powerhouse.

Candidate discriminators, in rough order of expected strength:

- **S1 VH local contrast.** A metal-roofed powerhouse on a flat bench in a steep valley is
  an isolated bright radar target — the same geometry the tower prior already validated at
  AUC 0.909 for ≥400 kV structures.
- **S2 BSI / NDBI.** Fresh excavation and spoil is the most likely optical signature of a
  recently built scheme.
- **Terrain position.** Valley floor, below a steep drop, beside a stream. Strongest as a
  *prior*, and the reason a DEM had to be added.
- **MNDWI.** A forebay pond, if larger than a pixel — better than NDWI for turbid, shadowed
  alpine water.

## What was built for this

| path | role |
|---|---|
| `scripts/build_hydro_labels.py` | PBF → site complexes + Gate 0 |
| `scripts/hydro_prob_probe.py` | Stage 1: what the existing v9 rasters already say |
| `src/subdetect/terrain.py`, `compose --sensor dem` | Copernicus GLO-30 → elev/slope/valley-position/relief, grid-exact to `composite_0.tif` |
| `scripts/hydro_detectability.py` | Stage 4: terrain-matched controls + per-feature AUC + balance table |
| `scripts/hydro_stage4_model.py` | Stage 4b: GroupKFold multivariate, ablations, permutation null, holdout |

Two pre-existing defects were found and fixed on the way in:

1. **`pixi run compose` was broken in the `default` env for every sensor.** `compose.py`
   imported `tower_prior` at module scope, which imports scipy, which was installed only
   in the `ml` env. Import moved into the `towerprior` branch; `scipy` added to
   `[pypi-dependencies]` (`postprocess.polygonize_chips_v2` had the same latent problem).
2. **Aspect sign.** A north-up raster's rows increase *southward*, so the row gradient must
   be negated. Caught by a cardinal-direction unit check: the first implementation returned
   the uphill bearing (north-facing → 180°) while slope magnitude stayed perfectly correct
   — i.e. it would have silently inverted every illumination-matching decision in Stage 4.

## Acceptance criteria (pre-registered)

### Gate 0 — labels
| gate | condition |
|---|---|
| G0.1 | PBF snapshot ≥ today − 14 d; report `n_sites_new` vs the July PBF |
| G0.2 | ≥ 120 sites inside Pakistan (excluding the `east_of_76` Ladakh overlap) |
| G0.3 | ≥ 100 sites and ≥ 60% with `site_area_m2` ≥ 20,000; also report `n(core_area_m2 ≥ 2,000)` |
| G0.4 | ≥ 8 clusters with ≥ 5 sites; largest cluster ≤ 25% of total |
| G0.5 | site count varies < 10% across `link_m ∈ {100, 150, 250}` |
| G0.6 | ≥ 60% of sites have S2 + S1 composites |
| G0.7 | `data/labels/pakistan/*.parquet` mtimes unchanged |

### Gate 1 — the free v9 probe
| gate | condition |
|---|---|
| G1.0 | ≥ 60 sites covered by a raster; `n_no_coverage` reported separately, never in a denominator |
| G1.1 PASS | AUC vs matched controls ≥ 0.70, CI lower > 0.50, and ≥ 25% of sites reach `p_max` ≥ 0.4 |
| G1.2 WEAK | AUC 0.60–0.70 → **mandatory**: recompute with switchyards masked. If it collapses, this is a substation detection wearing a hydro label |
| G1.3 | AUC must not rise > +0.05 per buffer step across 25/75/150 m. A pass failing this is downgraded to FAIL |
| G1.4 NULL | ≥ 70% `lost:silent` — **expected, and not evidence against feasibility** (see below) |
| G1.FATAL | AUC < 0.50 with controls outscoring sites at n ≥ 60 |

**Why a null at Gate 1 is a null probe, not a negative result:** hydro `power=plant`
polygons are burned as **ignore (−1)** in every substation chip mask
(`local_source.load_substation_labels` → role `plant`, `chips._burn_mask`), and
`min_sub_area_m2` was 20,000 m² — ~37× the median hydro powerhouse — in a training domain
(Indus plain + Indian Punjab) with no mountain terrain. The model was never taught this
class as either positive or negative. Gate 1 can only kill the free-lunch route (re-ranking
existing candidates); it cannot speak to whether the class is detectable.

### Gate 2 — terrain-matched separability
| gate | condition |
|---|---|
| G2.0 | ≥ 80 matched sites and ≥ 240 controls with complete features |
| G2.1 PASS | grouped-CV AUC ≥ **0.75**, site-bootstrap CI lower ≥ 0.68, **and** holdout-valley AUC ≥ 0.70 |
| G2.2 FAIL | CV < 0.65, or holdout < 0.60, or CI lower ≤ 0.55 → **do not spend GPU** |
| G2.3 | 0.65–0.75 → exactly **one** pre-declared extension (multi-date temporal-max S1), then decide |
| G2.4 | ablation: if context-only ≥ all − 0.03, the classifier detects *access and landform*, not hydropower — the product is a prior, not a U-Net |
| G2.5 | AUC by size bucket and capacity: reporting, not pass/fail |

**Prior to any of the above:** the covariate balance table must pass (|SMD| ≤ 0.10 and
per-covariate AUC in [0.45, 0.55]). An unbalanced covariate is a covariate the harness did
not control for, and any AUC quoted beside it is partly that covariate.

**Guards against a small-n false positive.** Every AUC carries a 1,000-sample bootstrap CI
resampling **sites**, not rows (at n≈159 that CI is roughly ±0.07, hence the deliberate
0.65–0.75 dead band). A 200× label-permutation null shuffled **within cluster**; if its
95th percentile clears 0.65 the thresholds are too low for this n and must be raised before
the real number is read. Two seeds minimum, mean reported and CI widened if they differ by
> 0.03. The holdout valley is scored once, at the end. Precedent: v8 → v8_run2, identical
config and a reseed, P@50 0.44 → 0.22 — with the *worse* run posting the *better* val IoU.

**Caliper loosening**, if the strata are too thin: `relief_1km` → `rel_elev_500m` →
`elev_m`, in that order, recorded per site in the samples CSV. `northness`, `d_river_m` and
`valid_frac_s2` are **never** loosened — those three are the ones whose relaxation would
manufacture a result.

### Holdout valley
**Swat/Daral Khwar + Dir/Panjkora, `[71.55, 34.55, 72.95, 35.65]`.** One hydrological
system, ~16 sites (~10% of corpus), 130–180 km from both large clusters. Chosen decisively
on coverage: Swat/Daral 7/9 and Dir 5/9 composite cells with S2 + S1 + tower prior. Kaghan,
Hunza and Astore are disqualified at **0/9** — a holdout that cannot be scored is not a
holdout — and Chitral at 1–2/9 despite having the most sites (12).

---

## Results

*(appended as stages complete)*

### Stage 0 — labels (2026-08-29, `pakistan-latest.osm.pbf`, 2026-07-06 snapshot)

**167 site complexes**, 153 inside Pakistan (14 flagged `east_of_76`, the Ladakh overlap).

| quantity | value |
|---|---|
| median **core** (powerhouse) area | **499 m²** |
| median **site-complex** area | **27,389 m²** |
| sites ≥ 20,000 m² | **142 (93%)** |
| sites with core ≥ 2,000 m² (no buffer) | **34** |
| sites with a penstock mapped | 33 |
| holdout (Swat/Dir) | 30 sites |

**The site-complex definition does what it was designed to do**: a 55× area increase moves
93% of the corpus from a size class with a documented 0.000 recall into the ≥20,000 m²
class where measured recall is 0.714. The unbuffered count (34) is the honest measure of
how many sites could ever be attacked as powerhouse footprints — which is why they aren't.

Core size distribution: `<1k` 100, `1k-2k` 16, `2k-5k` 16, `node_only` 14, `5k-20k` 12,
`≥20k` 9. Clusters: swat_daral 18, chitral_golen 16, hunza 13, skardu 8, dir_panjkora 8,
mangla_ajk 6, gilgit 5, kaghan 5, astore 4, warsak 3, tarbela 3, neelum 2.

| gate | result | |
|---|---|---|
| G0.1 freshness | **FAIL** | 53 d old; fresh PBF downloading (Geofabrik at ~13 KB/s) |
| G0.2 volume | PASS | 153 ≥ 120 |
| G0.3 size | PASS | 142 (93%) ≥ 20k m², vs 100 / 60% required |
| G0.4 spread | PASS | 8 clusters ≥ 5 sites, largest 12% |
| G0.5 stability | PASS | site count moves 2.4% across `link_m` 100/150/250 |
| G0.6 coverage | **FAIL** | **37%** have S2+S1 (need 60%) |

**G0.6 is the live constraint, and its cause is structural.** Of the 80 cells holding the
96 uncovered sites, 57 are inside the substation ROI but were never composited, and **23
are outside the ROI entirely**. The ROI rule is "within 20 km of a mapped power line or
substation" — which by construction excludes off-grid micro hydro, exactly the population
of interest. Compose was therefore given an explicit `--cells-file` path that bypasses ROI
selection (cells the ROI never selected are synthesised from the AOI lattice), and a fill
pass over those 80 cells is running.

Three correctness bugs were found and fixed while getting here, all of which would have
failed silently:

1. **`osmium export` emits closed ways twice** when `linear_tags` and `area_tags` are both
   true — measured: 252 of 256 `power=plant` objects arrived as *both* a LineString and a
   MultiPolygon. Undeduplicated, every closed-way plant would have been counted twice and
   contributed two differently-shaped buffers to its own site. Deduplicated on
   `(osm_type, osm_id)` keeping polygon > line > point.
2. **`buffer(0)` on a Point returns an empty polygon.** The polygon-healing idiom, applied
   blindly to all anchors, deleted all 59 generator *nodes* — a third of the corpus.
3. **A killed download left a truncated PBF** that `wget -c`-on-existence logic treated as
   complete. `--refresh-pbf` now always re-enters wget, and the header-timestamp read
   catches a corrupt file loudly.

### Stage 1 — v9 probe (`predictions_v9_fused/pakistan/prob_mean`, 398 rasters in window)

| buffer | covered | no-coverage | median p_max | ≥ 0.4 seed | silent (p_max < 0.05) |
|---|---|---|---|---|---|
| 25 m | 57 | 110 | **0.000** | 5% | 88% |
| 75 m | 58 | 109 | **0.000** | 9% | 83% |
| 150 m | 61 | 106 | **0.000** | 13% | 80% |

**G1.4 NULL — the pre-registered expected outcome.** Median peak probability is exactly
0.000 at every buffer; 83% of covered sites are wholly silent. G1.0 also fails narrowly
(58 covered vs 60 required), and coverage cannot improve without running inference on the
new cells, which needs GPU and is out of Stage 1's scope.

This kills the free-lunch route — re-ranking existing v9 candidates — and nothing else.
The model was never taught this class in either direction: hydro `power=plant` polygons are
burned as ignore (−1) in every substation chip mask, and `min_sub_area_m2` was 20,000 m²,
~40× the median hydro powerhouse.

**The switchyard split is the informative part.** Among covered sites, those with a mapped
substation within 300 m reach a maximum probability of **1.000**; those without top out at
**0.41**. Medians are 0.000 for both. So the handful of sites the model does see, it sees
because of their switchyard — precisely the confound G1.2 was written to detect, visible
here even though the overall result is null.

Note also that `frac_p_max_ge_0.4` climbs 5% → 9% → 13% purely with buffer size. That is
the mechanical artefact G1.3 exists to catch: a larger polygon captures more of a noisy
raster. It is why `p_max` is never quoted here without the sweep beside it.

### Stage 4 — separability

_pending_

## Verdict

_pending_
