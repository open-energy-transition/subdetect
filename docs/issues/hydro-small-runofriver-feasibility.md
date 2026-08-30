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

### Stage 0 — labels (re-run 2026-08-29 on a fresh `2026-08-28` snapshot)

**G0.1 now PASSES** (PBF age 1 d). The refresh was worth making: **101 hydro features are
new** since the 2026-07-06 snapshot (3,114 → 3,213 in the northern window), and mapped
penstocks went 33 → 35. Net site count barely moved (167 → 168), because most of the new
features attach to sites that already existed rather than creating new ones — so the July
numbers below stand, with 154 sites in PK and 143 (93%) ≥ 20,000 m².

Only **G0.6 (imagery coverage)** still fails, at 41% and rising as the compose fill runs.

**168 site complexes**, 154 inside Pakistan (14 flagged `east_of_76`, the Ladakh overlap).

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

### Stage 4 — separability (2026-08-30, `north_pk_strict`, strict pre-registered calipers)

**G0.6 passed first** (64% coverage) after the compose fill reached 99/154 sites with all
three layers, so this is the real run, not the earlier harness validation.

86 matched sites (65 study / 21 holdout), 1,028 terrain-matched controls, median 14 per
site. **G2.0 PASSES.**

| feature group | CV AUC | CI lower (out-of-fold) | holdout |
|---|---|---|---|
| **all** | **0.737** | **0.676** | **0.671** |
| S2 only | 0.665 | 0.604 | 0.606 |
| S1 only | 0.614 | 0.546 | 0.623 |
| context / terrain only | 0.593 | 0.551 | 0.591 |

Permutation null (within-group shuffle, n=200): median 0.564, **95th percentile 0.621** —
below the 0.65 ceiling, so the pre-registered thresholds remain valid at this n. p = 0.000.

**VERDICT: AMBIGUOUS.** The signal is real — comfortably above the permutation null — but
lands in the pre-declared 0.65–0.75 dead band, missing all three G2.1 conditions
simultaneously and narrowly: CV 0.737 vs 0.75, CI lower 0.676 vs 0.68, holdout 0.671 vs
0.70. Missing all three by a similar margin is a coherent picture rather than noise. It is
equally clearly not a G2.2 failure.

Per G2.3 that permits **exactly one** pre-declared extension — multi-date temporal-max S1 —
and then a decision. No open-ended tuning.

**Recall by powerhouse size, the line that matters most:**

| core bucket | n sites | AUC |
|---|---|---|
| `node_only` | 9 | **0.374** |
| `<1k` | 31 | 0.676 |
| `1k-2k` | 5 | 0.712 |
| `2k-5k` | 7 | 0.636 |
| `5k-20k` | 7 | 0.731 |
| `≥20k` | 6 | **0.758** |

Detectability still tracks the **real built footprint**, not the buffer that was drawn
around it. The site-complex label lifted the target above the *label-geometry* floor — that
was its job, and it worked — but it did not manufacture signal where the ground has none.
Node-only sites score *below chance*: their "complex" is a 75 m disc around a point whose
position need not coincide with any structure, and they should probably be excluded from
training rather than treated as positives.

**Two honest limitations, both visible in the balance table:**

| covariate | SMD | AUC | balanced |
|---|---|---|---|
| slope_deg | 0.074 | 0.527 | ✓ |
| northness | −0.042 | 0.482 | ✓ |
| rel_elev_500m | **−0.400** | 0.380 | ✗ |
| elev_m | 0.311 | 0.580 | ✗ |
| valid_frac_s2 | −0.232 | 0.601 | ✗ |
| d_river_m | −0.177 | 0.399 | ✗ |
| relief_1km | 0.133 | 0.539 | ✗ |

1. **Balance fails on valley position and river distance, and tightening the calipers does
   not fix it.** A strict re-run (`--max-loosen 0`) gave 82 matched sites and near-identical
   numbers, so this is structural, not a loosening artefact: hydro sites occupy the extreme
   low end of the valley-position distribution (median `rel_elev_500m` 0.186 vs 0.275 for
   controls) and hug rivers more tightly (65 m vs 107 m) than *any available* control can.
   There is simply not enough valley floor in these catchments to match against. So part of
   the measured separation is unavoidably "lower and nearer the water", and `context`-only
   sitting at 0.593 rather than 0.500 quantifies that residue. G2.4 still passes — the
   0.144 gap to `all` says the classifier is not merely reading landform — but the AUCs
   above should be read as an upper bound.
2. **The permutation null centres on 0.564, not 0.500.** A clean null should sit at chance.
   The residual is most likely the site/control geometry asymmetry (irregular real polygons
   vs circular control discs of matched area). It is below the pre-registered 0.65 ceiling
   so the gates stand, but any future run should shrink it — matching control *shape* as
   well as area is the obvious fix.

**Also worth recording: the n=19 harness-validation numbers were badly inflated.**
`vh_contrast_max` read 0.918 at n=19 and 0.704 at n=82; the multivariate `all` arm went
0.771 → 0.737 with a far lower CI. This is the v8/v8_run2 lesson repeating exactly, and is
why G2.0 exists as a gate rather than a footnote.

## How to complete this study

Everything except the final Stage 4 number is done. The remainder is unattended download:
filling S2 + S1 + DEM for the hydro cells the substation ROI never covered.

**Running now** (systemd user units, they survive a closed session):

```bash
systemctl --user status subdetect-hydro-s2      # S2 for the 80 uncovered cells
systemctl --user status subdetect-hydro-fill    # follows it with S1, then DEM
```

Note both were launched with `--setenv=PATH=<repo>/.pixi/envs/default/bin:...`; without it
`osmium` is not on the unit's PATH and the label build fails at the header-timestamp read.

**Check coverage** (G0.6 wants ≥ 60% of 154 sites; Stage 4's G2.0 wants ≥ 80 *matched*
sites, and roughly 90% of ready sites draw controls, so aim for ~95 with all three layers):

```bash
pixi run python -c "
import geopandas as gpd; from pathlib import Path
s=gpd.read_parquet('data/labels/pakistan_hydro/hydro_sites.parquet'); s=s[~s.east_of_76]
C=Path('data/composites/pakistan/composites'); f=lambda c,n:(C/c/n).exists()
print(sum(all(f(c,n) for n in ['composite_0.tif','composite_s1.tif','composite_dem.tif'])
          for c in s.cell), '/', len(s))"
```

**Then run the real Stage 4** (CPU only, ~1–2 h):

```bash
pixi run python scripts/build_hydro_labels.py --refresh-pbf      # re-check G0.6
pixi run -e ml python scripts/hydro_detectability.py --run-name north_pk_v1
pixi run -e ml python scripts/hydro_stage4_model.py --run-name north_pk_v1
```

**Read the covariate balance table before any AUC.** It is what licenses them; if it does
not pass, the S2/S1 numbers are partly the covariates it failed to control. Do not move a
gate threshold after seeing a number — that is the failure mode this whole structure exists
to prevent, and the v8/v8_run2 episode is the local precedent for how easily a good-looking
single number evaporates.

If Stage 4 passes, Stage 5 (chips + fine-tune) begins — and note two things settled here:
use `--fraction` / `all_touched=False` from day one, since the 1.70× label inflation is
worst in exactly this size class; and do **not** touch `pakistan.val_bbox`, which
`chips._split_of` uses for every substation chip index. Add a new AOI with the same
`grid_origin` instead.

## Verdict

_pending Stage 4 — the imagery fill is the only thing in the way._
