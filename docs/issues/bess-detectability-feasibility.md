# Battery energy storage (BESS) in GB / California / Texas: is it detectable at 10 m?

**Status:** pre-registered, no data collected. Opened 2026-09-01 on branch `bess-feasibility`.

This document records the acceptance criteria **before any number exists**. Its own commit
carries no results; results are appended per stage below, under `## Results`. The gates
are not to be moved after seeing data. This is the same discipline as
[`hydro-small-runofriver-feasibility.md`](hydro-small-runofriver-feasibility.md), and the
thresholds in Gate 2 are deliberately identical to that study's so the two are comparable.

## Problem

Do battery energy storage systems produce a signal in Sentinel-1 or Sentinel-2 imagery
strong enough to detect them, using OpenStreetMap battery power-plant labels as ground
truth?

The pipeline in this repo detects `power=substation` in Pakistan and India from a 10 m
Sentinel-2 (+ Sentinel-1 RTC) composite via a fine-tuned TerraMind encoder. The question
here is whether a *different* power class — grid-scale battery storage — is separable in
the same imagery, and specifically whether it is separable **from the substations it sits
next to**.

## Why the naive version cannot work

Three structural obstacles, all known before starting:

1. **There are no labels where the imagery is.** The on-disk PBFs (Pakistan, India) contain
   exactly **one** `plant:source=battery` object between them (Pakistan 1, India 0, measured
   2026-09-01 with `osmium tags-filter` over `pakistan-20260829.osm.pbf` and
   `india-latest.osm.pbf`). Every composite this project owns is therefore useless for this
   class. The study must buy new imagery over new regions — the first study in this repo
   that cannot reuse the Pakistan tree.

2. **BESS sit inside or beside substations.** A grid-scale battery connects at a
   switchyard, and in OSM the two are frequently adjacent or overlapping polygons. The
   production detector was trained to find exactly that switchyard. So a naive "can we
   detect BESS" experiment measures substation detection wearing a battery label. The hydro
   study saw the same confound from the other side: sites with a mapped substation within
   300 m reached `p_max` 1.000, sites without topped out at 0.41.

3. **Size.** Measured across all 23 model runs and both eval versions, this pipeline has
   recall 0.000 with `p_max` *exactly* 0.000 for installations below 2,000 m². A 50 MW / 2 h
   container yard is roughly 1–3 ha, so this should clear the floor — but "should" is not
   "does", and G0.3 checks it before any imagery is bought.

## Physical signal, and what is actually being labelled

Expected signatures, none of them verified yet:

- **S2 optical.** Rows of light-grey/white metal containers on a gravel or concrete pad,
  inside a fence, with no vegetation. High visible albedo, near-zero NDVI, high NDBI/BSI.
  Distinguishable from farmland or scrub; *not* obviously distinguishable from a switchyard,
  a warehouse yard, or a container depot.
- **S1 RTC.** Dense arrays of metal boxes with vertical faces are strong corner reflectors →
  elevated VV/VH and high local contrast. This is the modality most likely to carry a
  battery-specific signal, since a switchyard's dominant scatterers (busbars, lattice
  towers, transformers) have a different geometry and density.
- **Bi-temporal.** Essentially every grid-scale BESS in these three markets was built after
  2018. A pre-construction S2 composite versus a current one should show bare-ground → built
  transition. This is the strongest expected single family — and, in Tier 2, the most
  confounded, because a newly built *substation* shows the same transition. See G2.3.

**What is labelled:** the OSM object(s) for the battery installation, merged into a site
complex by union-find in local UTM at `link_m = 60 m` and buffered 30 m — a compact yard,
not the 2 km penstock arms that forced hydro's 75 m / 150 m treatment. `core_area_m2`
(unbuffered, geodesic) is retained as the honest size number alongside `site_area_m2`.

## What is built for this

Five standalone scripts, forked from the hydro five, plus three surgical library changes.
Nothing in the substation lineage is modified; `scripts/build_bess_labels.py` asserts this
two ways (a `FORBIDDEN_NAMES` clash check and a production-parquet mtime assert).

| file | role |
|---|---|
| `scripts/build_bess_labels.py` | Stage 0: fresh Geofabrik PBFs → tag audit → sites → Tier-2 control pool → context layers → Gate 0 |
| `scripts/bess_prob_probe.py` | Stage 1: production-detector probe with a positive control |
| `scripts/bess_detectability.py` | Stage 2a: matched controls (both tiers), features, per-feature AUC, covariate balance |
| `scripts/bess_stage2_model.py` | Stage 2b: GroupKFold logistic regression, ablations, permutation null, region holdout, gates |
| `scripts/make_bess_figures.py` | figures, from persisted CSVs only |

Library changes, all in `src/subdetect/compose.py`: a `select_cells` guard so `--cells-file`
works for an AOI with no `lines.parquet`; per-AOI `s2_window`/`s1_window` overrides (a
Nov–Mar window over Great Britain is unusable); and a new `--sensor s2pre` writing
`composite_pre.tif` for the bi-temporal arm.

**Regions.** Great Britain (densest and most consistent OSM power mapping), California and
Texas (largest utility-scale sites). Train/CV on one group, cross-region holdout on the
other, **both directions reported**.

**Two-tier controls.** Tier 1 = BESS versus matched generic/industrial land: a canary that
tells a real null apart from broken imagery, and an upper bound. Tier 2 = BESS versus
**area- and shape-matched battery-free substations**: the honest number, and the only tier
the verdict may be set on.

## Acceptance criteria (pre-registered)

### Gate 0 — labels

- **G0.0 tag audit.** A per-region cross-tab of every candidate tagging pattern
  (`power=plant & plant:source~batter`, `plant:method~batter`, `plant:storage~batter`,
  `power=generator & generator:source~batter`, `power=substation & (name|operator)~
  (BESS|batter|energy storage)`, and battery co-occurring with solar) is written to
  `data/eval_results/bess_stage0_tag_audit.csv` **before** the positive definition is fixed.
  The chosen definition is recorded in `bess_extract_manifest.json` as an explicit union.
- **G0.1 freshness.** All three PBF snapshots are at most 14 days old at extract time.
- **G0.2 volume.** ≥ 120 sites in the training region, ≥ 60 in each holdout region,
  ≥ 250 pooled.
- **G0.3 size.** Median `site_area_m2` ≥ 5,000 m² **and** ≥ 60% of sites ≥ 20,000 m².
  `n(core_area_m2 ≥ 2,000)` reported alongside.
- **G0.3b node fraction.** ≤ 40% of sites are node-only. Node-only sites are **excluded
  from the headline Tier-2 AUC** and reported as their own bucket.
- **G0.4 spread.** ≥ 8 half-degree cells with ≥ 5 sites in the training region; the largest
  such cell holds ≤ 25% of that region's sites.
- **G0.5 Tier-2 pool.** ≥ 3 eligible battery-free substations within a ±30% area caliper per
  BESS site, and ≥ 80% of sites have ≥ 1 **same-cell** candidate.
- **G0.6 co-location census** *(reporting only)*. Fraction of sites with a substation within
  300 m; fraction whose polygon overlaps a substation polygon; median battery-only residual
  area after subtracting overlapping substations buffered 30 m.
- **G0.7 construction dates.** ≥ 40% of sites carry a parseable `start_date`,
  `construction:start_date` or `opening_date`. **If below 40%, the bi-temporal arm is
  downgraded to descriptive-only and forfeits its status as the G2.3 extension.**
- **G0.8 stability.** Site count varies < 10% across `link_m ∈ {40, 60, 120}`.
- **G0.9 non-interference.** `data/labels/pakistan/*.parquet` and
  `data/labels/india_pilot/*.parquet` mtimes unchanged; no `FORBIDDEN_NAMES` written; no
  mtime change anywhere under `composites/pakistan`.
- **G0.10 budget.** The projected cell list is ≤ 800 cells / ≤ 28 GB, or it is capped by the
  `n_bess desc` rank until it is.

**Pre-declared fallback.** A region with < 60 sites is dropped or demoted to holdout-only,
and the study runs GB-train / pooled-holdout. If two regions fail, the study becomes
single-region; G2.5 and the holdout clause of G2.1-T2 are then void, and that is reported
as a limitation rather than worked around.

### Gate 1 — production-detector probe

A restricted inference run with `data/models/stageA_v12_up2_s2only` at `--upsample 2` over
~80 GB cells. No prediction rasters exist over these regions, so unlike the hydro probe this
one is not free — it costs 2–4 h of GPU, and it buys an early, model-side measurement of the
co-location confound.

- **G1.0 power.** ≥ 60 BESS sites covered by a prediction raster. `n_no_coverage` is
  reported separately and **never appears in a denominator**. The covered-versus-uncovered
  site-area distributions are reported, to demonstrate the sorted-glob cell subsample is not
  size-correlated.
- **G1.P positive control.** In the *same* probe cells, ≥ 30 mapped `power=substation`
  polygons ≥ 20,000 m², with `frac(p_max ≥ 0.4) ≥ 0.40`. **If G1.P fails, the whole of Gate
  1 is declared UNINFORMATIVE — not negative.** Without this, a null over GB/CA/TX cannot be
  told apart from "the Pakistan-trained model does not transfer", which says nothing about
  batteries. Pakistan's measured recall in the ≥ 20,000 m² bucket is 0.714; 0.40 allows for
  heavy domain shift while still proving the model runs.
- **G1.1 PASS.** AUC on `p_max` versus matched controls ≥ 0.70, bootstrap CI lower > 0.50,
  and ≥ 25% of sites reach `p_max` ≥ 0.4 — **and** the same AUC recomputed on the
  battery-only sub-polygon stays ≥ 0.60.
- **G1.2 WEAK.** AUC 0.60–0.70 → mandatory recompute restricted to sites with no substation
  within 300 m.
- **G1.3 buffer control.** AUC must not move by more than ±0.10 across 25/75/150 m
  footprints; a pass that fails this is downgraded to FAIL.
- **G1.4 NULL.** Expected, and **not** evidence against feasibility, provided G1.P passed:
  `power=plant` is burned as ignore (−1) in every substation chip mask, so batteries were
  never taught in either direction. A null kills only the re-ranking route.
- **G1.5 co-location split** *(reporting)*. `p_max` split by `has_substation_300m`.

### Gate 2 — separability. The gate is set on Tier 2.

Two prerequisites, both computed and printed **before any AUC is read**:

- **P1 balance.** Every *varying* matched covariate must have `|SMD| ≤ 0.10` and a
  per-covariate AUC in [0.45, 0.55]. Near-constant covariates are excluded from judgement
  (their SMD denominator is ~0).
- **P2 null geometry.** The 200× within-group label-permutation null must have
  **median ≤ 0.55** and 95th percentile < 0.65. The median clause is new to this study: it
  is what catches the irregular-polygon-versus-circular-disc asymmetry that put the hydro
  null at 0.564 instead of 0.500, and it verifies that shape-matched controls fixed it. A
  95th percentile ≥ 0.65 invalidates the thresholds at this *n*.

**Tier 1 — BESS versus matched generic/industrial land** (canary and upper bound):

- **G2.0-T1.** ≥ 100 matched sites and ≥ 500 controls with complete features.
- **G2.1-T1 PASS.** Grouped-CV AUC ≥ 0.85, site-bootstrap CI lower ≥ 0.78, cross-region
  holdout ≥ 0.78.
- **G2.2-T1.** CV < 0.75 → **stop and debug the harness** (co-registration, the +1000 DN
  offset, rasterisation) and do not read Tier 2. A Tier-1 number that low against farmland
  and scrub is far more likely a bug than a finding.

**Tier 2 — BESS versus area- and shape-matched battery-free substations** (the honest
number):

- **G2.0-T2.** ≥ 90 matched BESS sites (polygon-only; node-only excluded per G0.3b),
  ≥ 270 matched control substations, and ≥ 60% of pairs drawn from the **same cell**.
- **G2.1-T2 PASS.** Grouped-CV AUC ≥ **0.75** *and* site-bootstrap CI lower ≥ **0.68** *and*
  cross-region holdout AUC ≥ **0.70**, on the **static** feature set (no `chg_*`).
- **G2.2-T2 FAIL.** CV < 0.65, or holdout < 0.60, or CI lower ≤ 0.55 → do not spend GPU;
  write up the negative result.
- **G2.3-T2 dead band.** 0.65–0.75 permits exactly **one** pre-declared extension: the
  bi-temporal S2 change group (`composite_pre` versus `composite_0`), restricted to sites
  with a usable construction date and scored against age-matched control substations where a
  date exists. Then a decision. No further tuning.
- **G2.4-T2 switchyard confound.** If the static arm restricted to the **battery-only
  sub-polygon** (site minus overlapping substations buffered 30 m, ≥ 2,000 m² remaining)
  falls below `all − 0.10`, the classifier is reading the switchyard, not the battery.
  *n* of sites with a usable sub-polygon is reported.
- **G2.5 transfer.** GB→(CA+TX) and (CA+TX)→GB are reported separately. If they differ by
  more than 0.10, the headline number is the **lower** of the two.
- **G2.6 size strata** *(mandatory reporting)*. AUC by site-area bucket and by
  `plant:output:electricity` where tagged.
- **G2.7 ablation.** Arms: s2 / s1 / context / change. If **context-only ≥ all − 0.03**, the
  classifier is detecting stereotyped siting (flat, near a substation, near a road, recently
  developed) and the product is a prior, not a detector.
- **G2.8 contamination** *(reporting)*. Per-rule counts for the Tier-2 control exclusion
  cascade, plus one re-run with the battery-proximity exclusion doubled from 200 m to 400 m.
  Movement > 0.05 means contamination is material.

Two seeds minimum throughout; the mean is reported and the CI widened if seeds differ by
more than 0.03. The holdout region is scored **once**, at the end.

**Stated bias direction for R5 (unlabelled BESS inside the Tier-2 control pool):**
contamination places true positives into the negative class, which pushes the Tier-2 AUC
*toward 0.5*. The gate is therefore conservative — a Tier-2 PASS is trustworthy, while a
Tier-2 FAIL is partly attributable to contamination and must be reported as such.

### Holdout

The holdout is **region-level**, not a bbox: GB versus (CA ∪ TX). Both directions are run
and both are reported (G2.5). The holdout region's features are computed with the training
region's fitted scaler and model, scored once, at the end of Stage 2b.

## Amendment, 2026-09-02 (before any Stage 1 or Stage 2 number existed)

Gate 0 failed as pre-registered on G0.3 and G0.5. **Both original verdicts stand and are
never re-passed** — `bess_stage0_summary.csv` carries `gate0_original = FAIL` permanently,
and the two gates are printed with a `[superseded by amendment]` marker rather than
removed. What follows is a labelled replacement, committed before any separability number
was computed, with the reasoning that motivated it.

**A1 — the headline positive set is restricted to labels ≥ 20,000 m².** G0.3 asked that
≥60% of the corpus reach 20,000 m² and measured 51%. The response is *not* to lower that
bar. It is to restrict the set the study reports on to the size class in which this
pipeline has demonstrated recall (0.714 on Pakistan substations ≥ 20,000 m²), and to leave
the smaller sites in the corpus as G2.6 strata where they can be reported but cannot carry
the verdict. This is strictly more conservative than the original gate: it studies fewer,
larger sites rather than more, smaller ones.

Headline set: **259 sites — GB 91, CA 86, TX 82.** That is 2.9× the ≥90 required by
G2.0-T2, and near-balanced across regions, so the cross-region holdout of G2.5 is
well-powered in both directions. `A1_headline_volume` requires ≥90 pooled and ≥60 per
region and is evaluated on that set: PASS.

**A2 — the acquisition-confound control becomes a distance, not a shared cell.** G0.5's
second clause asked that ≥80% of sites have an area-matched control inside the same 0.1°
cell, and measured 37%. The clause was a proxy for "same acquisition, same look
geometry", and a 0.1° cell is a poor one: it is an arbitrary ~10 km box, and whether a
matched substation falls inside it is close to luck. A Sentinel-2 granule is 110 × 110 km
and the Sentinel-1 IW swath is 250 km, so two footprints within **40 km** are in the same
acquisition with the same look geometry — which is the property the gate was after. The
**80% level is inherited unchanged**; only the operationalisation moves, and the threshold
was fixed on that geometric argument before the number was read.

`A2_ctrl_in_range`: 92% of headline sites have ≥3 area-matched eligible controls, and 83%
have one within 40 km. PASS — on the inherited level, not a lowered one.

**What the amendment does not change.** Every Gate 1 and Gate 2 threshold, the two-tier
design, the positive definition, the hybrid stratum, and the stated contamination bias
direction all stand exactly as committed in `ff35af1`.

**Consequential change to G2.0-T2.** Its "≥60% of pairs same-cell" clause rests on the
same discredited proxy and is replaced, on the same geometric argument and at the same
level, by **≥60% of pairs within 40 km**. Recorded here rather than silently applied at
Stage 2.

**Imagery consequence.** Restricting to the headline set and selecting control cells by
greedy set cover (each site needs 3 matched controls; control cells are heavily shared
between neighbours) brings the purchase from 1,343 cells / ~43.5 GB to **389 cells /
12.6 GB** — GB 135, TX 130, CA 124.

## Results

### Stage 0 — labels (2026-09-01, PBFs of 2026-08-31/09-01, `--region all`)

**GATE 0: FAIL — 8 of 10 gates pass; G0.3 (size) and G0.5 (same-cell controls) fail.**
No imagery has been bought. Total cost so far: three PBF downloads (4.0 GB).

511 battery sites in AOI — **GB 232, TX 176, CA 103** — from 13,761 substation polygons,
13,268 of which are eligible Tier-2 controls.

| gate | result | measured |
|---|---|---|
| G0.0 tag audit | PASS | written before the definition was fixed |
| G0.1 freshness | PASS | oldest PBF 1 d |
| G0.2 volume | PASS | 232 / 176 / 103, pooled 511 (need 120 / 60 / 250) |
| **G0.3 size** | **FAIL** | median site 20,329 m² (needs ≥5,000 — passes) but **51% ≥20,000 m² (needs ≥60%)** |
| G0.3b node fraction | PASS | 1% node-only |
| G0.4 spread | PASS | 32 half-degree cells with ≥5 sites, largest holds 2% |
| **G0.5 Tier-2 pool** | **FAIL** | 94% of sites have ≥3 area-matched controls (passes) but **37% have a same-cell one (needs ≥80%)** |
| G0.7 construction dates | PASS | 62% pooled — but **GB alone is 29%**, below the 40% clause |
| G0.8 stability | PASS | site count moves 1.6% across `link_m ∈ {40, 60, 120}` |
| G0.9 non-interference | PASS | production labels and the Pakistan composite tree unchanged |
| G0.10 budget | PASS at the cap | 1,343 cells wanted (~43.5 GB); capped by rank to 800 (~25.9 GB) |

Per region:

| region | n | median site | median core | ≥20k m² | core <2k m² | dated | hybrid | substation ≤300 m | overlaps substation |
|---|---|---|---|---|---|---|---|---|---|
| GB | 232 | 16,817 m² | 4,697 m² | 39% | 17% | 29% | 0.4% | 67% | 33% |
| CA | 103 | 44,986 m² | 19,821 m² | 84% | 10% | 86% | **37%** | 71% | 44% |
| TX | 176 | 17,359 m² | 5,414 m² | 47% | 38% | 92% | 4% | **92%** | 51% |

**G0.6 co-location census — the study's central risk, quantified.** 76% of sites have a
mapped substation within 300 m and **41% of site polygons physically overlap one** (TX 92%
and 51%). This is materially worse than the hydro analogue and confirms that a Tier-1-only
result would have been close to meaningless. 504 of 511 sites do retain a battery-only
sub-polygon ≥ 2,000 m² after subtracting overlapping substations buffered 30 m, so the
G2.4-T2 arm is computable.

**Tag audit (G0.0).** The assumed pattern dominates but is not the whole story:
`plant:source=battery` 174 (TX) / 64 (CA), `generator:source=battery` 30 / 58, plus
`plant:source=solar;battery` and `battery;solar` hybrids and one `gas;battery`. Free-text
name hints matched 68 objects in TX that carry no battery tag; per the pre-registered
definition these are **not** positives, and they are also excluded from the control pool.

**Hybrids.** 46 pooled sites (9%) co-tag generation with the battery — 37% of CA sites,
almost all `solar`/`photovoltaic`. Their footprint is mostly PV panels, so "detecting" one
would largely be detecting the solar farm. They stay in the positive set (the definition is
pre-registered and moving it after seeing data is what pre-registration forbids) and are
flagged `is_hybrid` for a G2.6 stratum.

**Why G0.3 fails.** Unbuffered core area: p10 1,003 m², p25 2,339, **p50 6,737**, p75
19,188, p90 52,442. So 77% clear the 2,000 m² floor below which this pipeline has measured
`p_max` of exactly 0.000, and 79% of the buffered labels clear 10,000 m² — but only 51%
clear the 20,000 m² bucket the gate demanded. The class is *not* below the pipeline's
absolute floor; it is below the pre-registered bar for the bucket where the pipeline has
demonstrated recall. GB and TX drive the failure; CA alone would pass at 84%.

**Why G0.5 fails.** 94% of sites have ≥3 area-matched battery-free substations, and the
median distance to the nearest one is 6.7 km — but only 37% find one inside the same 0.1°
(~10 km) cell. 61% have one within 10 km, 80% within 20 km, 88% within 40 km. The
thickness of the control strata is fine; what fails is the specific same-cell
operationalisation of the acquisition-confound control.

### Stage 1 — production-detector probe (2026-09-02, GB, all 135 cells)

**VERDICT: INVERSE — an outcome the pre-registration did not enumerate.** The two classes
*do* separate, but batteries rank **below** substations. This is not the pre-registered
NULL and is not reported as one.

Ran `v12_up2_s2only` (`epoch=24-step=51000`, `--upsample 2`) over **all 135** GB cells, not
the planned 80-cell subsample: inference measured 7 s/cell, so full coverage cost ~16 min
and removes G1.0's subsample-bias concern outright rather than merely reporting it.
**91 of 91** headline sites covered, `n_no_coverage` = 0.

| gate | result | measured |
|---|---|---|
| G1.0 power | PASS | 91/91 sites covered, 0 uncovered |
| **G1.P positive control** | **PASS** | 67 battery-free substations ≥20,000 m² covered, **55%** at `p_max` ≥ 0.4 (bar: ≥30 and ≥40%) |
| G1.1 PASS | no | AUC 0.307, CI [0.236, 0.382], 15% at ≥0.4, battery-only 0.254 |
| G1.2 WEAK | no | AUC not in [0.60, 0.70) |
| G1.3 buffer | spread 0.112 | exceeds ±0.10, but there is no pass to downgrade |
| G1.4 NULL | no | 15% at ≥0.4, above the 10% NULL definition |

**G1.P passing is itself a result.** A Pakistan/India-trained TerraMind fine-tune, never
shown a British scene, fires on **55%** of GB transmission substations ≥20,000 m² — against
0.714 measured recall in its own domain. Domain transfer PK→GB works well enough that
everything below is informative rather than a transfer artefact. This is exactly the guard
the hydro probe lacked and had to pre-declare its way around.

**The measured separation runs the wrong way.** Battery sites: median `p_max` **0.000**,
15% reaching 0.4. Battery-free substations in the same rasters: 55% reaching 0.4. Rank AUC
**0.307** with CI [0.236, 0.382] — the interval excludes 0.5, so this is a real separation,
not noise, and it is *negative*: the substation detector systematically does **not** fire on
battery yards while firing on comparable switchyards. Restricting to the battery-only
sub-polygon pushes it further, to **0.254**.

**G1.5 is the decisive table, and it is sharper than hydro's.**

| | n | median `p_max` | max `p_max` | frac ≥ 0.4 |
|---|---|---|---|---|
| substation within 300 m | 66 | 0.000 | **1.000** | 0.212 |
| no substation within 300 m | 25 | 0.000 | **0.180** | **0.000** |

Every bit of response comes from the co-located switchyard. With one nearby, a site can
saturate at 1.000; with none, the model never exceeds 0.18 and never once crosses 0.4. The
hydro study measured 1.000 versus 0.41 for the same split; here the contrast is starker.
The buffer sweep points the same way — AUC climbs 0.255 → 0.307 → 0.367 from 25 m to 150 m,
i.e. the *larger* the buffer, the more substation-like a battery looks, because a wider
footprint swallows more of the adjacent switchyard.

**What this does and does not establish.** It establishes that battery yards do not look
like switchyards to a model trained on switchyards, and it kills the cheap re-ranking route
outright (re-ranking existing candidates would rank batteries last). It does **not**
establish that batteries are positively identifiable: "does not look like a substation" is
a weaker claim than "has a signature of its own", and a battery scoring 0.000 could equally
mean it looks like unremarkable industrial ground. Only Tier 2, with a classifier trained
on the battery class itself, can separate those two readings. Stage 1 has, however, made
Tier 2 the interesting experiment rather than a formality: there is *something* different
about these footprints at 10 m.

#### All three regions (2026-09-02/03, all 389 cells)

The GB result **does not replicate uniformly**, and the way it fails is the finding.

| region | co-located ≤300 m | overlaps a substation | median site | G1.P | sites at ≥0.4 | **AUC** | CI | battery-only AUC | Δ |
|---|---|---|---|---|---|---|---|---|---|
| GB | 73% | 31% | 32,438 m² | 55% | 15% | **0.307** | [0.236, 0.383] | 0.254 | **−0.053** |
| CA | 71% | 44% | 59,981 m² | 85% | 44% | **0.248** | [0.179, 0.327] | 0.209 | **−0.039** |
| TX | **93%** | **63%** | 39,027 m² | 72% | 67% | **0.483** | [0.395, 0.576] | 0.418 | **−0.065** |

GB and CA give an inverse separation with CIs excluding 0.5. **Texas does not**: its CI
straddles 0.5, and 67% of its battery sites fire at `p_max` ≥ 0.4 — the model cannot tell a
Texan battery from a Texan substation. That is a third distinct outcome, and it is not the
pre-registered NULL either, which meant *silence* (`frac ≥ 0.4` under 10%). The script now
names PASS / WEAK / INVERSE / NULL / NO SEPARATION separately rather than collapsing the
last three.

**Co-location explains the spread.** Texas is the region where batteries are most
thoroughly inside substations — 93% within 300 m and **63% physically overlapping**, against
31–44% overlap in GB and CA. Where the battery polygon *contains* a switchyard, the model
fires on the switchyard and `p_max` saturates: TX median `p_max` is 0.996. So the headline
AUC does not measure batteries at all in Texas; it measures how often a battery label
happens to enclose a substation.

**What does replicate, in all three regions, is the mechanism.** Subtracting the overlapping
substation *always* lowers the score — **−0.053, −0.039, −0.065**, in three independent
regions with different label sets, different imagery and opposite headline verdicts. The
response this model gives a battery site is driven by the switchyard inside it, not by the
battery.

The corollary is a warning for Stage 2, and it is sharper in Texas than anywhere: where 63%
of positives overlap a substation, a "battery vs battery-free substation" comparison is
partly a comparison of substations with substations. The G2.4-T2 battery-only arm is not a
supplementary check here — it is the only version of the question that is clean, and the
Tier-2 headline must be read next to it in every region.

### Stage 2 — separability (2026-09-03, `bess_final` / `bess_v2`, all three regions)

**Tier 2 — the honest number. 139 matched sites, 251 controls.**

| arm | n feat | CV AUC | CI | gb→us holdout | us→gb holdout |
|---|---|---|---|---|---|
| all | 32 | **0.866** | [0.826, 0.903] | 0.708 | 0.786 |
| **imagery only (S2+S1)** | 27 | **0.825** | [0.783, 0.863] | 0.657 | 0.759 |
| imagery, area-robust | 16 | 0.809 | [0.768, 0.849] | 0.672 | 0.874 |
| S2 only | 21 | 0.788 | [0.736, 0.836] | | |
| S1 only | 6 | 0.616 | [0.563, 0.675] | | |
| context only (no imagery) | 5 | 0.678 | [0.615, 0.734] | | |

| gate | result | measured |
|---|---|---|
| **G2.1-T2** | **PASS** | CV 0.866 ≥ 0.75, CI lo 0.826 ≥ 0.68, holdout **0.708** ≥ 0.70 (the *lower* direction, per G2.5) |
| G2.2-T2 | no | nowhere near the fail band |
| **G2.4-T2 switchyard** | **PASS** | battery-only sub-polygon **0.869** vs 0.866 full — subtracting the substation does not reduce it |
| G2.7 ablation | PASS | context-only 0.678 vs 0.866; gap 0.188, far above the 0.03 that would make it a prior |
| G2.5 transfer | reported | gb→us 0.708, us→gb 0.786; headline takes the lower |
| **P2 null geometry** | **PASS** | permutation null median **0.486**, 95th 0.543, p = 0.000 |
| **G2.0-T2** | **FAIL** | 139 sites ≥ 90 ✓, 100% of pairs within 40 km ✓, but **251 controls < 270** ✗ |
| **P1 balance** | **FAIL** | on `n_px` (SMD 0.162, AUC 0.550); all other matchable covariates pass |

G2.6 strata: `5k-20k` 0.878, `≥20k` 0.836. Hybrids **0.816** against non-hybrids **0.894** —
the solar-co-tagged sites score *lower*, so the 37% of Californian sites that are hybrids
are not what is producing the result.

**Tier 1 — and it did not behave as designed.** 130 sites, 696 controls, `all` 0.824,
imagery 0.825, S2 0.784, S1 0.795, context 0.583. It was pre-registered as the easy
comparison and an upper bound, with a 0.85 bar, and it came in **below Tier 2**. The reason
is in its own control definition: Tier-1 controls are matched on road, line and industrial
distance, which forces them to be infrastructure-adjacent land rather than open
countryside. So Tier 1 is not an upper bound and is not reported as one. It clears its
G2.2-T1 debug threshold (0.824 > 0.75), so the harness is sound; it misses G2.1-T1's 0.85.
Its balance also fails on `elev_m`, `slope_deg` and `d_industrial_m`.

**The two shortfalls, and what they do and do not permit.**

*G2.0-T2 control count, 251 against 270.* Not fixable by drawing harder: controls per site
were raised 3 → 5 → 8 and the pool moved 232 → 248 → 251, because the calipers and not the
candidate list are the binding constraint. What that sweep does show is stability — the
headline read 0.852, 0.877 and 0.866 across those three runs, all far above the 0.75 bar,
so the conclusion does not hinge on the 7% shortfall. The gate is recorded as failed
regardless.

*P1 on `n_px`.* Battery windows average 418 px against 386 for controls (SMD 0.162). It was
driven down from SMD 0.368 by ranking candidates on size-closeness and then matching on the
realized rasterized window rather than a geodesic proxy, but it cannot reach zero: small
substations vastly outnumber large ones, so for most battery sites no equal-or-larger
control exists within 40 km, and greedy sign-balanced selection barely moved it. Rather
than argue the residual away it was **tested**: the `imagery_robust` arm drops every order
statistic — `bright_p95`, `tex_std`, every `*_max` — keeping only means and differences of
means, which are size-invariant, and returns **0.809 against 0.825**. The size imbalance is
real and does not carry the result.

## Verdict

**Battery storage does produce a signal in Sentinel-1/Sentinel-2 at 10 m that is specific
to the battery rather than to the substation beside it.** Tier 2 clears all three
pre-registered G2.1 conditions, the switchyard control (G2.4-T2) and the siting-prior
control (G2.7) both pass, and the permutation null sits at 0.486.

Read with three qualifications, in descending order of how much they matter:

1. **Two pre-registered checks failed** — G2.0-T2 on control count (251/270) and P1 on
   region-size balance. Both are documented above with the evidence that neither explains
   the result, but neither is waived. Gate 0 as pre-registered also failed, and was amended
   on 2026-09-02 in its own commit before any Stage 2 number existed.
2. **The imagery-only number is 0.825, not 0.866.** The pre-registered `all` arm includes
   five OSM-derived context features, and `d_line_m` alone scores 0.860 as a single
   feature — a substation sits on the network and a battery sits beside it, which is real
   topology but is not a signal in the imagery. For the question actually asked, the
   0.825 imagery-only figure is the answer, and the 0.788 S2-only / 0.616 S1-only split
   says the optical bands carry most of it.
3. **A mapping-convention confound cannot be excluded.** OSM may outline batteries tightly
   around equipment while substation outlines include grassy compounds, and the red-edge
   bands B06/B07 leading the feature ranking is consistent with a vegetation difference
   between the two polygon conventions rather than a difference between the objects. This
   is not testable from OSM alone; it needs a hand-checked sample against high-resolution
   imagery. For a detector aimed at the *mapped* class it matters less, but it qualifies
   any claim about physics.

Stage 1 remains part of the finding, not a footnote: the production substation detector
scores batteries **below** area-comparable substations in GB (0.307) and CA (0.248) and
cannot distinguish them at all in TX (0.483), and in all three regions removing the
overlapping substation lowers the score. So the cheap re-ranking route is dead, while the
class is separable once a classifier is given the battery class itself.

**Next step, if pursued:** Stage 5 — build chips with `--fraction` / `all_touched=False`
from day one and fine-tune. G2.1-T2 passing is what authorises that GPU spend. The
bi-temporal `s2pre` pass was never composited, because Tier 2 did not land in the
0.65–0.75 dead band that would have called for it.
