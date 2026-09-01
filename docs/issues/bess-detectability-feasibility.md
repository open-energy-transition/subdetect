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

## Results

*(appended per stage; empty at pre-registration)*

### Stage 0 — labels

Not yet run.

### Stage 1 — production-detector probe

Not yet run.

### Stage 2 — separability

Not yet run.

## Verdict

Not yet reached.
