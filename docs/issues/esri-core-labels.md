# Core-only substation labels from Esri imagery: do they make a better S2/S1 detector?

**Status:** pre-registered 2026-10-31 on branch `esri-core-labels`. G0.1 and G0.2 passed;
full-corpus core labels are being written; G0.3/G0.4 pending; no training number exists yet.
Results are appended under `## Results`. The gates and decision rules below are not to be
moved after seeing data.

## Question

OSM `power=substation` polygons are often drawn round the whole fenced property -- the
switchyard plus empty ground, lawns, parking, offices and staff housing -- not round the
equipment. Every positive pixel of that property is taught to the model as "substation".
If the labels kept only the **core** (the equipment yard, as seen at ~0.5 m in Esri World
Imagery), does a 10 m Sentinel-2 and/or Sentinel-1 detector trained on them find
substations better in the field?

## Prior evidence, and why this is not a re-run of v6

v6/v6b/v6c (2026-07-15/16, memory `subdetect-label-refine-v6`) shrank Pakistan/India
polygons with an S1 backscatter gate + S2 NDVI<0.25, and the result was a mild regression
(more false positives, worse >=20k recall, fragmented candidates). This study differs in
four ways, each chosen against a specific weakness of that one:

1. **The core is seen, not inferred.** v6 had no high-resolution view; bare soil and gravel
   are spectrally the same at 10 m, so "NDVI<0.25" kept whatever was bare.
2. **Non-core property becomes IGNORE (-1), not background.** v6 burned the shrunk polygon,
   so trimmed property became a hard negative. If the trimming is imperfect, that teaches
   the model to *not* fire on real switchyard edges. Ignore only removes the noisy positive
   supervision; it asserts nothing about pixels we are not sure of.
3. **It covers the whole corpus.** v6 refined only the 3.1k Pakistan/India chips. The S2
   corpus here is 29.5k positive chips, 26.5k of them TorchGeo, and those are refined too.
4. **The core detector is learned from the labels' own noise** (below), not hand-tuned
   thresholds.

## Method

- **Imagery:** Esri World Imagery XYZ tiles (z18, ~0.5 m; fallback z17/z16 where Esri has
  only a "Map data not yet available" placeholder), resampled onto an integer sub-grid of
  each chip's own 10 m grid (`scripts/esri_core_fetch.py`). One target per spatially grouped
  positive component per chip (`scripts/esri_core_targets.py`): 51,361 targets.
- **Core detector:** DINOv2 ViT-B/14 (reg4) patch features (~7 m patches) + a linear probe
  trained to separate patches inside the original masks from patches >=10 m outside them,
  pooled over thousands of sites (`scripts/esri_core_extract.py`). The probe can only learn
  what is consistently inside-and-not-outside: the switchyard. Property land that bloated
  polygons sweep in looks like the land outside and scores low despite its positive label.
- **Label rewrite** (`scripts/esri_core_labels.py`): per target, core = probe P >= 0.5
  (cleaned), block-averaged to 10 m; new positive = core fraction >= 0.5 within the original
  positive (dilated 1 px for Esri/S2 offset); original positive pixels that are not core
  become ignore (-1); everything else in the chip is untouched. A target with imagery
  coverage < 0.7 keeps its original label. A target where no core is found becomes ignore.
- **TorchGeo chips are stored south-up** under a north-up transform (found while building
  this study; see `subdetect.core_labels`). All geo operations flip them first and flip the
  rewritten label back, so the corpus stays consistent with its own images.

## Stage 0 gates (label quality) -- must pass before any training

| gate | criterion | if it fails |
|---|---|---|
| G0.1 imagery | >= 90% of targets have Esri coverage >= 0.7 | refine only the covered ones; report |
| G0.2 probe | held-out-site patch AUC (inside vs outside) >= 0.85 | stop: the core detector is not reliable |
| G0.3 bloat | rewrite moves >= 10% of Pakistan/India positive pixels to ignore (S1 study) and >= 5% corpus-wide (S2 study) | that study stops as underpowered: "OSM labels are already tight enough" is the finding |
| G0.4 visual QC | 60 targets never seen during tuning (30 PK/India, 30 TorchGeo), graded good / acceptable / harmful: harmful <= 15%, good+acceptable >= 70% | retune once on the dev sample, re-grade a fresh 60 |

"Harmful" = the core cuts away substantial real equipment, or sits on something that is
not a substation. Grading is done by eye on `scripts/esri_core_qc.py` sheets (Esri with
original outline + core outline, beside the S2 window). It is not blind to the method.
Tuning uses a separate dev sample; the QC sample ids are drawn and frozen before grading.

## Stage 1: matched training arms

Every comparison is **paired by seed**: control and core arm share a seed (same decoder
init, same data order), identical config except the mask paths. `train.py` gains an
optional `seed` key for this (backward compatible).

- **S1 study** (cheap; the production fusion's S1 arm). Recipe = `v5_s1only` (S1RTC only,
  tiny backbone, Tversky 0.35/0.65), corpus `chips_v5/combined` (Pakistan+India), monitor
  `val/IoU_1`. Control = OSM polygons re-burned centre-in (the v16 burn); core = this
  study's rewrite of exactly those masks. **3 seeds per arm.**
- **S2 study** (expensive; ~10-20 GPU h per run on a shared GPU). Recipe = `v16_honestburn`
  (v11b corpus, S2 only) with max_epochs 60->40 and patience 15->10 for GPU budget (v16
  peaked at epoch 20; both arms alike). Control = v16's own index; core = its rewrite
  (OSM + TorchGeo).
  **2 seeds per arm**, a third pair only if GPU time allows -- the decision rule below is
  fixed regardless.

## Evaluation and decision rules

`scripts/field_eval.py` (eval_version 2) on `sindh_test`, labels `pakistan`, which scores
against the RAW OSM polygons -- never against the refined labels, which would hand the core
arm the win by construction. Hits are overlap-based, so a tighter detection inside a polygon
still counts.

Primary metric: candidate **AUC of the deployed mean fusion**, `0.5*(P_S1 + P_S2)`, with
the other arm held at production (S1 study: S2 = `v12_up2_s2only`; S2 study: S1 =
`v5_s1only`). Secondary: arm-alone AUC, P@20, P@50, n_hits, recall by area bucket,
n_fp_proxy, and the fraction of predicted area (P >= 0.5) lying inside an OSM polygon.

Per study, with paired differences d_s = core_s - control_s on the primary metric:

- **PASS** -- core labels help: mean d >= +0.02 AND d_s > 0 for every seed AND mean
  P@50 difference >= 0.
- **FAIL** -- core labels hurt: mean d <= -0.02 AND d_s < 0 for every seed.
- **NEUTRAL** -- anything else. Reported as "no detectable effect at this power", not as
  evidence of absence: the v8 reseed episode moved field AUC by 0.04 on an identical recipe.

## Results

### Stage 0 (2026-10-31)

**G0.1 imagery -- PASS.** 51,360 of 51,361 targets fetched (1 failure); coverage >= 0.7 for
99.996%. Mostly 2025 imagery, z18 (~0.5 m) for 98.5%, z17/z16 placeholder fallback for the rest.

**G0.2 probe -- PASS.** Probe = DINOv2 block 8 patch tokens + logistic regression, near-ring
(10-60 m) negatives weighted 3x, pooled over 3,000 sites (600 each: Pakistan, India,
TorchGeo Americas / Europe-Africa / Asia). The 12 frozen QC targets that the random
probe-site draw happened to include were dropped from probe training (and from the
held-out set), so G0.4 grades only sites the probe never saw. Held-out sites (20%):

| | inside vs all | inside vs near ring |
|---|---|---|
| pooled | **0.889** | 0.878 |
| Pakistan | 0.950 | 0.942 |
| India | 0.879 | 0.870 |
| TorchGeo Americas | 0.907 | 0.896 |
| TorchGeo Europe/Africa | 0.851 | 0.836 |
| TorchGeo Asia | 0.869 | 0.856 |

**Dev-sample retune (36 dev targets, never the QC sample).** Two method parameters changed
from the text above, both decided on the dev sample before any QC grading:

- **Threshold 0.5 -> 0.76.** The probe is fit with balanced class weights and up-weighted
  near-ring negatives, which puts its 0.5 point far too low: at 0.5, 20% of near-ring and
  13% of far patches (PK/India held-out) score as core, and dev sheets showed core outlines
  spreading over whole neighbourhoods. The threshold is set by a rule, not by eye: the
  held-out near-ring false-positive rate = 5% -> t = 0.761 (inside rate 0.58, far 0.025).
- **Dilation 1 px -> 0.** The 1 px "Esri/S2 offset" allowance adds positives wherever the
  core extends past the polygon. On dev it added +26% positive area to subs < 6,000 m2
  (+2.4% for larger ones) -- inflating exactly the small labels the study means to tighten.
  The trimmed (ignore) pixels are identical with and without it.

Dev-sample label statistics at the final settings (core targets only):

| | kept as positive | turned to ignore |
|---|---|---|
| India (12) | 62% | 38% |
| Pakistan (12) | 65% | 36% |
| TorchGeo (12) | 32% | 68% |

TorchGeo trims hardest; on dev this is partly correct (polygons offset from the visible
switchyard keep only the overlap, e.g. dev #28/#32) and partly grass/trees inside fenced
yards. Whether it is too aggressive is what G0.4's TorchGeo half has to answer.

**G0.3 / G0.4:** pending (full-corpus run `scripts/run_esri_core_labels.sh`).
