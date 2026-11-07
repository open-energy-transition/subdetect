# Core-only substation labels from Esri imagery: do they make a better S2/S1 detector?

**Status:** pre-registered 2026-10-31 on branch `esri-core-labels`. Stage 0 passed (G0.1-G0.4,
2026-11-01); S1 study finished 2026-11-02: NEUTRAL; S2 study finished 2026-11-05: NEUTRAL, leaning negative (both seeds below control).
Neither study meets PASS; production labels stay. Stage 2 (filter-only arm: drop bad OSM labels,
keep the rest whole) pre-registered and training from 2026-11-05.
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

**G0.3 bloat -- PASS (both studies).** Full corpus at the final settings
(`scripts/run_esri_core_labels.sh`, finished 2026-11-01 02:40):

| | targets | positive px | -> ignore | no core (whole group ignore) |
|---|---|---|---|---|
| India | 1,420 | 492,712 | 59.0% | 128 |
| Pakistan (+2 hardneg) | 1,221 | 182,127 | 43.1% | 28 |
| **S1 corpus (PK+IN)** | 2,641 | 674,839 | **54.7%** (bar 10%) | 156 |
| TorchGeo | 48,720 | 10,226,206 | 51.8% | 16,038 |
| **S2 corpus** | 51,361 | 10,901,045 | **52.0%** (bar 5%) | 16,194 |

The full corpus trims harder than the 36 dev targets suggested (India 59% vs 38% on dev).
The "no core" targets are almost all tiny: 92% of TorchGeo targets < 2,000 m2 get no core
(Europe/Africa 56% of all its targets), vs 3% at 6-20k m2 and 1% above; PK/IN < 2,000 m2: 36%.

**G0.4 visual QC -- PASS.** 60 frozen targets (`qc_ids.txt`, drawn before grading, none used
by the probe), sheets `data/esri_core/qc/g04/`, per-panel grades in `grades.csv`. Graded by
Claude (not blind to the method, as pre-registered): **44 good, 14 acceptable, 2 harmful**
-> good+acceptable 97% (bar >= 70%), harmful 3.3% (bar <= 15%). The two harmful cases:

- #49: an offshore platform. Esri renders open water, the probe finds nothing, and the whole
  S2-visible label becomes ignore. Coverage >= 0.7 does not catch this (the tile is valid).
- #52: a 350,700 m2 switchyard fetched at 2 m. Upsampled 4x, the core is fragmentary and
  roughly half of the real equipment goes to ignore. Factor-5 (2 m) targets are 1,212 of
  51,360 and lose 59% of their pixels vs 46% for 0.5 m crops.

Not acted on: the gate passed, so the method is not retuned on the QC sample.

**Supplementary check (outside the pre-registered gate).** The frozen QC sample turned out
to be skewed to large sites (TorchGeo < 2,000 m2: 1 of 30 vs 33% of the population; smallest
QC target 1,500 m2), so G0.4 says nothing about the largest effect by target count: the
16k tiny TorchGeo targets that lose their label entirely. 36 random "no core" targets
< 2,000 m2 (24 TorchGeo, 12 PK/IN; `supp_small_nocore_ids.txt`, sheets
`qc/supp_small_nocore/`): TorchGeo ones are 100-500 m2 polygons in European towns on roads,
parking, warehouse roofs, fields, or at most a kiosk-sized building; PK/IN ones are urban
rooftop / single-pixel polygons. None shows switchyard equipment; at 1-5 S2 pixels they are
label noise. 0/36 graded harmful. So for the S2 study "core labels" also means "kiosk-type
labels that S2 cannot see become ignore" -- ~0.4% of positive pixels, but a third of
TorchGeo objects. Worth remembering when reading the S2 result.

### Stage 1

S1 study launched 2026-11-01 02:56 (`subdetect-v17-s1`, `scripts/run_esri_core_study.sh s1
"0 1 2"`). It shares the GPU lock with earthpv's jobs, so it starts after earthpv v24
finishes training. Five of six runs trained 02:56-12:48. On 2026-11-01 14:59 systemd-oomd killed
the unit (session-wide memory pressure; it was only waiting on the GPU lock for core seed 2) and
the queued S2 study started in its place. Relaunched 2026-11-02 08:17 as `subdetect-v17-s1b`;
the runner now skips runs its log marks finished, so only core seed 2 trains before inference.

S2 study queued 2026-11-01 07:24 (`subdetect-v17-s2`, `scripts/run_esri_core_study.sh s2
"0 1"`, log `data/esri_core/run_s2.log`). It waits for the S1 unit to exit, then trains its
four runs under the same GPU lock. Partner arm for fusion = production S1 `v5`.

**Hard crash, 2026-11-02 ~13:55.** The machine went down with no clean shutdown (cause unknown:
no Xid/MCE in the journal, whose last 5 minutes were lost; memory fine; no kdump), about 30
minutes after S1 inference began sharing the GPU with S2 ctl seed 1 training under a private
lock. All S1 training was complete. Effect on the study:

- **S1:** `sindh_test_v17_s1_ctl_seed1/prob` was being written; 13 of 145 cells were lost (8 of
  them 0-byte tifs that `infer` would have skipped as done). The whole arm was deleted and
  regenerated. `_write` is now atomic and fsynced. The other rasters, checkpoints, composites and
  `field_eval.csv` were read back and are intact.
- **S2 ctl seed 1 resumes from epoch 5** (`last.ckpt` intact). **Deviation from the design:** its
  data order after epoch 5 no longer matches its core partner's, because a resume re-seeds rather
  than restoring RNG state. Same init and labels-only difference still hold, and the decision
  rule is unchanged. The other seven runs are unaffected.
- Relaunched 14:11 as `subdetect-v17-s1d` and `subdetect-v17-s2b` (S2 waits for S1), both back on
  the shared GPU lock, so the GPU jobs no longer overlap.

**S2 oomd kill, 2026-11-03 06:59.** systemd-oomd killed `subdetect-v17-s2b` during core seed 1
(user-session memory pressure 82% > 50%; the unit was the top reclaimer at 18.6 GB, mostly chip
page cache; earthpv's Spain compose was running beside it). Ctl seed 0, core seed 0 and ctl seed 1
had finished (best epochs 15, 8 and 24). Nothing ran for 34 h. Relaunched 2026-11-04 17:07 as
`subdetect-v17-s2c` with `Restart=on-failure`. Training and inference are idempotent, and
`esri_core_ab.py` keeps the last row per run. **Deviation:** core seed 1 resumes from `last.ckpt`
(epoch 7), so like its ctl partner its data order no longer matches after the resume. Seed 0 is
the only fully paired S2 seed. The decision rule is unchanged.

### S1 study result (2026-11-02): NEUTRAL

Primary metric = AUC of the mean fusion with production S2 `v12_up2`, on `sindh_test`, raw OSM labels:

| seed | fused AUC ctl | fused AUC core | d | P@50 ctl -> core | hits ctl -> core | candidates ctl -> core | arm-alone AUC ctl -> core |
|---|---|---|---|---|---|---|---|
| 0 | 0.909 | 0.907 | -0.002 | 0.74 -> 0.70 | 54 -> 52 | 690 -> 640 | 0.793 -> 0.797 |
| 1 | 0.856 | 0.878 | +0.022 | 0.74 -> 0.72 | 61 -> 51 | 1114 -> 614 | 0.843 -> 0.732 |
| 2 | 0.852 | 0.885 | +0.033 | 0.76 -> 0.70 | 56 -> 52 | 643 -> 732 | 0.692 -> 0.735 |
| mean | 0.872 | 0.890 | **+0.018** | -0.04 | -5.3 | -154 | -0.021 |

The PASS rule fails on all three conditions: mean d is below +0.02, seed 0 is negative, and mean
P@50 fell. FAIL does not apply either (mean d is positive). Per the pre-registered wording this is
"no detectable effect at this power", not evidence of absence.

Secondary metrics, descriptive only: core puts more of its predicted area inside OSM polygons in
every seed (fused share +0.06 to +0.10, e.g. 0.219 -> 0.322 on seed 0), which is the localisation
the labels aim for. It also loses recall on the largest substations (>=20,000 m2, fused: -0.143 on
seeds 1 and 2, unchanged on seed 0; arm alone: lower on all three) and finds fewer hits in all
three seeds. Reseed noise is as large as the effect: control arm-alone AUC alone spans 0.69-0.84.

### S2 study result (2026-11-05): NEUTRAL, both seeds negative

Core seed 1 finished 2026-11-04 19:28 (best epoch 4, stopped at 14). Inference, field_eval and
`esri_core_ab.py` finished 2026-11-05 06:12, with no further oomd kill. Primary metric = AUC of the
mean fusion with production S1 `v5_s1only`, on `sindh_test`, raw OSM labels:

| seed | fused AUC ctl | fused AUC core | d | P@50 ctl -> core | hits ctl -> core | candidates ctl -> core | arm-alone AUC ctl -> core |
|---|---|---|---|---|---|---|---|
| 0 | 0.869 | 0.859 | -0.010 | 0.68 -> 0.64 | 48 -> 47 | 680 -> 819 | 0.834 -> 0.831 |
| 1 | 0.876 | 0.852 | -0.024 | 0.68 -> 0.64 | 48 -> 47 | 629 -> 589 | 0.870 -> 0.783 |
| mean | 0.873 | 0.856 | **-0.017** | -0.04 | -1 | +50 | -0.045 |

PASS fails (mean d is negative). FAIL needs mean d <= -0.02 with every seed negative: both seeds
are negative, but the mean misses the bar by 0.003. So the verdict is NEUTRAL by the rule, and every
reading points the same way. Arm alone, core is worse on both seeds: AUC -0.045, P@50 0.12 -> 0.02
and 0.44 -> 0.24, recall lower in every area bucket. Unlike S1, the core S2 arm does **not** put
more of its predicted area inside OSM polygons: the inside share falls on both seeds (arm alone
-0.017, fused -0.017). For reference, production `v12_up2_mean_fusion` scores 0.868 / P@50 0.74
under eval_version 2; the control arms here match it and the core arms fall short.

Descriptive: the core arms hit their best validation score early (epochs 8 and 4, against 15 and
24 for control). Validation is scored on each arm's own labels, so this is not a field result.

### Conclusion

Neither study passes. S1 is neutral with a small, seed-unstable AUC gain (+0.018), lower P@50 and
fewer hits. S2 is neutral, leaning negative (-0.017, both seeds negative). Core-only labels from
Esri imagery do not give a better detector at this power, and production keeps the full OSM burn.
The label-quality gates all passed, so the null is not caused by bad trimming. Discarding about
half of all positive pixels (the ignored property land, plus 16k tiny targets) costs more
supervision than tighter outlines give back. That matches v6 and v7: in this project, data
quantity has been the recall lever.

## Stage 2: filter-only arm (pre-registered 2026-11-05, before training)

**Why.** The owner reviewed the core labels on the review page (`data/esri_core_review/`, built by
`scripts/esri_core_review_page.py`) and judged that the trimmed labels do not look good. The new
question is narrower: is the probe good enough to find the OSM labels that are *wrong*, if we drop
those and keep every other label as its full OSM burn?

**Rule.** Per target, core share = core_px / orig_px from the Stage 0 rewrite (`core_stats.parquet`,
thr 0.76, dilate 0). A target with core share < 0.10 has its whole group set to ignore (-1). This
covers every `no_core` target. All other targets keep the control's OSM burn untouched; `no_imagery`
targets are kept. Script: `scripts/esri_core_filter_labels.py`. The only difference from the control
arm is which targets are ignored.

**Why 0.10.** It was chosen before training, from a visual check rather than tuning: 24 random targets
of >= 6,000 m2 with core share < 0.10 (sheets in the session scratchpad, ids drawn with
random_state 3). About 20 of 24 sit on forest, houses, fields, construction sites, factory roofs or
empty land, and one is offset beside a real switchyard. One or two may be indoor (GIS) substations,
which a 10 m model cannot see as a switchyard anyway. The `no_core` targets < 2,000 m2 were already
checked in Stage 0's supplementary check (0/36 show equipment).

| corpus | targets dropped | positive px -> ignore |
|---|---|---|
| S2 (`data/chips_v17_filter`) | 18,071 of 51,361 (TorchGeo 36.6%, India 13.7%, Pakistan 3.5%) | 6.4% |
| S1 (`data/chips_v5_filter`) | 239 of 2,641 | 7.8% |

By target size (S2 corpus): 92% of targets < 2,000 m2 are dropped, 23% at 2-6k, 8% at 6-20k, 6% above.

**Design.** Arm `filt`, configs `terramind_sub_v17_{s1,s2}_filt_seed{N}.yaml` (control config with only
`index_path` and `checkpoint_dir` changed). It is paired by seed with the already-trained and evaluated
control runs (S2 seeds 0-1, S1 seeds 0-2), so only the filt arm trains. The S2 ctl seed 1 resume
deviation from Stage 1 carries over. Run order: S2 first, then S1 (`ARMS=filt`, logs
`data/esri_core/run_{s2,s1}_filt.log`). Evaluation and decision rule are unchanged from Stage 1
(`esri_core_ab.py --arm filt`): fused AUC on `sindh_test` against raw OSM polygons;
PASS = mean d >= +0.02, every seed > 0, mean P@50 difference >= 0; FAIL = mean d <= -0.02, every
seed < 0; NEUTRAL otherwise.

One point to read the result against: `sindh_test` scores against raw OSM polygons, so a model
that learned to skip the label types this filter drops (tiny kiosks, misplaced polygons) is not
rewarded for it there. Where Pakistan's own OSM labels are of those kinds, it is penalised.
