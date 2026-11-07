# v18: four-date S2, S1 on the full corpus, a training recipe without the epoch lottery, a second test region

**Status:** pre-registered 2026-11-05, before any v18 training or inference. Results are appended
under `## Results`. The decision rules below are not to be moved after seeing data.
2026-11-07: first verdict in, `s2_recipe` not adopted for S2 (fails non-inferiority on
sindh_test); the other arms are still running.

## Why these four

Ranking is already strong (the line-proximity prior took P@20 from 0.70 to 0.90); finding the
substations is the weak part. Production (`v12_up2` S2 + `v5_s1only` S1, mean fusion) finds 54 of
the 112 sindh_test substations of >= 1,000 m2 in its rasters, with recall 0 / 0.26 / 0.70 / 0.71 in
the 1-2k / 2-5k / 5-20k / >20k m2 buckets. Four levers have evidence behind them and have never been
pulled at full scale:

1. **Four single dates instead of one median composite (S2).** The median throws signal away in
   exactly the 2,000-20,000 m2 range: with the production model, peak probability inside real
   substations rose ~130x (2-5k) and ~2x (5-20k) when the four dates were scored separately and
   maxed; a matched retrain on 1,332 chips lifted sindh_test AUC 0.53 -> 0.70 and hits 9 -> 33
   (memory `subdetect-v14-multitemporal`). The full four-date corpus (27,965 chips,
   `chips_multidate_v11b`, 2026-08-06) was built and never trained on.
2. **S1 trained on the corpus S2 got.** S1 is the stronger arm (v5_s1only alone: AUC 0.796,
   P@50 0.42, 52 hits) but trains on 4,730 Pakistan/India chips; S2 added 26,496 TorchGeo chips
   and 2,600 hard negatives. Every change in data quantity so far has moved results (v7, v9).
3. **A training recipe that does not pick an epoch on 289 validation chips.** val/IoU_1 swings
   0.06-0.26 between neighbouring epochs and best epochs range 4-25; a reseed has moved P@50
   0.44 -> 0.22. earthpv's v22 recipe (fixed budget, 1,000 warmup steps, cosine to 1e-6, weight
   EMA, radiometric jitter, final weights as the model) ended the same lottery there: two seeds
   tied each other and the baseline.
4. **A second test region.** sindh_test holds 9 / 39 / 50 / 14 substations per bucket, so two or
   three hits move every metric.

## Leakage found while setting up

155 training chips of the v16 corpus lie inside the two test regions' cells (2.5 km pad): 34
Pakistani chips in sindh_test cells west of the 67 E val bbox, 110 TorchGeo chips (TorchGeo is
centred on real substations, 26 of them inside the Sindh val bbox) and 11 settlement hard
negatives. Every model up to v17 trained on them. **v18 arms drop them**, so a comparison of a v18
arm against v17 or production is conservative for v18; v18-vs-v18 comparisons are clean.

## The second region: gujarat_test

North Gujarat, India (Osmose `india_gujarat`, bbox 71.3-74.5 E, 23.0-24.7 N), built by
`scripts/build_test_region.py` with osmose_detect's own steps: 224 open "unfinished major power
line" endpoints more than 700 m from a mapped substation, every 0.1 deg cell within 10 km (287
cells), S2 + S1 composites on the production windows (2025-11-01 to 2026-03-15). Ground truth:
870 OSM substation polygons and 2 nodes pulled from Overpass on 2026-11-05
(`data/labels/gujarat_test`). Not in any training corpus once the leak chips are dropped (the
India pilot stops at 27.4 N). Same semi-arid dry-season domain as Sindh, different country and
mappers.

## Arms

All v18 training: v22 recipe, seed 1, Tversky 0.35/0.65/1.33 (unchanged), terramind_v1_tiny,
`drop_page_cache` (the oomd guard). Configs `configs/terramind_sub_v18_<arm>.yaml`, written by
`scripts/build_v18_indexes.py`.

| arm | data | control | question |
|---|---|---|---|
| `s2_recipe` | v16 corpus, single-date, 30 epochs | v17 S2 ctl seeds 0, 1 | does the recipe hold or improve S2? |
| `s1_recipe` | v5 S1 corpus, 40 epochs | v17 S1 ctl seeds 0-2 | does the recipe hold or improve S1? |
| `s2_1d` | the 27,818 v16 rows with a 4-date stack, single date, batch 4 x 4 | (control) | |
| `s2_4d` | same rows, 4 dates, TemporalWrapper max | `s2_1d` | do four dates beat one composite? |
| `s1_big` | `s1_recipe` + the 2,596 hard negatives + a random 6,000 of the 26,496 TorchGeo chips, with fetched S1 | `s1_recipe` | does S1 gain from ~3x the data? |

**Amendment before any data (2026-11-06 00:30):** the S1 fetch is bandwidth-bound on the shared
link (one chip, measured: 58 s with 4 scenes, 82 s with 8, 264 s with 16; production composites
take up to 16), so `s1_big` gets S1 from the median of 8 scenes for all hard negatives and a fixed
random 6,000 TorchGeo chips (`fetch_s1_chips.py` order, `run_v18_s1fetch.sh` TARGET 8600) instead
of all 29,092. The rule below is unchanged.

Plus one no-training check, **datemax**: production `v12_up2` run on each of the four cached
dates and maxed, against production on the composite.

TorchGeo chips are stored south-up under a north-up transform; the 4-date stacks and the fetched
S1 are north-up on that transform. `s2_4d` points TorchGeo rows at `mask[::-1]` copies (checked:
TorchGeo 4-date images correlate 0.3-0.85 with the flipped stored image and ~0 unflipped;
Pakistani chips 0.7-0.98 as stored). Fetched S1 is flipped to the stored orientation before it is
written (`scripts/fetch_s1_chips.py`).

## Evaluation and decision rules

`scripts/field_eval.py` (eval_version 2) on **both** regions: sindh_test against
`data/labels/pakistan`, gujarat_test against `data/labels/gujarat_test`. Primary metric per region:
candidate AUC of the deployed mean fusion with the production partner arm held fixed (an S2 arm
with `v5_s1only`, an S1 arm with `v12_up2`). Secondary: arm-alone AUC, P@20, P@50, hits, recall
by area bucket. Paired difference d = arm - control per region.

- **datemax**: adopted for inference if fused AUC d >= +0.02 in both regions and P@50 d >= 0 in
  both. Otherwise not adopted.
- **Recipe** (`s2_recipe`, `s1_recipe`): adopted as the training standard unless its fused AUC
  is more than 0.02 below the old recipe's seed mean in either region (non-inferiority: the
  recipe is for lower noise, not for a gain).
- **Four dates** (`s2_4d` vs `s2_1d`) and **S1 data** (`s1_big` vs `s1_recipe`):
  PASS = d >= +0.02 in both regions and P@50 d >= 0 in both; FAIL = d <= -0.02 in both;
  NEUTRAL otherwise ("no detectable effect at this power"). One seed per arm: a PASS earns a
  second seed pair before anything is promoted, and promotion also needs the arm to beat
  production's own fused numbers in both regions.

Cells without four clear dates fall back to their composite (datemax: one date; `s2_4d`: T = 1),
listed in `<region>/dates/fallback.txt` and reported.

## Results

### `s2_recipe` on sindh_test (2026-11-07): NOT ADOPTED for S2

**Run.** `configs/terramind_sub_v18_s2_recipe.yaml`, seed 1, trained 2026-11-06 00:22 to 20:55 to
its full budget (30 epochs); the model is `data/models/v18_s2_recipe/last.ckpt`, the final EMA
weights. systemd-oomd killed the unit at epoch 10 (10:23, memory pressure from the other units);
it resumed from `last.ckpt`, which carries the EMA state. Scored 2026-11-07 10:20 by
`field_eval.py` eval_version 2 against `data/labels/pakistan` (174 polygons + 7 node discs;
9 / 39 / 50 / 14 substations in the 1-2k / 2-5k / 5-20k / >20k m2 buckets), run names
`v18_sindh_s2_recipe` and `v18_sindh_s2_recipe_x_v5`.

**Primary metric: fused AUC with `v5_s1only` held fixed.**

| fused with `v5_s1only` | AUC | P@20 | P@50 | hits | candidates | recall 1-2k / 2-5k / 5-20k / >20k |
|---|---|---|---|---|---|---|
| **v18 `s2_recipe`** | **0.828** | 0.85 | 0.60 | 47 | 531 | 0 / 0.23 / 0.56 / 0.71 |
| v17 S2 ctl seed 0 | 0.869 | 0.95 | 0.68 | 48 | 680 | 0 / 0.23 / 0.60 / 0.71 |
| v17 S2 ctl seed 1 | 0.876 | 0.95 | 0.68 | 48 | 629 | 0 / 0.23 / 0.60 / 0.71 |
| v17 S2 ctl, seed mean | 0.873 | 0.95 | 0.68 | 48 | | |
| production (`v12_up2` + `v5_s1only`), reference | 0.868 | 0.85 | 0.74 | 54 | 650 | 0 / 0.26 / 0.70 / 0.71 |

d (fused AUC, arm - old-recipe seed mean) = **-0.045**, more than the 0.02 non-inferiority margin.
The rule fails the recipe if it fails in either region, so the gujarat_test score cannot change
the verdict; the runner still scores `s2_recipe` there once that region's date stacks exist
(step B), and it will be added here for the record.

**Secondary: the arm alone.** The loss is larger without the S1 partner and shows in every
bucket:

| arm alone | AUC | P@20 | P@50 | hits | candidates | recall 1-2k / 2-5k / 5-20k / >20k |
|---|---|---|---|---|---|---|
| v18 `s2_recipe` | 0.658 | 0.05 | 0.20 | 24 | 251 | 0 / 0.05 / 0.34 / 0.43 |
| v17 S2 ctl seed 0 | 0.834 | 0.05 | 0.12 | 36 | 608 | 0 / 0.08 / 0.52 / 0.57 |
| v17 S2 ctl seed 1 | 0.870 | 0.05 | 0.44 | 35 | 438 | 0 / 0.08 / 0.50 / 0.57 |

**What this result cannot separate.** The v17 controls trained on the 155 leak chips (34 of them
Pakistani chips in sindh_test cells), and v18 does not, as stated under "Leakage found while
setting up". Some of the -0.045 may be the leak rather than the recipe; this run does not measure
how much. A clean split would need the old recipe retrained on the v18 (leak-free) S2 index,
which is not part of this pre-registration. The verdict above stands under the rule as written.

**Consequences for the other arms.** None of the pre-registered comparisons change: `s2_4d` is
judged against `s2_1d` (both on the new recipe and the leak-free index), `s1_big` against
`s1_recipe`, and `s1_recipe` gets its own non-inferiority test against the v17 S1 controls.
Production S2 stays `v12_up2`.
