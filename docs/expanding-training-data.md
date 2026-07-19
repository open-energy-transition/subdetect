---
title: Expanding the training data
---

# Expanding the training data

The v6–v8 experiments (see [Model lineage](model-lineage.md#v6v8-chasing-the-s1-arm-with-data-not-architecture))
established that, for this model at this scale, **data quantity beats label
purity**. This page documents the two data-expansion pipelines built on that
finding — merging in an external public dataset, and mining hard negatives
from well-mapped regions worldwide — as runnable, copy-pasteable walkthroughs,
not just a description. Both are ordinary local scripts (no new CLI commands),
run directly with `pixi run -e ml python scripts/<name>.py`.

## 1. Merging in the TorchGeo `Substation` dataset

[TransitionZero's power-substation dataset](https://github.com/Lindsay-Lab/substation-seg)
(also published as `torchgeo.datasets.Substation`; paper:
[arXiv:2409.17363](https://arxiv.org/abs/2409.17363)) is 26,522 global
Sentinel-2 locations with binary substation masks — a 5–6× increase over this
project's entire S2 training set, and the natural fix for the S2-only arm's
data starvation (see [Model lineage](model-lineage.md#v9-growing-the-data-starved-s2-only-arm)).
It ships as two flat directories rather than this project's per-AOI GeoTIFF
chip layout, so it needs a conversion step before it can be merged in.

**Format, reverse-engineered from the actual `torchgeo` source** (the
dataset's own README documents neither of these):

- `image_stack/lat_<LAT>_lon_<LON>.npz`, key `arr_0`: `(revisits, 13, 228, 228)`
  float64, standard Sentinel-2 band order (confirmed against
  `torchgeo.datasets.Substation.rgb_bands = (3, 2, 1)`, i.e. B04/B03/B02 at
  those indices → bands are B01..B12 at indices 0..12).
- `mask.tar.gz` → `mask/lat_<LAT>_lon_<LON>.npz`, key `arr_0`: `(228, 228)`
  uint8; **class 3 = substation** (confirmed against `torchgeo`'s
  `__getitem__`: `mask[mask != 3] = 0; mask[mask == 3] = 1`).
- No Sentinel-1 companion — this data only ever feeds an S2-only (or the S2 half
  of a fusion) training run.

```bash
# 1. Convert the two flat directories into subdetect's chip format (median over
#    revisits, subsetted to the same 10 bands/order as LOCAL_BANDS, one UTM
#    GeoTIFF pair per location — same conventions as chips.py's own output).
#    ~20 min for the full 26,522 locations at --workers 10; use --limit for a
#    quick smoke test first.
pixi run -e ml python scripts/build_chips_from_substation_ds.py \
    --src /path/to/torchgeo_substation_dir \
    --out data/chips_torchgeo/substation_global \
    --workers 10                      # add --limit 200 to test first

# 2. Merge with the existing v5 combined index into a new training index.
pixi run -e ml python scripts/merge_chip_index_v9_s2only.py
# -> data/chips_v9/combined/index.parquet

# 3. Train (identical architecture/loss to v5_s2only; only data.index_path differs).
pixi run -e ml subdetect train --config configs/terramind_sub_v9_s2only.yaml
```

**Known trade-off, worth knowing before you merge:** every TorchGeo location
was curated around a real substation, with zero accompanying hard-negative
sampling. Merging it in pushed the combined index to **94.5% positive chips**
(29,548/31,252) — up from v5's more balanced ~65%. This is what motivated the
hard-negative mining pipeline below; if you're reproducing this without also
running that step, expect the model to over-predict positives more than v5 did.

**Evaluate the result against the field, not val/mIoU** — the val split
changed with the index rebuild, so `val/mIoU` is not comparable to any prior
run (see [Model lineage](model-lineage.md#fixing-the-evaluation-itself)):

```bash
pixi run -e ml subdetect infer --aoi pakistan --checkpoint <new checkpoint> \
    --out-dir data/predictions_v9
pixi run -e ml python scripts/field_eval.py --run-name v9_s2only \
    --prob-dir data/osmose_regions/sindh_test_v9/prob --labels pakistan
```

## 2. Mining global hard negatives from well-mapped regions

`scripts/build_global_hardneg_chips.py` counters exactly the imbalance above:
it samples chips from places where OpenStreetMap's `power=substation` coverage
is dense enough to be *trustworthy ground truth* — major world cities and
large open-water bodies — confirms via a live Overpass query that no known
substation falls near the sampled point, and writes those chips with an
all-background mask.

**Why per-candidate clearance, not a per-city veto.** An earlier version of
this script rejected an entire city outright if *any* substation existed
anywhere in its 0.1° cell. That gave zero chips from every major city tested:
Paris alone has 5,339 real (mostly small distribution) substations mapped
within a single cell; Amsterdam has 816. The fixed version fetches a city's
substations once, then keeps only candidate chip centers **≥1,600 m from every
one of them** — bigger than a 224 px (2.24 km) chip's half-diagonal, so the
substation itself provably can't appear in frame — and still finds clean chips
in a city's parks, rivers, and residential fringes. Cities that are dense
*everywhere* (Paris, Amsterdam, London, Copenhagen — 12 of the 65 tried
locations) correctly still yield nothing; that's the check working, not a bug.

```bash
pixi run -e ml python scripts/build_global_hardneg_chips.py \
    --out data/chips_global_hardneg \
    --chips-per-cell 6 --workers 3
# -> data/chips_global_hardneg/index.parquet
```

`--workers` controls compositing concurrency (Overpass itself is always
serialized internally regardless of this value — it rate-limits hard, and
running it concurrently just produces 429s and wastes retries). 3 is a safe
default on modest hardware; the run this page's numbers came from used 8
initially and it visibly starved the machine (each `annual_composite` call
already spins up its own internal concurrent COG reads, so outer×inner
concurrency compounds fast) — dropped back to 3 and it ran cleanly.

**Result from the 65-location list shipped in the script** (45 cities across
every inhabited continent + 20 large lakes/inland seas): **299 chips from
53/65 locations** (254 train / 45 val). Extend or replace `CITIES`/`WATER` in
the script to sample a different set of locations — each is just a
`(name, lat, lon)` tuple.

To fold these into a training run, merge the index the same way
`merge_chip_index_v9_s2only.py` does (concatenate parquets, keep the `aoi`
column for provenance) and point a config's `data.index_path` at the result.
This hasn't yet been merged into a training run as of this writing — see
[Model lineage](model-lineage.md#v9-growing-the-data-starved-s2-only-arm) for
current status.

## 3. Keeping stale checkpoint references from corrupting the next decision

Both pipelines above are only worth running against the checkpoint that's
actually best — and `scripts/osmose_detect.py`, `scripts/eval_decision_fusion.py`,
and `scripts/eval_small_subs.py` had all been silently pinned to **v4**
checkpoints since v5 shipped, meaning every v5–v8 comparison in this
project's history was validated against a stale production model. Find the
actual best checkpoint for a given run from its TensorBoard log rather than
guessing from `last.ckpt` (early stopping's `save_top_k` can prune the true
best out from under a naive "most recent file" check):

```bash
pixi run -e ml python3 -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
d = sorted(glob.glob('logs/lightning_logs/version_<N>/events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(d, size_guidance={'scalars': 0}); ea.Reload()
best = max(ea.Scalars('val/mIoU'), key=lambda x: x.value)
print('best step', best.step, 'value', round(best.value, 4))
"
```

then match that `step=` to a filename in `data/models/stageA_<run>/` and
update the `CKPTS`/`ARMS`/`S1_CKPT`/`S2_CKPT` constant at the top of each
script. There's no version-tracking mechanism for "which checkpoint is
production" beyond these hardcoded constants — checking them after every
retrain is a manual step, easy to forget, and the whole point of this section
is that it was forgotten for two full model generations.
