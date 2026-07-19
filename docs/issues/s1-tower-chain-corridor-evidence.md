# Corridor evidence from Sentinel-1 tower chains (persistent-scatterer convergence prior)

> Status: idea documented 2026-07-19; mapped-tower baseline validated the same day
> (see "Evidence so far"). File as a GitHub issue — `gh` is not installed on the
> analysis machine, so this lives in-repo for now.

## Problem

Field validation of the v9 leads (2026-07-19) showed the dominant false-positive
classes are **villages, construction sites and bare land**. A village can mimic a
substation's local texture at 10 m; it cannot mimic the grid's *topology*: a
transmission substation is a node where several tower chains converge and
terminate. The current ranking uses distance to **mapped OSM lines** only
(`line_dist_m` in `postprocess._rank`) — blind exactly where the highest-value
leads are, on **unmapped corridors** (the Osmose "line ends nowhere" premise).

## Physical signal

- A steel lattice tower is a stack of corner reflectors — a bright, temporally
  stable point target in S1 VH/VV (a classic persistent scatterer).
- A single tower is only ~1–2 px at 10 m and individually unreliable; but towers
  come in **chains with regular 250–400 m spacing along straight corridors**,
  detectable via alignment (Hough/RANSAC over bright-point sets), which village
  clutter cannot reproduce over kilometres.
- Substations are where ≥2 chain bearings meet: a **convergence count** is a
  direct anti-village discriminator.
- Chains operate at 1–5 km scale — beyond the model's 224 px (2.2 km) receptive
  field — so this must enter as a precomputed raster channel or ranking feature;
  the U-Net cannot learn it from chips.

## Evidence so far (mapped-tower baseline, sindh_test)

`scripts/tower_features_eval.py` on the v9 mean-fusion candidates (234 candidates,
32 hits, 16,878 mapped towers):

| ranking | AUC | P@20 | P@50 |
|---|---|---|---|
| conf_max alone | 0.930 | 0.95 | 0.58 |
| conf × tower-distance prior | **0.985** | 0.95 | **0.62** |
| conf × chain-bearing convergence | 0.978 | 0.95 | 0.60 |
| conf × mapped-line prior (reference) | 0.985 | 0.95 | 0.62 |

Feature-alone separation: `tower_dist_m` AUC **0.965** (beats the model's own
confidence), `n_bearings` 0.870. On this well-mapped region towers ≈ lines, so the
mapped-tower prior ties the line prior; the S1-derived version is what removes the
OSM-completeness dependence.

## Plan

1. **Done** — mapped-tower features (`data/osm/*_towers.geojsonseq`,
   `scripts/tower_features_eval.py`).
2. **Feasibility** (`scripts/s1_chain_detect.py`): are mapped tower locations
   photometrically separable as bright points in the existing S1 composites
   (VH/VV local contrast at tower px vs background)? If not at 10 m from a single
   median composite, revisit with a multi-date stack (temporal max compounds
   persistent scatterers) before abandoning.
3. **Chain detector**: bright-point extraction (local dB contrast) → chain fitting
   (collinearity + spacing regularity, ≥5 points over ≥1.5 km) → per-cell chain
   segments + convergence nodes.
4. **Validation**: precision = detected chains within 100 m of mapped lines;
   recall = mapped-line length covered. Then the money test: re-rank sindh_test
   candidates by conf × chain prior and compare AUC/P@50 against the mapped-line
   prior — and, critically, evaluate on cells whose lines are *hidden* from the
   detector to simulate unmapped corridors.
5. **Deployment options**: extra ranking feature in `postprocess._rank`;
   rasterized distance-to-chain input channel at train time; Osmose lead
   verification (does an S1 chain actually reach the candidate?).

## Results of the first feasibility pass (2026-07-19, `scripts/s1_chain_detect.py`)

Photometry: mapped tower locations ARE brighter than background in the existing S1
composites (3x3-max VH, median +1.8 dB) but weakly separable per point: **AUC 0.702**.
Chain detection on sindh_test (145 cells, single median composite), three settings:

| setting | chains | point precision vs lines | line coverage | re-rank AUC (conf=0.930) |
|---|---|---|---|---|
| 3.5 dB, no isolation filter | 859 | 0.03 | 0.05 | 0.877 (hurts) |
| 5 dB + isolation ≤3 nb/250 m | 1,498 | 0.08 | 0.09 | 0.930 (neutral, P@20 1.00) |
| 7 dB, ≤2 nb, ≥9 pts, ≥2.5 km | 67 | 0.33 | **0.02** | **0.970** (helps) |

Verdict: the signal is real (strict chains add +0.04 AUC over confidence alone) but
**never beats the mapped-line/tower prior (0.985)**, and the precision/coverage
trade-off is brutal — the main confuser is Sindh's canal/road-aligned villages,
which are themselves collinear bright-point chains (the isolation filter helps but
does not solve it). Key suspected limitation: the local S1 composites are temporal
*medians*, which suppress point targets whose brightness fluctuates with look
geometry; persistent-scatterer work wants a **temporal max / coherence over a
multi-date stack**. That is the next experiment before any further tuning of the
detector geometry — and the mapped-tower prior (AUC 0.985, production-ready,
`scripts/tower_features_eval.py`) is the thing to ship meanwhile.

## EHV addendum (2026-07-19, `scripts/ehv_detectability.py`)

Voltage-stratified photometry on sindh_test (towers assigned to nearest mapped line
<= 150 m; 8,389 tower samples vs 6,995 background):

| class | n | median VH contrast | AUC S1-VH | AUC S1-VV | AUC S2 brightness | AUC S2 NDVI | AUC S1+S2 rank fusion |
|---|---|---|---|---|---|---|---|
| **>=400 kV** | 1,632 | **+4.2 dB** | **0.909** | 0.856 | 0.573 | 0.527 | 0.835 |
| 220 kV | 1,369 | +3.2 dB | 0.820 | 0.792 | 0.527 | 0.511 | 0.744 |
| 132 kV | 2,534 | +1.8 dB | 0.633 | 0.615 | 0.523 | 0.462 | 0.607 |
| <=66 kV | 127 | +1.7 dB | 0.632 | 0.691 | 0.417 | 0.488 | 0.544 |

Corridor enrichment (fraction of 100 m line samples with a >=5 dB S1 bright point
within 60 m, vs a 1 km perpendicular-offset control corridor): **>=400 kV: 19.8% vs
5.1% = 3.9x lift**; 220 kV 1.5x; 132 kV 1.2x (noise).

Conclusions:
- **>=400 kV towers are individually detectable in S1** (AUC 0.909) — detectability
  scales with voltage exactly as radar-cross-section physics predicts, and explains
  why the all-tower test gave only 0.702 (132 kV dominates the tower population).
- **S2 sees nothing** (brightness/NDVI ~ chance) — a tower is sub-pixel at 10 m —
  and S1+S2 fusion *dilutes* the S1 signal (0.909 -> 0.835). For towers, S1 is the
  only sensor; don't fuse.
- **Automated line tracing still fails even for EHV**: an EHV-tuned chain pass
  (5 dB, >=7 pts, >=2 km) covers only 8% of >=400 kV line length at 8% point
  precision — Sindh's collinear canal/road village clutter defeats hard RANSAC
  chains on a single median composite, despite the 3.9x corridor enrichment being
  clearly present. Next moves, in order: (1) temporal-max multi-date S1 stack;
  (2) replace hard chain extraction with a probabilistic corridor scorer
  (enrichment vs offset-control along candidate bearings), which uses the 3.9x
  lift directly instead of demanding perfect collinear point recovery.

## Acceptance criteria

- Chain detector precision ≥ 80% vs mapped lines on sindh_test, recall ≥ 40%.
- Held-out (lines-hidden) re-ranking improves AUC over conf_max alone by ≥ 0.02.
- No recall loss for candidates on genuinely unmapped corridors (spot-check the
  Osmose leads that were manually confirmed).
