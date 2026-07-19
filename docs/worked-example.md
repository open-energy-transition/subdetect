---
title: Worked example
---

# Worked example: anatomy of one detection

Everything on this page is real output from the Yunnan pilot run of
`scripts/osmose_detect.py` — no illustrative/mocked data. The cell is
`00977_00242` (a 0.1°×0.1° tile, ~11 km × 10 km at this latitude), one of 153
composited during that pilot.

![Four-panel figure: Sentinel-2 true color, Sentinel-1 false color, predicted probability, detected polygons](assets/worked_example_panel.png)

## A. Sentinel-2, true color (bands B04/B03/B02)

The raw material: a cloud-masked dry-season median composite, 10 m/pixel. You can
see a settlement in the lower-left, agricultural plots (the pale rectangular
grid), and forested terrain along the upper edge. Nothing about this image alone
obviously marks any point as "substation" — which is exactly the problem optical-
only models struggle with (see the model-lineage discussion of the dominant
false-positive class, [bare/exposed land confused for a gravel yard](model-lineage.md)).

## B. Sentinel-1, false color (R=VV, G=VH, B=VH−VV, dB)

The same footprint in radar. The mountain ridge along the upper-left produces
strong terrain-driven backscatter (the orange/red texture — layover and slope
effects, not infrastructure). More useful for our purposes: substation gantries,
transformers, and busbars are **corner reflectors** — small geometric arrangements
of near-perpendicular metal surfaces that bounce radar directly back at the
sensor, producing a bright, spatially tight signature that bare land or crops
cannot replicate. This is *why* the project's best single model uses radar
alone — see [Model lineage](model-lineage.md#the-arc).

## C. Predicted probability

Output of the production stack, `P = P_S1only × (0.5 + 0.5 × P_S2only)`: the
S1-only model's own probability, softly damped (never fully veto'd) by whether
the S2-only model agrees. This cell alone contains **15 separate candidate
components** above the polygonize threshold — most of them sitting at almost
exactly 0.50 probability. That's not a coincidence: any pixel where the S1 model
is fully confident (P_S1 ≈ 1) but the S2 model is neutral (P_S2 ≈ 0) evaluates to
exactly `1 × (0.5 + 0.5×0) = 0.5`. In practice roughly 60% of all Osmose-pilot
candidates land on this "plateau" — see the note on it in
[Model lineage](model-lineage.md#open-problem-the-05-plateau).

## D. Detected polygons

Hysteresis polygonization (seed ≥ 0.4, grow through ≥ 0.2 — see **hysteresis** in
the [glossary](glossary.md)) turns those probability blobs into the outlined
polygons shown in green. The callout box highlights the **highest-confidence**
candidate in this cell specifically:

| | value |
|---|---|
| confidence | 0.82 |
| area | 5.7 ha (57,324 m²) |
| distance to nearest Osmose endpoint | 6,517 m |
| distance to nearest OSM power line | 765 m |

Note this is a *different* candidate from the pilot's overall **rank 1** lead
(confidence 0.50, but sitting *exactly* on both an Osmose endpoint and a mapped
power line — `endpoint_dist_m = 0`, `line_dist_m = 0`, `rank_score = 0.502`),
also present in this same cell. This is the clearest illustration of why the
pipeline ranks leads by `rank_score` rather than raw `confidence` for the Osmose
workflow specifically: a lower-confidence detection sitting exactly where an
unfinished line dead-ends is a stronger *"this explains the Osmose issue"* signal
than a higher-confidence detection 6.5 km away, even though it would lose on
confidence alone. Both orderings are valid depending on what you're looking
for — see the note on `rank_score` vs. `confidence` in the [glossary](glossary.md).

## Try it yourself

Regenerate this exact figure (or point it at a different cell/region):

```bash
pixi run -e ml python scripts/make_docs_assets.py --only worked-example
```

The script's `EXAMPLE_CELL`/`YUNNAN` constants can be edited to target any other
composited cell — everything here is read straight from
`data/osmose_regions/yunnan/composites/00977_00242/` and
`data/osmose_regions/yunnan/prob/00977_00242.tif`, nothing is precomputed or cached
specifically for this page.
