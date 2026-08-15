"""Re-burn an existing chip index's masks under a different label representation.

Both arms of the 2026-08-15 label-bias experiment are built here. Neither samples a new
corpus: the masks of an existing index (default v11b combined, the corpus behind the
v11b/v12_up2 lineage) are re-burned in place, leaving imagery, chip centres and the
train/val split byte-identical. Each arm is therefore a one-variable A/B against
`v11b_s2only`, which was trained on exactly these chips.

The defect both arms attack. `chips.py::_burn_mask` has always rasterized positives with
`all_touched=True`, which burns any pixel the polygon touches and so inflates a positive by
roughly its perimeter in pixels. Measured on 502 real installations lying entirely inside a
training chip (burned pixel area against true geodesic area):

    bucket            all_touched=True   centre-in   10x supersampled
    1,000-2,000 m2         1.70x           0.98x          1.00x
    2,000-5,000 m2         1.47x           1.00x          1.00x
    5,000-20,000 m2        1.29x           1.00x          1.00x
    >=20,000 m2            1.13x           1.00x          1.00x

Every checkpoint through v14 was trained on a footprint whose bias varies monotonically
with size, worst in exactly the bucket that has read 0% recall in every model to date and
has been attributed to a resolution floor (docs/model-lineage.md).

  --target binary    (v16_honestburn) Re-burn positives with all_touched=False. Removes the
                     bias and NOTHING else: still a two-class mask, so the trained model
                     still emits a probability raster and every downstream threshold,
                     hysteresis seed and fusion formula keeps its meaning. This is the cheap
                     decisive test of "is the small-substation floor partly supervision
                     noise?", and it should be read before the fraction arm.
  --target fraction  (v15_fraction) Re-burn as float32 sub-pixel coverage in [0, 1] for the
                     regression head. Removes the same bias AND adds true sub-pixel targets,
                     but changes the output semantics. Run it against the binary arm, not
                     against v11b: the difference between them is what is attributable to
                     sub-pixel supervision specifically.

Chips whose source polygons are not on disk (`torchgeo_substation`, the hard-negative sets)
cannot be re-burned. Under --target binary they keep their original masks untouched, so the
arm changes only the 3,684 OSM-labelled chips -- which is the whole population the
measurement above describes. Under --target fraction they must at least change dtype, so
their binary masks are cast to float32 {0.0, 1.0}; that leaves ~80% of the corpus supplying
no sub-pixel information, which is why a neutral fraction result needs `--osm-only` before
it means anything.

Usage:
  pixi run -e ml python scripts/build_reburn_chip_index.py --target binary
  pixi run -e ml python scripts/build_reburn_chip_index.py --target fraction
  pixi run -e ml python scripts/build_reburn_chip_index.py --target fraction --osm-only \
      --out-root data/chips_v15_fraction_osm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from subdetect.chips import _write_tif, reburn_mask
from subdetect.config import Settings
from subdetect.local_source import load_substation_labels

ROOT = Path(__file__).resolve().parents[1]
# AOIs whose chips were burned from OSM polygons we still have on disk, and can therefore
# be re-burned. Everything else keeps (binary) or is cast from (fraction) its existing mask.
OSM_AOIS = ("pakistan", "india_pilot")
DEFAULT_OUT = {"fraction": "data/chips_v15_fraction", "binary": "data/chips_v16_honestburn"}


def _cast_to_float(src_path: Path, out_path: Path) -> None:
    """Cast an existing int16 class mask to the float32 fraction convention: 1 -> 1.0
    (fully covered), 0 -> 0.0, -1 -> -1.0 (ignore). No sub-pixel information is created;
    this only puts the chip in the dtype/semantics the regression task reads."""
    with rasterio.open(src_path) as src:
        arr, transform, crs = src.read(1), src.transform, src.crs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tif(out_path, arr.astype("float32"), transform, crs, "float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["fraction", "binary"], required=True)
    ap.add_argument("--index", type=Path,
                    default=ROOT / "data/chips_v11b/combined/index.parquet")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Defaults per --target: see DEFAULT_OUT")
    ap.add_argument("--labels-dir", type=Path, default=ROOT / "data/labels")
    ap.add_argument("--min-area-m2", type=float, default=0.0,
                    help="Label floor used when the source corpus was built (v5+ = 0)")
    ap.add_argument("--prefer-refined", action="store_true",
                    help="Use refined label polygons. OFF by default because the v5/v9/v11b "
                         "corpora were built from raw OSM labels, and --verify enforces it")
    ap.add_argument("--factor", type=int, default=10,
                    help="fraction only: supersampling factor for coverage (10 = ~1 m)")
    ap.add_argument("--verify", type=int, default=12,
                    help="Per OSM AOI, re-burn this many chips the ORIGINAL way "
                         "(binary, all_touched=True) and require an exact match against the "
                         "mask on disk before re-burning anything (0 to skip). Guards "
                         "against label settings that differ from the ones the source "
                         "corpus was built with, which would silently break the A/B")
    ap.add_argument("--osm-only", action="store_true",
                    help="Drop chips with no source polygons (TorchGeo), keeping only "
                         "re-burnable positives + hard negatives")
    a = ap.parse_args()
    out_root = a.out_root or (ROOT / DEFAULT_OUT[a.target])
    fraction = a.target == "fraction"

    index = pd.read_parquet(a.index)
    print(f"source index: {len(index)} chips, {int((index.split == 'val').sum())} val, "
          f"{int((index.sub_pixels > 0).sum())} with substation")
    print(index.aoi.value_counts().to_string())

    settings = Settings.load()
    labels: dict[str, gpd.GeoDataFrame] = {
        aoi: load_substation_labels(a.labels_dir / aoi, min_area_m2=a.min_area_m2,
                                    prefer_refined=a.prefer_refined)
        for aoi in OSM_AOIS if (index.aoi == aoi).any()
    }

    # Prove the label settings reconstruct the corpus before trusting any re-burn.
    if a.verify:
        for aoi, lab in labels.items():
            rows = index[(index.aoi == aoi) & (index.sub_pixels > 0)] \
                .drop_duplicates("chip_id").head(a.verify)
            bad = []
            for _, r in rows.iterrows():
                with rasterio.open(r["mask"]) as src:
                    on_disk = src.read(1)
                mine = reburn_mask(Path(r["mask"]), None, lab, settings.node_ignore_radius_m,
                                   fraction=False, min_area=a.min_area_m2, all_touched=True)
                if not np.array_equal(mine, on_disk):
                    bad.append(r["chip_id"])
            if bad:
                raise SystemExit(
                    f"verify FAILED for {aoi}: {len(bad)}/{len(rows)} chips do not "
                    f"reconstruct (e.g. {bad[:3]}). The --min-area-m2 / --prefer-refined "
                    f"settings here differ from the ones {a.index} was built with; the "
                    "re-burned masks would not be a matched baseline."
                )
            print(f"verify {aoi}: {len(rows)}/{len(rows)} chips reconstruct exactly")

    if a.osm_only:
        keep = index.aoi.isin(list(OSM_AOIS)) | (index.sub_pixels == 0)
        print(f"--osm-only: dropping {int((~keep).sum())} chips with no source polygons")
        index = index[keep].reset_index(drop=True)

    # One burn per distinct source mask (the corpus oversamples pakistan 2x by duplicating
    # whole rows, which point at the same file).
    out_masks = out_root / "masks"
    todo = index[["aoi", "chip_id", "mask"]].drop_duplicates(["aoi", "chip_id"])
    remap: dict[tuple[str, str], str] = {}
    n_reburn = n_cast = n_kept = n_skip = 0
    for _, r in tqdm(list(todo.iterrows()), desc="masks"):
        key = (r.aoi, r.chip_id)
        if r.aoi not in labels and not fraction:
            # Binary arm: no source polygons, and no dtype change needed -- leave the chip
            # pointing at its original mask rather than making a byte-identical copy.
            remap[key] = r["mask"]
            n_kept += 1
            continue
        out_path = out_masks / str(r.aoi) / f"{r.chip_id}.tif"
        remap[key] = str(out_path.relative_to(ROOT)) if out_path.is_relative_to(ROOT) \
            else str(out_path)
        if out_path.exists():
            n_skip += 1
        elif r.aoi in labels:
            reburn_mask(Path(r["mask"]), out_path, labels[r.aoi],
                        settings.node_ignore_radius_m, fraction=fraction,
                        min_area=a.min_area_m2, factor=a.factor, all_touched=False)
            n_reburn += 1
        else:
            _cast_to_float(Path(r["mask"]), out_path)
            n_cast += 1
    print(f"masks: {n_reburn} re-burned from OSM polygons, {n_cast} cast to float32, "
          f"{n_kept} left pointing at the original, {n_skip} already present")

    # Repoint the index and recompute `sub_pixels`. It drops for re-burned chips (that is
    # the bias being removed) and so is NOT comparable across indices; for the fraction arm
    # it also changes meaning to "pixels with ANY coverage", which is why `sub_frac_sum`
    # carries the size-honest chip-level quantity there.
    new_mask, sub_pixels, frac_sum = [], [], []
    for _, r in tqdm(list(index.iterrows()), desc="index"):
        p = remap[(r.aoi, r.chip_id)]
        with rasterio.open(p) as src:
            band = src.read(1)
        new_mask.append(p)
        sub_pixels.append(int((band > 0).sum() if fraction else (band == 1).sum()))
        frac_sum.append(float(band[band > 0].sum()))
    out = index.assign(mask=new_mask, sub_pixels=sub_pixels)
    if fraction:
        out["sub_frac_sum"] = frac_sum

    out_path = out_root / "combined" / "index.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    n_pos = int((out.sub_pixels > 0).sum())
    src_px = int(index.sub_pixels.sum())
    print(f"combined: {len(out)} chips ({int((out.split == 'val').sum())} val, "
          f"{n_pos} positive [{100 * n_pos / len(out):.1f}%]) -> {out_path}")
    print(f"positive pixels: {src_px:,} -> {int(out.sub_pixels.sum()):,} "
          f"({100 * out.sub_pixels.sum() / max(src_px, 1) - 100:+.1f}%, the label inflation "
          "removed from the re-burnable chips)")


if __name__ == "__main__":
    main()
