"""Typer CLI: osm -> roi -> compose -> chips -> train -> evaluate -> infer -> postprocess -> export."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.command()
def osm(
    aoi: str = typer.Option(..., help="AOI name from configs/aoi.yaml (pakistan, india_pilot)"),
    out_dir: Path = typer.Option(Path("data/labels")),
) -> None:
    """Download a Geofabrik PBF and extract OSM power=substation/plant/line to GeoParquet."""
    from subdetect.osm import build_labels

    build_labels(aoi=aoi, out_dir=out_dir)


@app.command()
def refine_labels(
    aoi: str = typer.Option(..., help="AOI name (e.g. pakistan)"),
    labels_dir: Path = typer.Option(Path("data/labels")),
    composites_dir: Path = typer.Option(Path("data/composites")),
) -> None:
    """Shrink oversized substation polygons using S1 existence + S2 NDVI evidence."""
    from subdetect.label_refine import refine_labels as _refine

    _refine(aoi=aoi, labels_dir=labels_dir, composites_dir=composites_dir)


@app.command()
def roi(
    aoi: str = typer.Option(..., help="AOI name (e.g. pakistan)"),
    radius_km: float = typer.Option(20.0, help="ROI buffer radius around lines/substations"),
    min_voltage: float = typer.Option(0.0, help="Only lines with voltage >= this (V) seed the ROI"),
) -> None:
    """Compute the inference ROI cells (power-corridor + substation buffer) + a diagnostic PNG."""
    from subdetect.roi import run_roi

    run_roi(aoi=aoi, radius_km=radius_km, min_voltage=min_voltage)


@app.command()
def compose(
    aoi: str = typer.Option(..., help="AOI name"),
    sensor: str = typer.Option("s2", help="s2 (composite_0.tif) or s1 (composite_s1.tif)"),
    out_dir: Path = typer.Option(Path("data/composites")),
    radius_km: float = typer.Option(20.0, help="ROI radius (Pakistan; India uses max_train_cells)"),
    min_voltage: float = typer.Option(0.0),
    limit: int = typer.Option(0, help="Cap number of cells (0 = all; label cells first)"),
    workers: int = typer.Option(1, help="Concurrent cells (I/O-bound; 3-6 is a good range)"),
) -> None:
    """Build S2 (or S1) composites for an AOI's ROI cells (STAC, resumable)."""
    from subdetect.compose import run_compose

    run_compose(aoi=aoi, out_dir=out_dir, sensor=sensor, radius_km=radius_km,
                min_voltage=min_voltage, limit=limit, workers=workers)


@app.command()
def chips(
    aoi: str = typer.Option(..., help="AOI name"),
    labels_dir: Path = typer.Option(Path("data/labels")),
    out_dir: Path = typer.Option(Path("data/chips")),
    limit: int = typer.Option(0, help="Cap number of chips (0 = no cap; for smoke tests)"),
    s1: bool = typer.Option(False, "--s1", help="Also write co-registered S1 chips (needs S1 composites)"),
    min_area_m2: float = typer.Option(
        None, help="Override the label floor for training masks (0 = no size floor)"),
    prefer_refined: bool = typer.Option(
        True, help="Use substations_poly_refined.parquet when present (--no-prefer-refined forces raw labels)"),
    voltage_only: bool = typer.Option(
        False, help="Only train on substations with a valid OSM voltage tag"),
    voltage_weight: int = typer.Option(
        1, help="Oversample voltage-tagged substations by this factor (chip-level, keeps untagged ones)"),
) -> None:
    """Sample training chips: composite windows + burned substation masks."""
    from subdetect.chips import build_chips

    build_chips(aoi=aoi, labels_dir=labels_dir, out_dir=out_dir, limit=limit, with_s1=s1,
                min_area_m2=min_area_m2, prefer_refined=prefer_refined, voltage_only=voltage_only,
                voltage_weight=voltage_weight)


@app.command()
def train(
    config: Path = typer.Option(Path("configs/terramind_sub.yaml")),
    smoke: bool = typer.Option(False, help="50-step smoke run"),
) -> None:
    """Fine-tune TerraMind for substation segmentation via TerraTorch."""
    from subdetect.train import run_training

    run_training(config=config, smoke=smoke)


@app.command()
def evaluate(
    aoi: str = typer.Option("pakistan", help="AOI with a val split"),
    checkpoint: Path = typer.Option(..., help="Trained model checkpoint"),
    chips_dir: Path = typer.Option(Path("data/chips")),
    threshold: float = typer.Option(0.3),
    min_area_m2: float = typer.Option(
        None, help="Override the label floor for ground-truth installations (0 = all)"),
) -> None:
    """Report pixel IoU/F1 and per-installation recall by area + voltage."""
    from subdetect.evaluate import evaluate as _eval

    _eval(aoi=aoi, checkpoint=checkpoint, chips_dir=chips_dir, threshold=threshold,
          min_area_m2=min_area_m2)


@app.command()
def infer(
    aoi: str = typer.Option(..., help="AOI name (e.g. pakistan)"),
    checkpoint: Path = typer.Option(..., help="Trained model checkpoint"),
    out_dir: Path = typer.Option(Path("data/predictions")),
    limit: int = typer.Option(0, help="Cap number of cells (0 = all)"),
    upsample: int = typer.Option(
        1, help="Must match the checkpoint's training-time data.upsample (e.g. 2 for v12_up2)"),
) -> None:
    """Tiled inference over an AOI, writing probability GeoTIFFs per cell."""
    from subdetect.infer import run_inference

    run_inference(aoi=aoi, checkpoint=checkpoint, out_dir=out_dir, limit=limit, upsample=upsample)


@app.command()
def postprocess(
    aoi: str = typer.Option(...),
    pred_dir: Path = typer.Option(Path("data/predictions")),
    threshold: float = typer.Option(0.3, help="Probability threshold (recall-oriented)"),
) -> None:
    """Threshold, polygonize, add grid-proximity prior + known/new status."""
    from subdetect.postprocess import run_postprocess

    run_postprocess(aoi=aoi, pred_dir=pred_dir, threshold=threshold)


@app.command()
def export(
    aoi: str = typer.Option(...),
    pred_dir: Path = typer.Option(Path("data/predictions")),
) -> None:
    """Export candidates as GeoParquet/GeoJSON + MapRoulette challenge (new leads)."""
    from subdetect.export import run_export

    run_export(aoi=aoi, pred_dir=pred_dir)


if __name__ == "__main__":
    app()
