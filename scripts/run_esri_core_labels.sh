#!/usr/bin/env bash
# ESRI core-label study, Stage 0 production run (docs/issues/esri-core-labels.md):
# probe over every fetched crop -> core labels for both studies -> frozen-QC sheets for G0.4.
#
#   systemd-run --user --unit=subdetect-esri-labels --working-directory=$PWD --setenv=HOME=$HOME \
#       -- /usr/bin/bash scripts/run_esri_core_labels.sh
#
# THR / DILATE were fixed on the dev sample (2026-10-31), never on the QC sample:
#   THR 0.76 = held-out-site near-ring (10-60 m) false-positive rate of 5% for probe.pt
#   DILATE 0 = the 1 px Esri/S2 offset ring added +26% area to <6000 m2 subs on dev
# `apply` skips targets whose prob map exists, so a relaunch resumes.
set -u
PIXI=/home/tobi/.pixi/bin/pixi
C=data/esri_core
LOG=$C/labels_run.log
THR=0.76
DILATE=0
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

say "APPLY pk/india"
"$PIXI" run -e ml python scripts/esri_core_extract.py apply --aoi pakistan india_pilot pakistan_hardneg >> "$LOG" 2>&1 \
  || { say "APPLY pk/india FAILED"; exit 1; }

say "LABELS S1 corpus (chips_v5_honestburn -> chips_v5_core)"
"$PIXI" run -e ml python scripts/esri_core_labels.py --index data/chips_v5_honestburn/combined/index.parquet \
    --out-root data/chips_v5_core --aoi pakistan india_pilot pakistan_hardneg --thr $THR --dilate $DILATE >> "$LOG" 2>&1 \
  || { say "LABELS S1 FAILED"; exit 1; }

say "APPLY all (torchgeo)"
"$PIXI" run -e ml python scripts/esri_core_extract.py apply >> "$LOG" 2>&1 || { say "APPLY all FAILED"; exit 1; }

say "LABELS S2 corpus (chips_v16_honestburn -> chips_v17_core)"
"$PIXI" run -e ml python scripts/esri_core_labels.py --index data/chips_v16_honestburn/combined/index.parquet \
    --out-root data/chips_v17_core --thr $THR --dilate $DILATE >> "$LOG" 2>&1 \
  || { say "LABELS S2 FAILED"; exit 1; }

say "QC sheets (frozen qc_ids.txt, G0.4)"
"$PIXI" run -e ml python scripts/esri_core_labels.py --index data/chips_v16_honestburn/combined/index.parquet \
    --out-root $C/qc_labels --ids-only $C/qc_ids.txt --thr $THR --dilate $DILATE --preview $C/preview_qc >> "$LOG" 2>&1
"$PIXI" run -e ml python scripts/esri_core_qc.py --core $C/preview_qc --ids $C/qc_ids.txt --out $C/qc/g04 >> "$LOG" 2>&1
say "DONE"
