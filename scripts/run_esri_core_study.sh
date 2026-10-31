#!/usr/bin/env bash
# ESRI core-label study runner (docs/issues/esri-core-labels.md).
#
#   scripts/run_esri_core_study.sh s1 "0 1 2"     # S1 study: train, infer, fuse, field_eval
#   scripts/run_esri_core_study.sh s2 "0 1"       # S2 study
#
# Launch under systemd so the harness cannot kill it (memory: feedback-background-chaining):
#   systemd-run --user --unit=subdetect-v17-s1 --working-directory=$PWD --setenv=HOME=$HOME \
#       -- /usr/bin/bash scripts/run_esri_core_study.sh s1 "0 1 2"
#
# Paired arms are trained back to back per seed (ctl then core), so if the run is cut short
# the finished seeds are still complete pairs. train.py resumes from last.ckpt, so a relaunch
# continues where it stopped; a finished run (EarlyStopping) re-exits within one epoch.
# The fixed partner arm for fusion: S1 study -> production S2 v12_up2 (sindh_test_v12);
# S2 study -> production S1 v5_s1only (sindh_test_v5).
set -u
STUDY=$1
SEEDS=$2
PIXI=/home/tobi/.pixi/bin/pixi
LOG=data/esri_core/run_${STUDY}.log
case $STUDY in
  s1) PARTNER=v12; UP=1 ;;
  s2) PARTNER=v5; UP=1 ;;
  *) echo "unknown study $STUDY"; exit 2 ;;
esac

best_ckpt() {  # newest top-k checkpoint by the monitored metric = the one ModelCheckpoint kept last
  "$PIXI" run -e ml python - "$1" <<'EOF'
import sys, torch
from pathlib import Path
d = Path(sys.argv[1])
cb = torch.load(d / "last.ckpt", map_location="cpu", weights_only=False)["callbacks"]
for k, v in cb.items():
    if k.startswith("ModelCheckpoint"):
        print(v["best_model_path"]); break
EOF
}

for s in $SEEDS; do
  for arm in ctl core; do
    cfg=configs/terramind_sub_v17_${STUDY}_${arm}_seed${s}.yaml
    echo "$(date '+%F %T') TRAIN $cfg" >> "$LOG"
    "$PIXI" run -e ml train --config "$cfg" >> "data/esri_core/train_v17_${STUDY}_${arm}_seed${s}.log" 2>&1
    echo "$(date '+%F %T') TRAIN exit $? $cfg" >> "$LOG"
  done
done

for s in $SEEDS; do
  for arm in ctl core; do
    name=v17_${STUDY}_${arm}_seed${s}
    ck=$(best_ckpt data/models/stageA_${name} | tail -1)
    echo "$(date '+%F %T') INFER $name <- $ck" >> "$LOG"
    "$PIXI" run -e ml python scripts/infer_sindh_test_ckpts.py infer --arm "${name}=${ck}:${UP}" >> "$LOG" 2>&1
    "$PIXI" run -e ml python scripts/infer_sindh_test_ckpts.py fuse --a "$name" --b "$PARTNER" --out "${name}_fused" >> "$LOG" 2>&1
    "$PIXI" run -e ml python scripts/field_eval.py --run-name "$name" \
        --prob-dir "data/osmose_regions/sindh_test_${name}/prob" --labels pakistan >> "$LOG" 2>&1
    "$PIXI" run -e ml python scripts/field_eval.py --run-name "${name}_fused" \
        --prob-dir "data/osmose_regions/sindh_test_${name}_fused/prob" --labels pakistan >> "$LOG" 2>&1
  done
done
"$PIXI" run -e ml python scripts/esri_core_ab.py --prefix "v17_${STUDY}" --seeds $SEEDS >> "$LOG" 2>&1
echo "$(date '+%F %T') DONE" >> "$LOG"
