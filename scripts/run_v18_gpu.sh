#!/usr/bin/env bash
# v18 GPU side (docs/issues/v18-improvements.md): the datemax check, the gujarat_test baselines,
# then five trainings, each scored on both test regions as soon as the region is ready.
# Every GPU step takes earthpv's data/.gpu.lock (shared with earthpv and the v17 study).
# Steps leave markers in data/v18_markers/; training resumes from its own last.ckpt.
#
#   systemd-run --user --collect --unit=subdetect-v18-gpu -p WorkingDirectory="$PWD" \
#     -p LimitNOFILE=65536:65536 -p MemoryMax=14G bash scripts/run_v18_gpu.sh
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=data/v18_gpu.log
MARK=data/v18_markers
GPU_LOCK=${GPU_LOCK:-/run/media/tobi/aidisc/earthpv/data/.gpu.lock}
PY=.pixi/envs/default/bin/python
MLPY=.pixi/envs/ml/bin/python
V12=data/models/stageA_v12_up2_s2only/terramind-sub-epoch=24-step=51000.ckpt
V5=data/models/stageA_v5_s1only/terramind-sub-epoch=24-step=6975.ckpt
S2C0=data/models/stageA_v17_s2_ctl_seed0/terramind-sub-epoch=15-step=32640.ckpt
S2C1=data/models/stageA_v17_s2_ctl_seed1/terramind-sub-epoch=24-step=51000.ckpt
S1C0=data/models/stageA_v17_s1_ctl_seed0/terramind-sub-epoch=17-step=5022.ckpt
S1C1=data/models/stageA_v17_s1_ctl_seed1/terramind-sub-epoch=12-step=3627.ckpt
S1C2=data/models/stageA_v17_s1_ctl_seed2/terramind-sub-epoch=14-step=4185.ckpt
mkdir -p "$MARK"
export PYTHONUNBUFFERED=1
say() { echo "$(date '+%F %T') V18: $*" >> "$LOG"; }
gpu() { flock "$GPU_LOCK" "$@"; }
step() {  # step <marker> <cmd...>: run once, marker on success
  local m=$1; shift
  [ -e "$MARK/$m" ] && return 0
  say "start $m"
  "$@" >> "$LOG" 2>&1
  local rc=$?
  say "done $m rc=$rc"
  [ $rc -eq 0 ] && touch "$MARK/$m"
  return $rc
}
labels_of() { [ "$1" = sindh_test ] && echo pakistan || echo "$1"; }
evals() {  # evals <region> <name>...: field_eval each arm-alone or fused prob dir
  local region=$1; shift
  for n in "$@"; do
    step "eval_${region}_${n}" $MLPY scripts/field_eval.py --run-name "v18_${region%%_test}_${n}" \
      --prob-dir "data/osmose_regions/${region}_${n}/prob" --labels "$(labels_of "$region")"
  done
}
# Production arm-alone prob dirs that fusion partners read: sindh_test has v12 and v5 from
# earlier runs; gujarat_test gets them in step B.
score_arm() {  # score_arm <region> <arm> <ckpt> <mode> <partner>
  local region=$1 arm=$2 ck=$3 mode=$4 partner=$5
  step "infer_${region}_${arm}" gpu $MLPY scripts/infer_region_ckpts.py infer --region "$region" \
    --arm "${arm}=${ck}|${mode}|1" || return 1
  step "fuse_${region}_${arm}" $PY scripts/infer_region_ckpts.py fuse --region "$region" \
    --a "$arm" --b "$partner" --out "${arm}_x_${partner}" || return 1
  evals "$region" "$arm" "${arm}_x_${partner}"
}
TRAINED=()
score_all_ready() {  # every trained v18 arm on every region that is ready for it
  local entry arm mode partner
  for entry in "${TRAINED[@]}"; do
    IFS=: read -r arm mode partner <<< "$entry"
    if [ "$mode" = single ] || [ -e "$MARK/dates_sindh" ]; then
      score_arm sindh_test "$arm" "data/models/v18_${arm}/last.ckpt" "$mode" "$partner"
    fi
    [ -e "$MARK/B_gujarat" ] && \
      score_arm gujarat_test "$arm" "data/models/v18_${arm}/last.ckpt" "$mode" "$partner"
  done
}
all_scored() {  # every trained arm fused and evaluated in both regions, A and B done
  local entry arm mode partner r
  [ -e "$MARK/A_done" ] && [ -e "$MARK/B_gujarat" ] || return 1
  for entry in "${TRAINED[@]}"; do
    IFS=: read -r arm mode partner <<< "$entry"
    for r in sindh_test gujarat_test; do
      [ -e "$MARK/eval_${r}_${arm}_x_${partner}" ] || return 1
    done
  done
}
datemax_sindh() {  # step A, once sindh_test's date stacks exist
  [ -e "$MARK/dates_sindh" ] || return 0
  [ -e "$MARK/A_done" ] && return 0
  step A_infer gpu $MLPY scripts/infer_region_ckpts.py infer --region sindh_test \
    --arm "v12dmax=$V12|datemax|2" || return 1
  step A_fuse $PY scripts/infer_region_ckpts.py fuse --region sindh_test --a v12dmax --b v5 \
    --out v12dmax_x_v5 || return 1
  evals sindh_test v12dmax v12dmax_x_v5
  touch "$MARK/A_done"; say "datemax on sindh_test done"
}
catch_up() { datemax_sindh; gujarat_baselines; score_all_ready; }
gujarat_baselines() {  # step B, once gujarat_test's composites and date stacks exist
  [ -e "$MARK/dates_gujarat" ] || return 0
  [ -e "$MARK/B_gujarat" ] && return 0
  local R=gujarat_test
  step B_infer gpu $MLPY scripts/infer_region_ckpts.py infer --region $R \
    --arm "v12=$V12|single|2" --arm "v12dmax=$V12|datemax|2" --arm "v5=$V5|single|1" \
    --arm "v17s2c0=$S2C0|single|1" --arm "v17s2c1=$S2C1|single|1" \
    --arm "v17s1c0=$S1C0|single|1" --arm "v17s1c1=$S1C1|single|1" \
    --arm "v17s1c2=$S1C2|single|1" || return 1
  for a in v12 v12dmax v17s2c0 v17s2c1; do
    step "fuse_${R}_${a}" $PY scripts/infer_region_ckpts.py fuse --region $R --a $a --b v5 --out "${a}_x_v5"
    evals $R "$a" "${a}_x_v5"
  done
  for a in v17s1c0 v17s1c1 v17s1c2; do
    step "fuse_${R}_${a}" $PY scripts/infer_region_ckpts.py fuse --region $R --a $a --b v12 --out "${a}_x_v12"
    evals $R "$a" "${a}_x_v12"
  done
  evals $R v5
  touch "$MARK/B_gujarat"; say "gujarat_test baselines done"
}
train() {  # train <arm>: v22-recipe run to its fixed budget, 5 attempts (resumes last.ckpt)
  local arm=$1 cfg=configs/terramind_sub_v18_$1.yaml
  [ -e "$MARK/train_$arm" ] && return 0
  for attempt in 1 2 3 4 5; do
    say "start train_$arm (attempt $attempt)"
    gpu $MLPY -m subdetect.cli train --config "$cfg" >> "data/v18_train_${arm}.log" 2>&1 \
      && { touch "$MARK/train_$arm"; say "done train_$arm"; return 0; }
    say "train_$arm attempt $attempt failed"; sleep 120
  done
  say "FAILED train_$arm"; return 1
}

# Training does not wait for the network side: each arm trains when the GPU is free, and every
# region is scored as soon as its data exist (A: datemax on sindh_test, B: gujarat_test
# baselines, then each trained arm; the four-date arm only where date stacks exist).
for entry in s2_recipe:single:v5 s1_recipe:single:v12 s2_1d:single:v5 s2_4d:temporal:v5; do
  arm=${entry%%:*}
  if [ "$arm" = s1_recipe ] && [ ! -e "$MARK/index_s1_recipe" ]; then
    step index_s1_recipe $PY scripts/build_v18_indexes.py s1 --recipe-only || exit 1
  fi
  catch_up
  train "$arm" || exit 1
  TRAINED+=("$entry")
  catch_up
done

# S1 on the larger corpus, once the S1 chip fetch has finished.
until [ -e "$MARK/s1_fetch" ]; do catch_up; sleep 600; done
step index_s1_big $PY scripts/build_v18_indexes.py s1 || exit 1
train s1_big || exit 1
TRAINED+=("s1_big:single:v12")
until all_scored; do catch_up; all_scored || sleep 600; done
say "V18 DONE: data/eval_results/field_eval.csv (run names v18_*), docs/issues/v18-improvements.md"
