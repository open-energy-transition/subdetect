#!/usr/bin/env bash
# v18 network side (docs/issues/v18-improvements.md): four-date stacks for sindh_test, the
# gujarat_test composites and their four-date stacks. No GPU. Markers in data/v18_markers/ tell
# scripts/run_v18_gpu.sh what is ready. The S1 chip fetch runs in its own unit
# (scripts/run_v18_s1fetch.sh), so the evaluation side is not queued behind it.
#
#   systemd-run --user --collect --unit=subdetect-v18-net -p WorkingDirectory="$PWD" \
#     -p MemoryMax=8G bash scripts/run_v18_net.sh
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=data/v18_net.log
MARK=data/v18_markers
PY=.pixi/envs/default/bin/python
mkdir -p "$MARK"
export PYTHONUNBUFFERED=1
say() { echo "$(date '+%F %T') NET: $*" >> "$LOG"; }

n_cells() { $PY -c "import json; print(len(json.load(open('data/osmose_regions/$1/cells.json'))['cells']))"; }
n_composed() { find -L "data/osmose_regions/$1/composites" -name composite_s1.tif 2>/dev/null | wc -l; }
n_dates() {  # stacks plus cells recorded as short of four clear dates
  local d=data/osmose_regions/$1/dates
  echo $(( $(ls "$d"/*.tif 2>/dev/null | wc -l) + $(cat "$d/fallback.txt" 2>/dev/null | wc -l) ))
}
n_region_cells() { ls -d "data/osmose_regions/$1/composites"/*/ 2>/dev/null | wc -l; }

# Passes repeat until a pass adds nothing (each command is resumable; a failed cell is retried).
until_done() {  # until_done <marker> <count-fn> <target> <cmd...>
  local m=$1 fn=$2 target=$3; shift 3
  [ -e "$MARK/$m" ] && { say "skip $m"; return 0; }
  local prev=-1 now
  for pass in $(seq 1 8); do
    now=$($fn "${@: -1}")
    say "$m pass $pass: $now of $target"
    [ "$now" -ge "$target" ] && break
    [ "$now" -eq "$prev" ] && { say "$m: a pass added nothing, stopping at $now of $target"; break; }
    prev=$now
    timeout 21600 "${@:1:$#-1}" --region "${@: -1}" >> "$LOG" 2>&1
  done
  touch "$MARK/$m"; say "done $m ($($fn "${@: -1}") of $target)"
}

# sindh_test's date stacks and gujarat_test's composites download side by side.
until_done dates_sindh n_dates "$(n_region_cells sindh_test)" \
  $PY scripts/infer_region_ckpts.py fetch-dates sindh_test &
until_done compose_gujarat n_composed "$(n_cells gujarat_test)" \
  $PY scripts/build_test_region.py compose --workers 2 gujarat_test
wait
[ -e data/labels/gujarat_test/substations_poly.parquet ] || \
  $PY scripts/build_test_region.py labels --region gujarat_test >> "$LOG" 2>&1
until_done dates_gujarat n_dates "$(n_composed gujarat_test)" \
  $PY scripts/infer_region_ckpts.py fetch-dates gujarat_test
say "NET DONE"
