#!/usr/bin/env bash
# v18: S1 RTC chips for every v16 chip without S1 (TorchGeo, settlement and global hard
# negatives), scripts/fetch_s1_chips.py, in wall-clock-capped passes (a long odc/dask run
# degrades like compose does) until a pass adds nothing. Then the index and the s1_fetch marker
# that scripts/run_v18_gpu.sh waits for. Network only.
#
#   systemd-run --user --collect --unit=subdetect-v18-s1fetch -p WorkingDirectory="$PWD" \
#     -p MemoryMax=8G -E MAX_ITEMS=8 bash scripts/run_v18_s1fetch.sh
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=data/v18_s1fetch.log
MARK=data/v18_markers
PY=.pixi/envs/default/bin/python
MAX_ITEMS=${MAX_ITEMS:-8}
# Hard negatives (2,596) first, then TorchGeo in a fixed random order, up to TARGET chips: at
# ~80 s a chip on the shared link the full 29,092 would take about two weeks (2026-11-06).
TARGET=${TARGET:-8600}
WORKERS=${WORKERS:-2}
mkdir -p "$MARK"
export PYTHONUNBUFFERED=1
say() { echo "$(date '+%F %T') S1FETCH: $*" >> "$LOG"; }
count() { find data/chips_s1_extra/s1 -name '*.tif' 2>/dev/null | wc -l; }
prev=-1
for pass in $(seq 1 60); do
  now=$(count)
  say "pass $pass: $now chips written"
  [ "$now" -ge "$TARGET" ] && { say "reached $TARGET"; break; }
  [ "$now" -eq "$prev" ] && { say "a pass added nothing; stopping"; break; }
  prev=$now
  timeout 7200 $PY scripts/fetch_s1_chips.py fetch --workers "$WORKERS" --max-items "$MAX_ITEMS" \
    --limit $((TARGET - now)) >> "$LOG" 2>&1
done
$PY scripts/fetch_s1_chips.py index >> "$LOG" 2>&1 && touch "$MARK/s1_fetch"
say "S1 FETCH DONE ($(count) chips)"
