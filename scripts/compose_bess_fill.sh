#!/usr/bin/env bash
# Imagery fill for the BESS feasibility study (docs/issues/bess-detectability-feasibility.md).
#
# Order matters: S1, DEM and s2pre all pin themselves to the cell's composite_0.tif
# GeoBox, and compose.py's per-cell worker returns early (no fetch, no error) when that
# base layer is absent. So S2 must complete for an AOI before anything else runs on it.
#
# Every stage skips existing outputs, so this is kill-and-rerun safe. Launch under
# systemd-run --user with an explicit PATH: without it osmium is not on the unit's PATH,
# and a minimal environment does not inherit pixi either.
set -u
cd /run/media/tobi/aidisc/subdetect
PIXI=/home/tobi/.pixi/bin/pixi
WORKERS=${WORKERS:-5}
AOIS="gb_bess ca_bess tx_bess"

# Pre-flight: the composites live on / via per-AOI symlinks inside the aidata tree, and
# aidata itself has under 10 GB free. Refuse to start if the real target is short.
free_gb=$(df -BG --output=avail /home/tobi/subdetect-data | tail -1 | tr -dc '0-9')
if [ "${free_gb:-0}" -lt 20 ]; then
  echo "FATAL: only ${free_gb} GB free on the composite target; need >= 20" >&2
  exit 1
fi
for a in $AOIS; do
  real=$(readlink -f "data/composites/$a")
  case "$real" in
    /home/tobi/subdetect-data/*) ;;
    *) echo "FATAL: data/composites/$a resolves to $real, not the out-of-tree root" >&2
       exit 1 ;;
  esac
done
echo "pre-flight OK: ${free_gb} GB free, all three AOI symlinks resolve out of tree"

# s2pre is DEFERRED by default. It feeds only the bi-temporal change group, which is the
# single pre-declared G2.3 extension -- read only if Tier 2 lands in the 0.65-0.75 dead
# band. It costs a second full S2 pass (measured 38 s/cell over 389 cells, so ~4 h), and
# buying conditional data before knowing whether the condition holds is speculative.
# Run `SENSORS=s2pre scripts/compose_bess_fill.sh` if and when the dead band is hit.
for sensor in ${SENSORS:-s2 s1 dem}; do
  for a in $AOIS; do
    r=${a%%_bess}
    cells="data/bess_cells_${r}.txt"
    [ -s "$cells" ] || { echo "SKIP $a $sensor: no $cells"; continue; }
    echo "=== $(date -Is)  $a  $sensor  ($(wc -l < "$cells") cells) ==="
    "$PIXI" run compose --aoi "$a" --sensor "$sensor" \
        --cells-file "$cells" --workers "$WORKERS" || echo "WARN: $a $sensor exited $?"
  done
done
echo "=== $(date -Is) fill complete ==="
for a in $AOIS; do
  d="data/composites/$a/composites"
  printf "%-9s S2 %4d  S1 %4d  DEM %4d  PRE %4d\n" "$a" \
    "$(ls "$d"/*/composite_0.tif 2>/dev/null | wc -l)" \
    "$(ls "$d"/*/composite_s1.tif 2>/dev/null | wc -l)" \
    "$(ls "$d"/*/composite_dem.tif 2>/dev/null | wc -l)" \
    "$(ls "$d"/*/composite_pre.tif 2>/dev/null | wc -l)"
done
