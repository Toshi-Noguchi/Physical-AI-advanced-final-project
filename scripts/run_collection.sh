#!/usr/bin/env bash
# Collect demonstrations with several Isaac Sim workers in parallel. Each worker runs its chunks one after
# another (a fresh process per chunk keeps the stage small: finished boxes are only parked, never deleted).
#
#   scripts/run_collection.sh data/raw/run1 3 4 42     # out, workers, chunks per worker, episodes per chunk
set -euo pipefail

OUT=${1:-data/raw/run1}
WORKERS=${2:-3}
CHUNKS=${3:-4}
EPISODES=${4:-42}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$REPO/$OUT/logs"

for ((w = 0; w < WORKERS; w++)); do
  (
    sleep $((w * 20))  # Kits that start in the same second share cache/log files; one of them once hung
    for ((c = 0; c < CHUNKS; c++)); do
      "$REPO/docker/run.sh" scripts/collect_demos.py --out "$OUT" --worker "$w" --chunk "$c" --episodes "$EPISODES" \
        2>&1 | grep --line-buffered -E '^\[collect\]|Traceback|Error:|RuntimeError' || true
    done
  ) > "$REPO/$OUT/logs/worker_$w.log" &
done
wait
echo "kept $(ls -d "$REPO/$OUT"/w*_c*_e* 2>/dev/null | wc -l) episodes in $OUT"
