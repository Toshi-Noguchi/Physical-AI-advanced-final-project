#!/usr/bin/env bash
# Convert raw demonstrations to a LeRobot v3.0 dataset inside the LeRobot container (built on first use).
#
#   docker/convert.sh data/raw/run1 --out data/lerobot/pi_final_suction_pick
set -euo pipefail

IMAGE=${LEROBOT_IMAGE:-pi-final-lerobot:latest}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" -f "$REPO/policy/lerobot.Dockerfile" "$REPO/policy"
fi

args=(--rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/tmp/hf -v "$REPO:/workspace" -w /workspace)
[[ -t 0 && -t 1 ]] && args+=(-it)
exec docker run "${args[@]}" "$IMAGE" python policy/to_lerobot.py "$@"
