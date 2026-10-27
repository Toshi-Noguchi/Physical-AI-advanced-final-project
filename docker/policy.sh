#!/usr/bin/env bash
# Serve a fine-tuned SmolVLA checkpoint to the simulator on 127.0.0.1:6123 (LeRobot container, GPU).
#
#   docker/policy.sh outputs/models/smolvla_pick     # a pretrained_model directory written by lerobot-train
#
# CONTAINER_NAME=... names the container (scripts/run_cell.sh uses it to stop the server).
set -euo pipefail

IMAGE=${LEROBOT_IMAGE:-pi-final-lerobot:latest}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HF_HOME_DIR=${HF_HOME:-$HOME/.cache/huggingface}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" -f "$REPO/policy/lerobot.Dockerfile" "$REPO/policy"
fi

args=(
  --rm --runtime=nvidia --gpus all --network host --ipc host
  --user "$(id -u):$(id -g)" -e HOME=/tmp
  -e HF_HOME=/hf -v "$HF_HOME_DIR:/hf"
  -e POLICY_PORT="${POLICY_PORT:-6123}" -e PYTHONPATH=/workspace/policy
  -v "$REPO:/workspace" -w /workspace
)
# No -it: a server needs no TTY, and with one a server started with `&` stops as soon as it touches the terminal.
[[ -n "${CONTAINER_NAME:-}" ]] && args+=(--name "$CONTAINER_NAME")
exec docker run "${args[@]}" "$IMAGE" python policy/smolvla_server.py "$@"
