#!/usr/bin/env bash
# Run a script in the inspection-VLM container (Molmo2), built on first use; weights come from the host's
# Hugging Face cache.
#
#   docker/vlm.sh policy/eval_inspection.py data/inspection/eval1
#
# CONTAINER_NAME=... names the container (scripts/run_cell.sh uses it to stop the server).
set -euo pipefail

IMAGE=${VLM_IMAGE:-pi-final-vlm:latest}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HF_HOME_DIR=${HF_HOME:-$HOME/.cache/huggingface}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" -f "$REPO/policy/vlm.Dockerfile" "$REPO/policy"
fi

args=(
  --rm --runtime=nvidia --gpus all --network host --ipc host
  --user "$(id -u):$(id -g)" -e HOME=/tmp
  -e HF_HOME=/hf -v "$HF_HOME_DIR:/hf"
  -e PYTHONPATH=/workspace/policy
  -v "$REPO:/workspace" -w /workspace
)
[[ -t 0 && -t 1 ]] && args+=(-it)
[[ -n "${CONTAINER_NAME:-}" ]] && args+=(--name "$CONTAINER_NAME")
exec docker run "${args[@]}" "$IMAGE" python "$@"
