#!/usr/bin/env bash
# The whole cell with both models: start the SmolVLA and Molmo2 servers, wait until both listen, run the line
# (scripts/run_cell.py) in Isaac Sim, and stop the servers when it ends. Arguments go to run_cell.py.
#
#   scripts/run_cell.sh --boxes 12 --record --out outputs/cell
#   CKPT=outputs/train/smolvla_pick/checkpoints/010000/pretrained_model scripts/run_cell.sh --boxes 4
#
# Needs about 25 GB of free GPU memory (Isaac Sim ~6, Molmo2-8B ~17-19, SmolVLA ~1-2). The GPU may be shared, so
# it refuses to start without some margin on top (NEED_MIB=... to change the bar). Server logs go to outputs/logs/.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CKPT=${CKPT:-outputs/models/smolvla_pick}
NEED_MIB=${NEED_MIB:-28000}
LOGS=$REPO/outputs/logs

[[ -f "$REPO/$CKPT/model.safetensors" ]] || { echo "no SmolVLA checkpoint in $CKPT (set CKPT=...)" >&2; exit 1; }
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if ((free < NEED_MIB)); then
  echo "only $free MiB of GPU memory free, the cell needs about $NEED_MIB MiB" >&2
  exit 1
fi

cleanup() { docker rm -f pi-final-smolvla pi-final-molmo2 >/dev/null 2>&1 || true; }
cleanup  # leftovers of an interrupted run would hold the container names and the ports
trap cleanup EXIT
trap 'exit 130' INT TERM

mkdir -p "$LOGS"
declare -A pid
CONTAINER_NAME=pi-final-smolvla "$REPO/docker/policy.sh" "$CKPT" > "$LOGS/smolvla_server.log" 2>&1 &
pid[smolvla]=$!
CONTAINER_NAME=pi-final-molmo2 "$REPO/docker/vlm.sh" policy/vlm_server.py > "$LOGS/molmo2_server.log" 2>&1 &
pid[molmo2]=$!

# A server only listens once its weights are on the GPU.
for name in smolvla molmo2; do
  echo "waiting for the $name server (outputs/logs/${name}_server.log)"
  until grep -q "listening on" "$LOGS/${name}_server.log"; do
    if ! kill -0 "${pid[$name]}" 2>/dev/null; then
      echo "the $name server exited:" >&2
      tail -n 20 "$LOGS/${name}_server.log" >&2
      exit 1
    fi
    sleep 5
  done
done

"$REPO/docker/run.sh" scripts/run_cell.py "$@"
