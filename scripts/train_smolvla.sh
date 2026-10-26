#!/usr/bin/env bash
# Fine-tune SmolVLA (lerobot/smolvla_base) on the suction-pick demonstrations with LeRobot, on this machine's GPU.
#
#   scripts/train_smolvla.sh                                        # full run with the defaults below
#   STEPS=20 BATCH=8 DATASET=data/lerobot/test_small OUT=outputs/train/smoke scripts/train_smolvla.sh
#   scripts/train_smolvla.sh --wandb.enable=true                    # extra arguments go to lerobot-train
#
# The dataset keys (observation.images.camera1..3, observation.state, action) already match smolvla_base.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${LEROBOT_IMAGE:-pi-final-lerobot:latest}
HF_HOME_DIR=${HF_HOME:-$HOME/.cache/huggingface}
DATASET=${DATASET:-data/lerobot/pi_final_suction_pick}
OUT=${OUT:-outputs/train/smolvla_pick}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-64}
WORKERS=${WORKERS:-8}
SAVE_FREQ=${SAVE_FREQ:-2000}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" -f "$REPO/policy/lerobot.Dockerfile" "$REPO/policy"
fi

args=(
  --rm --runtime=nvidia --gpus all --ipc host
  --user "$(id -u):$(id -g)" -e HOME=/tmp
  -e HF_HOME=/hf -v "$HF_HOME_DIR:/hf"
  -v "$REPO:/workspace" -w /workspace
)
[[ -t 0 && -t 1 ]] && args+=(-it)
exec docker run "${args[@]}" "$IMAGE" lerobot-train \
  --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=local/pi_final_suction_pick --dataset.root="$DATASET" \
  --batch_size="$BATCH" --steps="$STEPS" --num_workers="$WORKERS" \
  --save_freq="$SAVE_FREQ" --log_freq=50 \
  --output_dir="$OUT" --job_name=smolvla_pick --wandb.enable=false "$@"
